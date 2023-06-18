import torch.nn as nn
from einops import rearrange
import torch
from flash_attn.flash_attention import FlashAttention
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func

class FlashCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                    (default: 1/sqrt(d_keys) where d_keys is computed at
                    runtime)
        attention_dropout: The dropout rate to apply to the attention
                        (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                triton=False):
        super().__init__()
        if attention_dropout != 0.0 or not triton:
            assert flash_attn_unpadded_kvpacked_func is not None, 'FlashAttention is not installed'

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)


    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None, key_padding_mask=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            ), None
        else:
            if key_padding_mask is None:
                batch_size, seqlen_q = q.shape[0], q.shape[1]
                seqlen_k = kv.shape[1]
                assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]
                q = rearrange(q, 'b s ... -> (b s) ...')
                kv = rearrange(kv, 'b s ... -> (b s) ...')
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q,
                                            dtype=torch.int32, device=q.device)
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k,
                                            dtype=torch.int32, device=kv.device)
                output = flash_attn_unpadded_kvpacked_func(
                    q, kv, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
                return output, None
            else:
                nheads = kv.shape[-2]
                batch_size, seqlen_q = q.shape[0], q.shape[1]
                seqlen_k = kv.shape[1]
                assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]
                #q = rearrange(q, 'b s ... -> (b s) ...')
                kv = rearrange(kv, 'b s ... -> (b s) ...')
                q = rearrange(q, 'b s h d -> b s (h d)')
                q_unpad, indices, cu_seqlens_q, seqlen_q_max = unpad_input(q, key_padding_mask)
                q_unpad = rearrange(q_unpad, 'nnz (h d) -> nnz h d', h=nheads)
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k,
                                            dtype=torch.int32, device=kv.device)
                output_unpad = flash_attn_unpadded_kvpacked_func(
                    q_unpad, kv, cu_seqlens_q, cu_seqlens_k, seqlen_q_max, seqlen_k,
                    self.drop.p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen_q),
                                'b s (h d) -> b s h d', h=nheads)
                return output, None

    
class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        #self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, q, k, v, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        qkv = torch.cat([q, k, v], dim=2)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
    
class FlashCrossMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        #self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashCrossAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        kv = torch.cat([k, v], dim=2)
        kv = rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.num_heads)
            
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
    
    
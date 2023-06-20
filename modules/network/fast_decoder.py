from typing import Optional
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from fast_attention import FlashMHA, FlashCrossMHA


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = FlashMHA(embed_dim=d_model, num_heads=nhead, attention_dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query,
                     query_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, k, v=query, 
                              key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout(query2)
        query = self.norm(query)

        return query

    def forward_pre(self, query,
                    query_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        query2 = self.norm(query)
        q = k = self.with_pos_embed(query2, query_pos)
        query2 = self.self_attn(q, k, v=query2,
                              key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout(query2)
        
        return query

    def forward(self, query,
                query_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(query,
                                    query_key_padding_mask, query_pos)
        return self.forward_post(query,
                                 query_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = FlashCrossMHA(embed_dim=d_model, num_heads=nhead, attention_dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, kv,
                     query_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        query2 = self.multihead_attn(q=self.with_pos_embed(query, query_pos),
                                   k=self.with_pos_embed(kv, pos),
                                   v=kv,
                                   key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout(query2)
        query = self.norm(query)
        
        return query

    def forward_pre(self, query, kv,
                    query_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        query2 = self.norm(query)
        query2 = self.multihead_attn(q=self.with_pos_embed(query2, query_pos),
                                   k=self.with_pos_embed(kv, pos),
                                   v=kv,
                                   key_padding_mask=query_key_padding_mask)[0]
        query = query + self.dropout(query2)

        return query

    def forward(self, query, kv,
                query_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(query, kv,
                                    query_key_padding_mask, pos, query_pos)
        return self.forward_post(query, kv,
                                 query_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Architechture_1(nn.Module):
    def __init__(self, dmodel, nhead: dict, depth: dict, dropout, activation="gelu", normalize_before=True) -> None:
        super().__init__()
        assert len(nhead["self"]) == depth["self"]
        assert len(nhead["cross"]) == depth["cross"]
        assert all(dmodel % np.array(nhead["self"]) == 0)
        assert all(dmodel % np.array(nhead["cross"]) == 0)
        self.dmodel = dmodel
        self.nhead = nhead
        self.depth = depth
        self.normalize_before = normalize_before
        self.self_attn = nn.ModuleList([SelfAttentionLayer(dmodel, 
                                                           nhead["self"][i], 
                                                           dropout,
                                                           activation, 
                                                           normalize_before) for i in range(depth["self"])])
        self.cross_attn = nn.ModuleList([CrossAttentionLayer(dmodel,
                                                             nhead["cross"][i],
                                                             dropout,
                                                             activation,
                                                             normalize_before) for i in range(depth["cross"])])
        self.self_fnn = nn.ModuleList([FFNLayer(dmodel, 2048, dropout, activation, normalize_before) for _ in range(depth["self"])])
        self.cross_fnn = nn.ModuleList([FFNLayer(dmodel, 2048, dropout, activation, normalize_before) for _ in range(depth["cross"])])
                                                
        
        
    def forward(self, x, kv, pos=None, query_pos=None, query_key_padding_mask=None):
        for attn, fnn in zip(self.cross_attn, self.cross_fnn):
            x = attn(x, kv, query_key_padding_mask, pos, query_pos)
            x = fnn(x)
        for attn, fnn in zip(self.self_attn, self.self_fnn):
            x = attn(x, query_key_padding_mask, query_pos)
            x = fnn(x)
        return x
    
class Architechture_1(nn.Module):
    def __init__(self, dmodel, nhead: dict, depth: int, dropout, activation="gelu", normalize_before=True) -> None:
        super().__init__()
        assert len(nhead["self"]) == depth
        assert len(nhead["cross"]) == depth
        assert all(dmodel % np.array(nhead["self"]) == 0)
        assert all(dmodel % np.array(nhead["cross"]) == 0)
        self.dmodel = dmodel
        self.nhead = nhead
        self.depth = depth
        self.normalize_before = normalize_before
        self.self_attn = nn.ModuleList([SelfAttentionLayer(dmodel, 
                                                           nhead["self"][i], 
                                                           dropout,
                                                           activation, 
                                                           normalize_before) for i in range(depth["self"])])
        self.cross_attn = nn.ModuleList([CrossAttentionLayer(dmodel,
                                                             nhead["cross"][i],
                                                             dropout,
                                                             activation,
                                                             normalize_before) for i in range(depth["cross"])])
        self.fnn = nn.ModuleList([FFNLayer(dmodel, 2048, dropout, activation, normalize_before) for _ in range(depth["self"])])
        
    def forward(self, x, kv, pos=None, query_pos=None, query_key_padding_mask=None):
        for i in range(self.depth):
            x = self.cross_attn[i](x, kv, query_key_padding_mask, pos, query_pos)
            x = self.self_attn[i](x, query_key_padding_mask, query_pos)
            x = self.fnn[i](x)
        return x
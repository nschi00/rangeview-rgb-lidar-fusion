import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())
sys.path.append("modules")
from overfit_test import overfit_test
from torch.nn import functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from ResNet import BasicBlock, BasicConv2d, conv1x1
import torch
import torch.nn as nn
import math
from mix_transformer import Block, OverlapPatchEmbed, Mlp, DropPath
from segformer_head import SegFormerHead
from functools import partial

""" This is a torch way of getting the intermediate layers
 return_layers = {"layer4": "out"}
 if aux:
      return_layers["layer3"] = "aux"
 backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
"""

class Fusion(nn.Module):
    """
    All scale fusion with resnet50 backbone
    Basic fusion with torch.cat + conv1x1
    use_att: use cross attention or not
    fusion_scale: "all_early" or "all_late" or "main_late" or "main_early"
    name_backbone: backbone for RGB, only "resnet50" at the moment possible
    branch_type: semantic, instance or panoptic
    stage: whether to only use the enc, pixel_decoder or combined pixel/transformer decoder output (combination)
    """

    def __init__(self, nclasses, aux=True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64):

        super(Fusion, self).__init__()

        self.inplanes = 64

        self.initial_conv_lidar = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, self.inplanes, kernel_size=3, padding=1))
        
        self.initial_conv_rgb = nn.Sequential(BasicConv2d(3, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, self.inplanes, kernel_size=3, padding=1))

        # self.semantic_output = nn.Conv2d(128, nclasses, 1)

        # if self.aux:
        #     self.aux_heads = nn.ModuleDict()
        #     for i in range(2, 5):
        #         self.aux_heads["layer{}".format(i)] = nn.Conv2d(128, nclasses, 1)

        self.segfusion = SegFusion(img_size=[64, 192], patch_size=4, in_chans=self.inplanes, num_classes=nclasses,
                                      embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                                      mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.1,
                                      attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                      depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        

    def forward(self, x, rgb):
        # * get projection features
        x_lidar = self.initial_conv_lidar(x)
        x_rgb = self.initial_conv_rgb(rgb)

        out = self.segfusion(x_lidar, x_rgb)

        out = F.softmax(out, dim=1)

        # if self.aux:
        #     out = [out]
        #     for i in range(2, 5):
        #         out.append(self.aux_heads["layer{}".format(i)](x_lidar_features[i]))
        #         out[-1] = F.softmax(out[-1], dim=1)

        return out


class SegFusion(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        in_channels_head = (torch.Tensor(embed_dims)*2).int().tolist()
        def divide(tuple_data, divisor):
            result = tuple(elem // divisor for elem in tuple_data)
            return result
        # patch_embed
        self.patch_embed_lidar1 = OverlapPatchEmbed(img_size=img_size,
                                              patch_size=patch_size,
                                              stride=1,
                                              in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed_lidar2 = OverlapPatchEmbed(img_size=divide(img_size, 2),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed_lidar3 = OverlapPatchEmbed(img_size=divide(img_size, 4),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed_lidar4 = OverlapPatchEmbed(img_size=divide(img_size, 8),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        self.patch_embed_rgb1 = OverlapPatchEmbed(img_size=img_size,
                                              patch_size=patch_size,
                                              stride=1,
                                              in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed_rgb2 = OverlapPatchEmbed(img_size=divide(img_size, 2),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed_rgb3 = OverlapPatchEmbed(img_size=divide(img_size, 4),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed_rgb4 = OverlapPatchEmbed(img_size=divide(img_size, 8),
                                              patch_size=patch_size,
                                              stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block_lidar1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.cross_block_lidar1 = CrossAttentionBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[0]], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
        self.norm_lidar1 = norm_layer(embed_dims[0])

        self.block_rgb1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.cross_block_rgb1 = CrossAttentionBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[0] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
        self.norm_rgb1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block_lidar2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.cross_block_lidar2 = CrossAttentionBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[1] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
        self.norm_lidar2 = norm_layer(embed_dims[1])

        self.block_rgb2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.cross_block_rgb2 = CrossAttentionBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[1] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
        self.norm_rgb2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block_lidar3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.cross_block_lidar3 = CrossAttentionBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[2] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
        self.norm_lidar3 = norm_layer(embed_dims[2])

        self.block_rgb3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.cross_block_rgb3 = CrossAttentionBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[2] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
        self.norm_rgb3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block_lidar4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.cross_block_lidar4 = CrossAttentionBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[3] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
        self.norm_lidar4 = norm_layer(embed_dims[3])

        self.block_rgb4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.cross_block_rgb4 = CrossAttentionBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + depths[3] - 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
        self.norm_rgb4 = norm_layer(embed_dims[3])

        self.head = SegFormerHead(embedding_dim=512, in_channels_head=in_channels_head, img_size=img_size, num_classes=self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x, y):
        assert(x.shape == y.shape)
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed_lidar1(x)
        y, _, _ = self.patch_embed_rgb1(y)
        x_temp = x
        y_temp = y
        for i, blk in enumerate(self.block_lidar1):
            x = blk(x, H, W)
        x = self.cross_block_lidar1(x, y_temp, H, W)
        x = self.norm_lidar1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for i, blk in enumerate(self.block_rgb1):
            y = blk(y, H, W)
        y = self.cross_block_rgb1(y, x_temp, H, W)
        y = self.norm_rgb1(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(torch.cat((x, y), dim=1))

        # stage 2
        x, H, W = self.patch_embed_lidar2(x)
        y, _, _ = self.patch_embed_rgb2(y)
        x_temp = x
        y_temp = y
        for i, blk in enumerate(self.block_lidar2):
            x = blk(x, H, W)
        x = self.cross_block_lidar2(x, y_temp, H, W)
        x = self.norm_lidar2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for i, blk in enumerate(self.block_rgb2):
            y = blk(y, H, W)
        y = self.cross_block_rgb2(y, x_temp, H, W)
        y = self.norm_rgb2(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(torch.cat((x, y), dim=1))

        # stage 3
        x, H, W = self.patch_embed_lidar3(x)
        y, _, _ = self.patch_embed_rgb3(y)
        x_temp = x
        y_temp = y
        for i, blk in enumerate(self.block_lidar3):
            x = blk(x, H, W)
        x = self.cross_block_lidar3(x, y_temp, H, W)
        x = self.norm_lidar3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for i, blk in enumerate(self.block_rgb3):
            y = blk(y, H, W)
        y = self.cross_block_rgb3(y, x_temp, H, W)
        y = self.norm_rgb3(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(torch.cat((x, y), dim=1))

        # stage 4
        x, H, W = self.patch_embed_lidar4(x)
        y, _, _ = self.patch_embed_rgb4(y)
        x_temp = x
        y_temp = y
        for i, blk in enumerate(self.block_lidar4):
            x = blk(x, H, W)
        x = self.cross_block_lidar4(x, y_temp, H, W)
        x = self.norm_lidar4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for i, blk in enumerate(self.block_rgb4):
            y = blk(y, H, W)
        y = self.cross_block_rgb4(y, x_temp, H, W)
        y = self.norm_rgb4(y)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(torch.cat((x, y), dim=1))

        return outs

    def forward(self, x, y):
        x = self.forward_features(x, y)
        x = self.head(x)

        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H, W):
        B, N, C = x.shape
        assert(x.shape == y.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm_x = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H, W):
        x = x + self.drop_path(self.attn(self.norm_x(x), self.norm_y(y), H, W))
        x = x + self.drop_path(self.mlp(self.norm(x), H, W))

        return x

if __name__ == "__main__":
    model = Fusion(20, False).cuda()
    overfit_test(model, 1, (3, 64, 192), (5, 64, 192))
import sys
import os
import math
import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
from overfit_test import overfit_test
from mix_transformer import MixVisionTransformer, Mlp
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from timm.models.layers import trunc_normal_
from copy import deepcopy

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.GELU()

    def forward(self, x):
        if type(x) == tuple or type(x) == list:
            x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class RangeFormer(nn.Module):
    def __init__(self, n_classes, img_size: tuple) -> None:
        super().__init__()
        self.initial_conv = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                            BasicConv2d(64, 128, kernel_size=3, padding=1),
                                            BasicConv2d(128, 128, kernel_size=3, padding=1))
        self.model = MixVisionTransformer(img_size=img_size, patch_size=3, in_chans=128)
        self.end_mlp = nn.Sequential(BasicConv2d(128*4, 128, kernel_size=3, padding=1),
                                     BasicConv2d(128, 20, kernel_size=3, padding=1))
        self.unify_mlp = nn.ModuleList([BasicConv2d(64, 128, kernel_size=3, padding=1),
                                        BasicConv2d(128, 128, kernel_size=3, padding=1),
                                        BasicConv2d(256, 128, kernel_size=3, padding=1),
                                        BasicConv2d(512, 128, kernel_size=3, padding=1)])
        self.aux_mlp = nn.ModuleList([BasicConv2d(128, 20, kernel_size=1),
                                      BasicConv2d(128, 20, kernel_size=1),
                                      BasicConv2d(128, 20, kernel_size=1),
                                      BasicConv2d(128, 20, kernel_size=1)])

    def forward(self, lidar):
        B, _, H, W = lidar.shape
        lidar_feats = self.initial_conv(lidar)
        lidar_atts = self.model(lidar_feats)
        for i, att in enumerate(lidar_atts):
            lidar_atts[i] = F.interpolate(self.unify_mlp[i](att), (H,W), mode='bilinear', align_corners=True)

        out = []
        final_pred = F.softmax(self.end_mlp(torch.cat(lidar_atts, dim=1)), dim=1)
        out.append(final_pred)
        for i, att in enumerate(lidar_atts):
            aux_out = self.aux_mlp[i](att)
            out.append(F.softmax(aux_out, dim = 1))

        return out

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

class Cross_Attention(nn.Module):
    def __init__(self,
                 in_channels = [256, 512, 1024],
                 hidden_dim=256,
                 n_head=8) -> None:
        super().__init__()
        self.quere_pe = PositionalEncoding1D(hidden_dim)
        self.num_feature_levels = 3

        self.input_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                self.input_proj.append(nn.Conv2d(in_channel, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
            self.decoder.append(nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True))
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(self, A, B): # A: query, B: key, value
        for i in range(len(self.input_proj)):
            B_in = B["feat" + str(i + 1)]
            B_in = self.input_proj[i](B_in)
            B_in = B_in.view(B_in.shape[0], -1, B_in.shape[1]) + self.level_embed.weight[i][None, None, :]
            A = A + self.quere_pe(A)
            A = self.decoder[i](A, B_in, B_in)[0]
        return A

class Self_Attention(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 n_head=8,
                 depth=1) -> None:
        super().__init__()
        self.quere_pe = PositionalEncoding1D(hidden_dim)
        self.decoder = nn.ModuleList()
        for _ in range(depth):
            self.decoder.append(nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True))

    def forward(self, A): # A: query, B: key, value
        for layer in self.decoder:
            A = A + self.quere_pe(A)
            A = layer(A, A, A)[0]
        return A

class Attention_Fusion(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 in_channels = [256, 512, 1024],
                 n_head=8,
                 depth=3) -> None:
        super().__init__()
        fusion_block = nn.ModuleList([Cross_Attention(hidden_dim=hidden_dim,
                                                      n_head=n_head,
                                                      in_channels=in_channels),
                                      Self_Attention(hidden_dim=hidden_dim, n_head=n_head)])
        self.fusion_blocks = nn.ModuleList([deepcopy(fusion_block) for _ in range(depth)])

    def forward(self, A, B): # A: query, B: key, value
        A = A.view(A.shape[0], -1, A.shape[1])
        out = []
        for blk in self.fusion_blocks:
            A = blk[0](A, B) # * fusion attention
            A = blk[1](A)    # * self attention
            out.append(A)
        return out

if __name__ == "__main__":
    model = RangeFormer(20, img_size=(64, 512))
    overfit_test(model, 6, None, (5, 64, 512))
    pass
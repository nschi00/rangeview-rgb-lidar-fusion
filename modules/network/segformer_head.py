# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
#from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
import warnings

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768, reshape=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.reshape = reshape

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.act(x)
        if self.reshape:
            x = x.transpose(1, 2).view(b, -1, h, w)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, embedding_dim=256, in_channels_head=[32, 64, 128, 256], num_classes=20, img_size=224):
        super().__init__()
        self.img_size = img_size
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels_head

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='SyncBN', requires_grad=True)
        # )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_2 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_3 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_4 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=self.img_size, mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=self.img_size, mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=self.img_size, mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resize(_c1, size=self.img_size, mode='bilinear', align_corners=False)

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # x = self.dropout(_c)
        _c4 = F.softmax(self.linear_pred_4(_c4), dim=1)
        _c3 = F.softmax(self.linear_pred_3(_c3), dim=1)
        _c2 = F.softmax(self.linear_pred_2(_c2), dim=1)
        _c1 = F.softmax(self.linear_pred_1(_c1), dim=1)
        x = F.softmax(self.linear_pred(x), dim=1)

        return [x , _c1, _c2, _c3, _c4]
    

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
import sys
import os
import math
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
from overfit_test import overfit_test
from torch.nn import functional as F
from ResNet import ResNet_34
from timm.models.layers import trunc_normal_
return_layers = {"layer3": "out"}


class CENet(nn.Module):
    def __init__(self, n_classes, d_model=128) -> None:
        super().__init__()
        self.lidar_model = ResNet_34(n_classes, aux=True)
        self.feat_3d_red = BasicConv2d(256, d_model, kernel_size=3, padding=1)
        self.prediction = BasicConv2d(d_model, n_classes, kernel_size=1, padding=0)
        
    def forward(self, lidar, rgb):

        # with torch.no_grad():
        B, _, H, W = lidar.shape
        lidar_out = self.lidar_model(lidar, rgb)
        lidar_feature_high_dim = self.lidar_model.feature_3D
        lidar_feature = self.feat_3d_red(lidar_feature_high_dim)
        
        fused_pred = F.softmax(self.prediction(lidar_feature), dim=1)
        out = [fused_pred] + lidar_out
        return out, lidar_feature_high_dim

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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        if type(x) == tuple or type(x) == list:
            x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


if __name__ == "__main__":
    model = CENet(20)
    overfit_test(model, 6, (3, 256, 768), (5, 64, 512))
    pass
import sys
import os
import math
import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
from position_encoding import PositionEmbeddingSine
from overfit_test import overfit_test
from torch.nn import functional as F
from ResNet import ResNet_34
from RangeFormer import BasicConv2d
from timm.models.layers import trunc_normal_
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from fast_decoder import Architechture_1
return_layers = {"layer3": "out"}


class Fusion(nn.Module):
    def __init__(self, n_classes, d_model=128, depth=[4,4], n_queries=6500) -> None:
        super().__init__()
        assert type(depth) == list and len(depth) == 2
        self.lidar_model = ResNet_34(n_classes, aux=True)
        self.n_queries = n_queries
        rgb_backbone = resnet50(weights="IMAGENET1K_V2")
        #rgb_backbone.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        self.rgb_backbone = IntermediateLayerGetter(rgb_backbone, return_layers=return_layers)
        self.fusion = Architechture_1(d_model,
                                      {"self": [2,4,4,8], "cross":[2,4,4,8]}, 
                                      {"self": depth[0], "cross": depth[1]}, 
                                      dropout=0., normalize_before=True)
        self.feat_2d_red = BasicConv2d(1024, d_model, kernel_size=1, padding=0)
        self.feat_3d_red = BasicConv2d(256, d_model, kernel_size=3, padding=1)
        self.prediction = BasicConv2d(d_model, n_classes, kernel_size=1, padding=0)
        self.pos = PositionEmbeddingSine(d_model//2)
        self.query_pos = nn.Embedding(n_queries, d_model)
        print("Fusion model initialized")
        
    def forward(self, lidar, rgb):
        assert lidar.shape[1] == 6, "Lidar input must have 6 channels include a mask channel"
        B, _, H, W = lidar.shape
        lidar, mask = lidar[:, :5, :, :], lidar[:, 5, :, :].bool()
        lidar_out = self.lidar_model(lidar, rgb)
        lidar_feature = self.lidar_model.feature_3D
        lidar_feature = self.feat_3d_red(lidar_feature)
        
        rgb_feature = self.rgb_backbone(rgb)["out"]
        rgb_feature = self.feat_2d_red(rgb_feature)
        rgb_pos = self.pos(rgb_feature)
        
        query = lidar_feature.flatten(2).transpose(1, 2)
        rgb_feature = rgb_feature.flatten(2).transpose(1, 2)
        rgb_pos = rgb_pos.flatten(2).transpose(1, 2)
        query = []
        query_mask = []
        for i in range(B):
            query_curr = lidar_feature[i, :, mask[i, :]]
            mask_curr = torch.zeros(self.n_queries, dtype=torch.bool, device=query_curr.device)
            mask_curr[:query_curr.shape[1]] = True
            query_curr = F.pad(query_curr, (0, self.n_queries - query_curr.shape[1]), "constant", 0.0)
            query_mask.append(mask_curr)
            query.append(query_curr)
        query = torch.stack(query, dim=0).transpose(1, 2)
        query_mask = torch.stack(query_mask, dim=0)
        query_fused = self.fusion(query, rgb_feature, 
                                  query_key_padding_mask=query_mask, 
                                  pos=rgb_pos, 
                                  query_pos=self.query_pos.weight)
        
        for i in range(B):
            lidar_feature[i, :, mask[i, :]] = query_fused[i, query_mask[i, :], :].transpose(0, 1)
        fused_pred = F.softmax(self.prediction(lidar_feature), dim=1)
        out = [fused_pred] + lidar_out
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




if __name__ == "__main__":
    model = Fusion(20)
    overfit_test(model, 6, (3, 256, 768), (6, 64, 512))
    pass

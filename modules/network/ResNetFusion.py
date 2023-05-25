import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F
from Fusion import BasicConv2d
from positional_encodings.torch_encodings import PositionalEncoding1D
import fvcore.nn.weight_init as weight_init
from copy import deepcopy


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1, downsample = None, reidual ="conv1_1"):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
#                         nn.BatchNorm2d(out_channels),
#                         nn.LeakyReLU())
#         self.conv2 = nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
#                         nn.BatchNorm2d(out_channels))
        
#         self.downsample = downsample
#         self.relu = nn.LeakyReLU()
#         self.out_channels = out_channels
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
    
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

    def __init__(self, nclasses, norm_layer=None):

        super(Fusion, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layer_list = {'layer1': 'feat1', 
                      'layer2': 'feat2', 
                      'layer3': 'feat3'}
        self.backbone_3d = resnet50(pretrained=False)
        self.backbone_3d.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_3d = IntermediateLayerGetter(self.backbone_3d, return_layers=layer_list)
        self.backbone_2d = resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.backbone_2d = IntermediateLayerGetter(self.backbone_2d, return_layers=layer_list)
        for p in self.backbone_2d.parameters():
            p.requires_grad = False
        for n, p in self.backbone_2d.named_children():
            for n_, p_ in p.named_children():
                continue
            if "layer" in n:
                for param in p_.parameters():
                    param.requires_grad = True
        self.feature_preserve = BasicConv2d(256, 128, kernel_size=1, padding=0)
        self.query_generate = BasicConv2d(1024, 256, kernel_size=1, padding=0)
        self.semantic_prediction = BasicConv2d(256, nclasses, kernel_size=1, padding=0)
        self.fusion_layer = Attention_Fusion()

        
    def forward(self, lidar, rgb):
        proj_size = lidar.shape[2:]
        #rgb_size = rgb.shape[2:]
        lidar = self.backbone_3d(lidar)
        query = self.query_generate(lidar["feat3"])
        query_size = query.shape
        rgb = self.backbone_2d(rgb)
        # mask_preserved = F.interpolate(rgb["feat1"], size=proj_size, mode='bilinear', align_corners=True)
        # mask_preserved = self.feature_preserve(mask_preserved)
        queries = self.fusion_layer(query, rgb) # * shallow to deep
        for query in queries:
            query = query.view(query_size)
            print("a)")
        # out = []
        # for query in reversed(queries):
        #     final_feat = torch.einsum('bqf,bqhw->bfhw', query, mask_preserved)
        #     final_feat = self.semantic_prediction(final_feat)
        #     out.append(F.softmax(final_feat, dim=1))
        #     return out
        return queries

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
    import time
    l_weight = torch.ones(20).cuda()
    l_weight[0] = 0.0
    model = Fusion(20, l_weight=l_weight).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    from ioueval import *
    evaluator = iouEval(20, device='cuda', ignore=0)
    
    input_3D = torch.randn(2, 5, 64, 512).cuda()
    input_rgb = torch.rand(2, 3, 452, 1032).cuda()
    label = torch.randint(0, 20, (2, 64, 512)).cuda()
    for i in range(20000):
        
        model.train()
        #with torch.no_grad():
        start_time = time.time()
        outputs = model(input_3D, input_rgb, label)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp = [(input_3D.shape[2], input_3D.shape[3])] * 2
        predicted_semantic_map = model.seg_eval(outputs)
        evaluator.addBatch(predicted_semantic_map, label)
        accuracy = evaluator.getacc()
        
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        fwt = time.time() - start_time
        time_train.append(fwt)
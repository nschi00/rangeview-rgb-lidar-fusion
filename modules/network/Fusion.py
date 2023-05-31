import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'modules'))
from overfit_test import overfit_test
from transformers import Mask2FormerModel
from torch.nn import functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from third_party.SwinFusion.models.network_swinfusion import Cross_BasicLayer, PatchUnEmbed
from ResNet import BasicBlock, BasicConv2d, conv1x1
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
import numpy as np
from SwinFusion import SwinFusion


""" This is a torch way of getting the intermediate layers
 return_layers = {"layer4": "out"}
 if aux:
      return_layers["layer3"] = "aux"
 backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
"""


class BackBone(nn.Module):
    """_summary_

    Args:
        name: name of the backbone
        use_att: whether to use attention
        fuse_all: whether to fuse all the layers in the backbone
        branch_type: semantic, instance or panoptic
        stage: whether to only use the encoder, pixel_decoder or
               combined pixel/transformer decoder output
        in_size: input size of the image
    """

    def __init__(self, name,
                 use_att=False,
                 fuse_all=False,
                 branch_type="semantic",
                 stage="combination",
                 in_size=(384, 1024)):
        super().__init__()
        assert name in ["resnet50", "mask2former"], \
            "Backbone name must be either resnet50 or mask2former"
        assert branch_type in ["semantic", "instance", "panoptic"], \
            "Branch type must be either semantic, instance or panoptic"
        assert stage in ["enc", "pixel_decoder", "combination"]

        def get_smaller(in_size, scale):
            """
            Returns a tuple of the smaller size given an input size and scale factor.
            """
            return (in_size[0] // scale, in_size[1] // scale)

        self.name = name
        self.in_size = in_size
        self.branch_type = branch_type
        self.stage = stage
        self.fuse_all = fuse_all
        output_res = [1, 2, 4, 8]  # * Define output resolution
        if use_att and fuse_all:
            # * only fuse with attention in 2 resolutions to save resources
            output_res = [1, 1, 4, 4]
        if stage == "enc":
            hidden_dim = [96, 192, 384, 768]
            self.layer_list = ["encoder_hidden_states"] * 4
        elif stage == "pixel_decoder":
            hidden_dim = [256] * 4
            self.layer_list = ["decoder_hidden_states"] * 3 + ["decoder_last_hidden_state"]
        else:  # * stage == "combination"
            hidden_dim = [128] * 4
            self.layer_list = []  # not necessary for combination

        if name == "resnet50":
            hidden_dim = [256, 512, 1024, 2048]
            output_red = [2, 4, 8, 16]
            input_red = [4, 8, 16, 32]
            self.layer_list = ["feat1", "feat2", "feat3", "feat4"]
            self.processor = None
            self.backbone = resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
            assert fuse_all == True, "ResNet50 only supports fuse_all=True"
            self.backbone = IntermediateLayerGetter(self.backbone, return_layers={'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
            # test_input = torch.randn(1, 3, in_size[0], in_size[1])
            # test_output = self.backbone(test_input)
            # for p in self.backbone.parameters():
            #     p.requires_grad = False
            # for n, p in self.backbone.named_children():
            #     #last_layer = 0
            #     for n_, p_ in p.named_children():
            #         continue
            #         # if int(n_) > last_layer:
            #         #     last_layer = int(n_)
            #     if "layer" in n:
            #         for param in p_.parameters():
            #             param.requires_grad = True
            #     #print(n, last_layer)
            #     #p.requires_grad = True
            # pytorch_total_params = sum(p.numel()
            #                    for p in self.backbone.parameters() if p.requires_grad)
            # print(self.backbone)
            # print(pytorch_total_params)
            if fuse_all:
                self.upscale_layers = nn.ModuleList(
                    [Upscale_head(hidden_dim[i], 128, 
                                  get_smaller(in_size, input_red[i]), 
                                  get_smaller(in_size, output_red[i]))
                                for i in range(4)])

        elif name == "mask2former":
            weight = "facebook/mask2former-swin-tiny-cityscapes-{}".format(
                branch_type)
            if stage != "combination":
                self.backbone = Mask2FormerModel.from_pretrained(
                    weight).pixel_level_module
            else:
                self.backbone = Mask2FormerModel.from_pretrained(weight)
                self.class_predictor = nn.Linear(256, 128)

            if fuse_all:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[i], 128,
                                                                  get_smaller(in_size, 4),
                                                                  get_smaller(in_size, output_res[i]))
                                                     for i in range(4)])
            else:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128,
                                                                  get_smaller(in_size, 4), in_size)])

        else:
            raise NotImplementedError

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(
                x, output_hidden_states=True) if self.name != "resnet50" else self.backbone(x)

        outputs = []
        if self.name == "mask2former" and self.stage == "combination":
            class_queries_logits = self.class_predictor(torch.stack(x.transformer_decoder_intermediate_states,
                                                                    dim=0).transpose(1, 2))
            masks_queries_logits = x.masks_queries_logits
            del x

            for i in range(1, 8, 2):
                masks_classes = class_queries_logits[-i]
                # [batch_size, num_queries, height, width]
                masks_probs = masks_queries_logits[-i]
                # Semantic segmentation logits of shape (batch_size, num_channels_lidar=128, height, width)
                out = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
                outputs.append(out)
                if not self.fuse_all:
                    break
        else:
            for i in range(len(self.layer_list)):
                if type(x[self.layer_list[i]]) == tuple:
                    out = x[self.layer_list[i]][i]
                    outputs.append(out)
                else:
                    out = x[self.layer_list[i]]
                    outputs.append(out)
                    outputs.reverse() if self.name != "resnet50" else outputs

        for j, layer in enumerate(self.upscale_layers):
            outputs[j] = layer(outputs[j])

        return outputs

class FusionDouble(nn.Module):
    """_summary_
    Twin Fusion for LiDAR and RGB

    Args:
        nclasses: number of classes
        block: block type
        layers: number of layers
        if_BN: whether to use batch normalization
        norm_layer: normalization layer
        groups: number of groups
        width_per_group: width per group
    """

    def __init__(self, nclasses, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64):

        super(FusionDouble, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.initial_conv_lidar = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                                BasicConv2d(64, 128, kernel_size=3, padding=1),
                                                BasicConv2d(128, 128, kernel_size=3, padding=1))
        
        self.initial_conv_rgb = nn.Sequential(BasicConv2d(3, 64, kernel_size=3, padding=1),
                                              BasicConv2d(64, 128, kernel_size=3, padding=1),
                                              BasicConv2d(128, 128, kernel_size=3, padding=1))

        self.inplanes = 128

        self.unet_layers_lidar = nn.ModuleList()
        self.unet_layers_rgb = nn.ModuleList()
        self.resi_convs = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers_lidar.append(self._make_layer(block, 128, layers[i], stride=j))
            self.unet_layers_rgb.append(self._make_layer(block, 128, layers[i], stride=j))
            self.resi_convs.append(BasicConv2d(256, 128, kernel_size=1, padding=0))

        self.end_conv_lidar = nn.Sequential(BasicConv2d(640, 256, kernel_size=3, padding=1),
                                            BasicConv2d(256, 128, kernel_size=3, padding=1))
        
        self.end_conv_rgb = nn.Sequential(BasicConv2d(640, 256, kernel_size=3, padding=1),
                                          BasicConv2d(256, 128, kernel_size=3, padding=1))

        self.semantic_output = nn.Conv2d(128, nclasses, 1)
        
        """FUSION LAYERS"""
        # ! Redefine for different resolutions
        project_size = np.array([64, 192])  
        img_size = np.array([64, 192])
        # TODO:Check original paper for better understanding
        self.fusion_layer = [
            SwinFusion(img_size_A=project_size, img_size_B=img_size, patch_size_A=(2,2), patch_size_B=(2,2), window_size=16, mlp_ratio=2),
            SwinFusion(img_size_A=project_size//2, img_size_B=img_size//2, patch_size_A=(2,2), patch_size_B=(2,2), window_size=8, mlp_ratio=2),
            SwinFusion(img_size_A=project_size//4, img_size_B=img_size//4, patch_size_A=(1,1), patch_size_B=(1,1), window_size=4, mlp_ratio=2),
            SwinFusion(img_size_A=project_size//8, img_size_B=img_size//8, patch_size_A=(1,1), patch_size_B=(1,1), window_size=4, mlp_ratio=2)
        ]
        self.fusion_layer = nn.ModuleList(self.fusion_layer)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x, rgb):
        # * get RGB features
        proj_size = x.size()[2:]
        rgb_size = rgb.size()[2:]

        # * get projection and rgb features
        x_lidar = self.initial_conv_lidar(x)
        x_rgb = self.initial_conv_rgb(rgb)
        
        proj_size = x_lidar.size()[2:]
        rgb_size = x_rgb.size()[2:]
        
        x_lidar_features = {'0': x_lidar}
        x_rgb_features = {'0': x_rgb}
        del x_lidar
        del x_rgb

        for i in range(0, 4):
            x_lidar_features[str(i + 1)] = self.unet_layers_lidar[i](x_lidar_features[str(i)])
            x_rgb_features[str(i + 1)] = self.unet_layers_rgb[i](x_rgb_features[str(i)])
            x_lidar_features[str(i + 1)], x_rgb_features[str(i + 1)] = self.fusion_layer[i](x_lidar_features[str(i + 1)], x_rgb_features[str(i + 1)])
            #x_lidar_features[str(i + 1)] = self.resi_convs[i](torch.cat([x_lidar_features[str(i + 1)], lidar_fused], dim=1))
            #x_rgb_features[str(i + 1)] = self.resi_convs[i](torch.cat([x_rgb_features[str(i + 1)], rgb_fused], dim=1))
            
        for i in range(2, 5):
            x_lidar_features[str(i)] = F.interpolate(
                x_lidar_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)
            x_rgb_features[str(i)] = F.interpolate(
                x_rgb_features[str(i)], size=rgb_size, mode='bilinear', align_corners=True)
            
        x_lidar_features = list(x_lidar_features.values())
        x_rgb_features = list(x_rgb_features.values())
        
        out_lidar = self.end_conv_lidar(torch.cat(x_lidar_features, dim=1))
        out_rgb = self.end_conv_rgb(torch.cat(x_rgb_features, dim=1))
   
        out_lidar = self.semantic_output(out_lidar)
        out_lidar = F.softmax(out_lidar, dim=1)
        
        out_rgb = self.semantic_output(out_rgb)
        out_rgb = F.softmax(out_rgb, dim=1)

        return [out_lidar, out_rgb]

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
                 norm_layer=None, groups=1, width_per_group=64, use_att=False, fusion_scale='main_late',
                 name_backbone="resnet50", branch_type="semantic", stage="combination"):

        super(Fusion, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        assert fusion_scale in ["all_early", "all_late", "main_late", "main_early"]
        if use_att:
            assert fusion_scale in ["main_early", "main_late"]
        # if fusion_scale == "main_late":
        #     assert aux == False
        # * whether to fuse all scales or only main scales
        self.ALL = "all" in fusion_scale
        # * whether to fuse early or late
        self.EARLY = "early" in fusion_scale
        self.use_att = use_att

        self.backbone = BackBone(name=name_backbone, use_att=use_att,
                                 fuse_all=self.ALL, stage=stage, branch_type=branch_type)

        """BASEMODEL"""
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.initial_conv = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, 128, kernel_size=3, padding=1),
                                          BasicConv2d(128, 128, kernel_size=3, padding=1))

        self.inplanes = 128

        self.unet_layers = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers.append(self._make_layer(block, 128, layers[i], stride=j))

        self.end_conv = nn.Sequential(BasicConv2d(640, 256, kernel_size=3, padding=1),
                                      BasicConv2d(256, 128, kernel_size=3, padding=1))

        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_heads = nn.ModuleDict()
            for i in range(2, 5):
                self.aux_heads["layer{}".format(i)] = nn.Conv2d(128, nclasses, 1)

        """FUSION LAYERS"""
        if not use_att:
            self.fusion_layer = BasicConv2d(256, 128, kernel_size=1, padding=0)
        else:
            self.fusion_layer = Cross_Attention()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x, rgb):
        # * get RGB features
        proj_size = x.size()[2:]
        rgb_out = self.backbone(rgb)

        # * get projection features
        x = self.initial_conv(x)
        """FUSION MAIN SCALES"""
        x_features = {'0': x}
        del x

        for i in range(0, 4):
            x_features[str(i + 1)] = self.unet_layers[i](x_features[str(i)])
            if self.EARLY and i == 0:
                x_features[str(i + 1)] = self.fusion_layer((x_features[str(i + 1)], rgb_out[i]))
                continue

            if self.ALL:
                x_features[str(i + 1)] = self.fusion_layer((x_features[str(i + 1)], rgb_out[i]))

        # TODO: consider implement "all" early fusion for cross attention: very F***ing expensive
        for i in range(2, 5):
            x_features[str(i)] = F.interpolate(
                x_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)

        x_features = list(x_features.values())
        out = torch.cat(x_features, dim=1)
        out = self.end_conv(out)

        """LATE FUSION"""
        if not self.EARLY:
            out = self.fusion_layer((out, rgb_out[0]))
            # ! Size reduction due to patch size: patch_size = 1 can be heavy to calculate
            if out.shape != rgb_out[0].shape:
                out = F.interpolate(
                    out, size=rgb_out[0].shape[2:], mode='bilinear', align_corners=True)

        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            out = [out]
            for i in range(2, 5):
                out.append(self.aux_heads["layer{}".format(i)](x_features[i]))
                out[-1] = F.softmax(out[-1], dim=1)

        return out


class Upscale_head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, size_in: tuple, size_out: tuple) -> None:
        super(Upscale_head, self).__init__()
        inter_channels = in_channels // 4
        h_in, w_in = size_in
        h_out, w_out = size_out
        self.size_inter = ((h_in + h_out) // 2, (w_in + w_out) // 2)
        self.size_out = size_out

        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(
            inter_channels, out_channels, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_inter = self.conv1(F.interpolate(
            x, size=self.size_inter, mode='bilinear', align_corners=True))
        x_inter = self.relu(self.bn1(x_inter))

        x_out = self.conv2(F.interpolate(
            x_inter, size=self.size_out, mode='bilinear', align_corners=True))
        x_out = self.relu(self.bn2(x_out))
        return x_out
    
class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.view(-1, self.embed_dim, Wh, Ww)

        return x
    
    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops
    
class Cross_Attention(nn.Module):
    def __init__(self, A_size, B_size, A_patchs, B_patchs, depth, hidden_dim=128, n_head=8) -> None:
        super().__init__()
        to_2tuple(A_patchs)
        to_2tuple(B_patchs)
        to_2tuple(A_size)
        to_2tuple(B_size)
        self.hidden_dim = hidden_dim
        self.patch_embed_A = PatchEmbed(patch_size=A_patchs, in_chans=128, embed_dim=hidden_dim)
        self.patch_embed_B = PatchEmbed(patch_size=B_patchs, in_chans=128, embed_dim=hidden_dim)
        self.patch_unembed = PatchUnEmbed(patch_size=A_patchs, in_chans=128, embed_dim=hidden_dim)
        self.A_size = A_size
        self.B_size = B_size
        
        self.A_reso = (int(A_size[0]/A_patchs[0]), int(A_size[1]/A_patchs[1]))
        self.B_reso = (int(B_size[0]/B_patchs[0]), int(B_size[1]/B_patchs[1]))
        
        self.A_pe = nn.Parameter(torch.zeros(1, hidden_dim, self.A_reso[0], self.A_reso[1]))
        trunc_normal_(self.A_pe, std=.02)
        self.B_proj = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0, bias=False), 
                                    nn.LeakyReLU(inplace=True))
        
        self.decoder = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True)
        assert self.A_reso == self.B_reso, "Resolution of A and B must be the same"
        
    
    def forward(self, A, B):
        # TODO: Delete this later, only for testing
        #B = torch.transpose(B, 2, 3)
        A = self.patch_embed_A(A)
        B = self.patch_embed_B(B)
        #h_A, w_A = self.A_reso
        assert A.shape[2:] == self.A_reso, "A resolution must be the same as the patch resolution"
        assert B.shape[2:] == self.B_reso, "B resolution must be the same as the patch resolution"
        
        A = A + self.A_pe
        B = B + self.B_proj(self.A_pe)
        A = A.view(-1, self.A_reso[0]*self.A_reso[1], self.hidden_dim)
        B = B.view(-1, self.B_reso[0]*self.B_reso[1], self.hidden_dim)
        out = self.decoder(A, B, B)
        return out
        

    
def get_n_params(model):
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    #print(pytorch_total_params)
    return pytorch_total_params

if __name__ == "__main__":
    model = FusionDouble(20)
    overfit_test(model, 6, (3,64,192), (5,64,192), rgb=True)

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())

from transformers import Mask2FormerModel
from torch.nn import functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from third_party.SwinFusion.models.network_swinfusion import Cross_BasicLayer, PatchUnEmbed
from ResNet import BasicBlock, BasicConv2d, conv1x1
from torchvision.models.segmentation import deeplabv3_resnet50
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
                 in_size=(64, 512)):
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
            hidden_dim = [2048] + [1024] * 3
            self.layer_list = ["out"] + ["aux"] * 3
            self.processor = None
            self.backbone = deeplabv3_resnet50(weights='DEFAULT').backbone

            if fuse_all:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[i],
                                                                  128, (8, 16),
                                                                  get_smaller(in_size, output_res[i]))
                                                     for i in range(3)])
                self.upscale_layers.append(BasicConv2d(hidden_dim[3], 128, 1))
            else:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128, (8, 16), in_size)])

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
    """
    All scale fusion with resnet50 backbone
    Basic fusion with torch.cat + conv1x1
    use_att: use cross attention or not
    fusion_scale: "all_early" or "all_late" or "main_late" or "main_early"
    name_backbone: backbone for RGB, only "resnet50" at the moment possible
    branch_type: semantic, instance or panoptic
    stage: whether to only use the enc, pixel_decoder or combined pixel/transformer decoder output (combination)
    """

    def __init__(self, nclasses, aux=False, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64):

        super(FusionDouble, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        """BASEMODEL"""
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.initial_conv_lidar = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, 128, kernel_size=3, padding=1),
                                          BasicConv2d(128, 128, kernel_size=3, padding=1))
        
        self.initial_conv_rgb = nn.Sequential(BasicConv2d(3, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, 128, kernel_size=3, padding=1),
                                          BasicConv2d(128, 128, kernel_size=3, stride=2, padding=1))

        self.inplanes = 128

        self.unet_layers_lidar = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers_lidar.append(self._make_layer(block, 128, layers[i], stride=j))

        self.unet_layers_rgb = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers_rgb.append(self._make_layer(block, 128, layers[i], stride=j))

        self.end_conv_lidar = nn.Sequential(BasicConv2d(640, 256, kernel_size=3, padding=1),
                                      BasicConv2d(256, 128, kernel_size=3, padding=1))
        
        self.end_conv_rgb = nn.Sequential(BasicConv2d(640, 256, kernel_size=3, stride=(3,1), padding=1),
                                      BasicConv2d(256, 128, kernel_size=3, padding=1))

        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_heads = nn.ModuleDict()
            for i in range(2, 5):
                self.aux_heads["layer{}".format(i)] = nn.Conv2d(128, nclasses, 1)

        """FUSION LAYERS"""
        # ! Redefine for different resolutions
        project_size = np.array([64, 512])  
        img_size = np.array([192, 512])
        self.fusion_layer = [
            SwinFusion(img_size_A=project_size, img_size_B=img_size, patch_size_A=(1,2), patch_size_B=(3,2), window_size=8),
            SwinFusion(img_size_A=project_size//2, img_size_B=img_size//2, patch_size_A=(1,2), patch_size_B=(3,2), window_size=4),
            SwinFusion(img_size_A=project_size//4, img_size_B=img_size//4, patch_size_A=(1,2), patch_size_B=(3,2), window_size=2),
            SwinFusion(img_size_A=project_size//8, img_size_B=img_size//8, patch_size_A=(1,2), patch_size_B=(3,2), window_size=2)
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
        rgb = torch.transpose(rgb, 2, 3) # * Make RGB same size as LiDAR
        proj_size = x.size()[2:]
        rgb_size = rgb.size()[2:]
        # rgb_out = self.backbone(rgb)

        # * get projection features
        x_lidar = self.initial_conv_lidar(x)
        x_rgb = self.initial_conv_rgb(rgb)
        
        proj_size = x_lidar.size()[2:]
        rgb_size = x_rgb.size()[2:]
        
        """FUSION MAIN SCALES"""
        x_lidar_features = {'0': x_lidar}
        x_rgb_features = {'0': x_rgb}
        del x_lidar
        del x_rgb

        for i in range(0, 4):
            x_lidar_features[str(i + 1)] = self.unet_layers_lidar[i](x_lidar_features[str(i)])
            x_rgb_features[str(i + 1)] = self.unet_layers_rgb[i](x_rgb_features[str(i)])
            x_lidar_features[str(i + 1)], x_rgb_features[str(i + 1)] = self.fusion_layer[i](x_lidar_features[str(i + 1)], x_rgb_features[str(i + 1)])
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
                 name_backbone="mask2former", branch_type="semantic", stage="combination"):

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
            self.fusion_layer = Cross_SW_Attention()

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



if __name__ == "__main__":
    import time
    model = FusionDouble(20).cuda()
    print(model)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")

    # for module_name, module in model.named_parameters():
    #     if "fusion" in module_name:
    #         print(module_name, module)

    time_train = []
    for i in range(20):
        input_3D = torch.rand(2, 5, 64, 512).cuda()
        input_rgb = torch.rand(2, 3, 512*2, 192*2).cuda()
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_3D, input_rgb)
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)
        print("Forward time per img: %.3f (Mean: %.3f)" % (
            fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)

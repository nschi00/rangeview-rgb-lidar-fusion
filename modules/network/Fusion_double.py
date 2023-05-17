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


""" This is a torch way of getting the intermediate layers
 return_layers = {"layer4": "out"}
 if aux:
      return_layers["layer3"] = "aux"
 backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
"""


# class BackBone(nn.Module):
#     """_summary_

#     Args:
#         name: name of the backbone
#         use_att: whether to use attention
#         fuse_all: whether to fuse all the layers in the backbone
#         branch_type: semantic, instance or panoptic
#         stage: whether to only use the encoder, pixel_decoder or
#                combined pixel/transformer decoder output
#         in_size: input size of the image
#     """

#     def __init__(self, name,
#                  use_att=False,
#                  fuse_all=False,
#                  branch_type="semantic",
#                  stage="combination",
#                  in_size=(64, 512)):
#         super().__init__()
#         assert name in ["resnet50", "mask2former"], \
#             "Backbone name must be either resnet50 or mask2former"
#         assert branch_type in ["semantic", "instance", "panoptic"], \
#             "Branch type must be either semantic, instance or panoptic"
#         assert stage in ["enc", "pixel_decoder", "combination"]

#         def get_smaller(in_size, scale):
#             """
#             Returns a tuple of the smaller size given an input size and scale factor.
#             """
#             return (in_size[0] // scale, in_size[1] // scale)

#         self.name = name
#         self.in_size = in_size
#         self.branch_type = branch_type
#         self.stage = stage
#         self.fuse_all = fuse_all
#         output_res = [1, 2, 4, 8]  # * Define output resolution
#         if use_att and fuse_all:
#             # * only fuse with attention in 2 resolutions to save resources
#             output_res = [1, 1, 4, 4]
#         if stage == "enc":
#             hidden_dim = [96, 192, 384, 768]
#             self.layer_list = ["encoder_hidden_states"] * 4
#         elif stage == "pixel_decoder":
#             hidden_dim = [256] * 4
#             self.layer_list = ["decoder_hidden_states"] * 3 + ["decoder_last_hidden_state"]
#         else:  # * stage == "combination"
#             hidden_dim = [128] * 4
#             self.layer_list = []  # not necessary for combination

#         if name == "resnet50":
#             hidden_dim = [2048] + [1024] * 3
#             self.layer_list = ["out"] + ["aux"] * 3
#             self.processor = None
#             self.backbone = deeplabv3_resnet50(weights='DEFAULT').backbone

#             if fuse_all:
#                 self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[i],
#                                                                   128, (8, 16),
#                                                                   get_smaller(in_size, output_res[i]))
#                                                      for i in range(3)])
#                 self.upscale_layers.append(BasicConv2d(hidden_dim[3], 128, 1))
#             else:
#                 self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128, (8, 16), in_size)])

#         elif name == "mask2former":
#             weight = "facebook/mask2former-swin-tiny-cityscapes-{}".format(
#                 branch_type)
#             if stage != "combination":
#                 self.backbone = Mask2FormerModel.from_pretrained(
#                     weight).pixel_level_module
#             else:
#                 self.backbone = Mask2FormerModel.from_pretrained(weight)
#                 self.class_predictor = nn.Linear(256, 128)

#             if fuse_all:
#                 self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[i], 128,
#                                                                   get_smaller(in_size, 4),
#                                                                   get_smaller(in_size, output_res[i]))
#                                                      for i in range(4)])
#             else:
#                 self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128,
#                                                                   get_smaller(in_size, 4), in_size)])

#         else:
#             raise NotImplementedError

#         for p in self.backbone.parameters():
#             p.requires_grad = False

#     def forward(self, x):
#         with torch.no_grad():
#             x = self.backbone(
#                 x, output_hidden_states=True) if self.name != "resnet50" else self.backbone(x)

#         outputs = []
#         if self.name == "mask2former" and self.stage == "combination":
#             class_queries_logits = self.class_predictor(torch.stack(x.transformer_decoder_intermediate_states,
#                                                                     dim=0).transpose(1, 2))
#             masks_queries_logits = x.masks_queries_logits
#             del x

#             for i in range(1, 8, 2):
#                 masks_classes = class_queries_logits[-i]
#                 # [batch_size, num_queries, height, width]
#                 masks_probs = masks_queries_logits[-i]
#                 # Semantic segmentation logits of shape (batch_size, num_channels_lidar=128, height, width)
#                 out = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
#                 outputs.append(out)
#                 if not self.fuse_all:
#                     break
#         else:
#             for i in range(len(self.layer_list)):
#                 if type(x[self.layer_list[i]]) == tuple:
#                     out = x[self.layer_list[i]][i]
#                     outputs.append(out)
#                 else:
#                     out = x[self.layer_list[i]]
#                     outputs.append(out)
#                     outputs.reverse() if self.name != "resnet50" else outputs

#         for j, layer in enumerate(self.upscale_layers):
#             outputs[j] = layer(outputs[j])

#         return outputs


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

        # self.backbone = BackBone(name=name_backbone, use_att=use_att,
        #                          fuse_all=self.ALL, stage=stage, branch_type=branch_type)

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
                                          BasicConv2d(128, 128, kernel_size=3, padding=1))

        self.inplanes = 128

        self.unet_layers_lidar = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers_lidar.append(self._make_layer(block, 128, layers[i], stride=j))

        self.unet_layers_rgb = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers_rgb.append(self._make_layer(block, 128, layers[i], stride=j))

        self.end_conv = nn.Sequential(BasicConv2d(1280, 640, kernel_size=3, padding=1),
                                    BasicConv2d(640, 256, kernel_size=3, padding=1),
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
            self.fusion_layer = Cross_SW_Attention(in_chans=640, embed_dim=128, patch_size=1)

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
        # rgb_out = self.backbone(rgb)

        # * get projection features
        x_lidar = self.initial_conv_lidar(x)
        x_rgb = self.initial_conv_rgb(rgb)
        """FUSION MAIN SCALES"""
        x_lidar_features = {'0': x_lidar}
        x_rgb_features = {'0': x_rgb}
        del x_lidar
        del x_rgb

        for i in range(0, 4):
            x_lidar_features[str(i + 1)] = self.unet_layers_lidar[i](x_lidar_features[str(i)])
            x_rgb_features[str(i + 1)] = self.unet_layers_rgb[i](x_rgb_features[str(i)])
            # if self.EARLY and i == 0:
            #     x_features[str(i + 1)] = self.fusion_layer((x_features[str(i + 1)], rgb_out[i]))
            #     continue

            # if self.ALL:
            #     x_features[str(i + 1)] = self.fusion_layer((x_features[str(i + 1)], rgb_out[i]))

        # TODO: consider implement "all" early fusion for cross attention: very F***ing expensive
        for i in range(2, 5):
            x_lidar_features[str(i)] = F.interpolate(
                x_lidar_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)
            x_rgb_features[str(i)] = F.interpolate(
                x_rgb_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)
            
        if not self.use_att:
            x_lidar_features = list(x_lidar_features.values())
            x_rgb_features = list(x_rgb_features.values())
            out = torch.cat(x_lidar_features + x_rgb_features, dim=1)
            out = self.end_conv(out)
        else:
            x_lidar_features = torch.cat(list(x_lidar_features.values()), dim=1)
            x_rgb_features = torch.cat(list(x_rgb_features.values()), dim=1)
            out = self.fusion_layer((x_lidar_features, x_rgb_features))

        # """LATE FUSION"""
        # if not self.EARLY:
        #     out = self.fusion_layer((out, rgb_out[0]))
        #     # ! Size reduction due to patch size: patch_size = 1 can be heavy to calculate
        #     if out.shape != rgb_out[0].shape:
        #         out = F.interpolate(
        #             out, size=rgb_out[0].shape[2:], mode='bilinear', align_corners=True)

        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            out = [out]
            for i in range(2, 5):
                out.append(self.aux_heads["layer{}".format(i)](x_lidar_features[i]))
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


class Cross_SW_Attention(nn.Module):
    """_summary_

    Args:
        pe_type: "abs" or "adaptive"
        fusion_type: "x_main" or "double_fuse"
        embed_dim (int): embedded dimension.
        input_size (tuple[int]): Input size HxW.
        patch_size (tuple[int]): patch size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, pe_type="abs", fusion_type="x_main", in_chans=128, embed_dim=128, input_size=(64, 512),
                 patch_size=2, depth=3, num_heads=8, window_size=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.pe_type = pe_type
        self.fusion_type = fusion_type
        self.window_size = window_size

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer)

        self.patch_unembed = PatchUnEmbed(
            img_size=input_size, patch_size=patch_size, embed_dim=embed_dim,
            norm_layer=norm_layer)

        if self.pe_type == 'abs':
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                input_size[0] // patch_size[0],
                input_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed_A = nn.Parameter(
                torch.zeros(
                    1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            self.absolute_pos_embed_B = nn.Parameter(
                torch.zeros(
                    1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed_A, std=0.02)
            trunc_normal_(self.absolute_pos_embed_B, std=0.02)
        elif self.pe_type == 'adaptive':
            pass
        # TODO: add adaptive PE
        else:
            raise NotImplementedError(f"Not support pe type {self.pe_type}.")

        self.pos_drop = nn.Dropout(p=drop)

        """Input resolution = Patch's resolution"""
        self.cross_attention = Cross_BasicLayer(dim=embed_dim,
                                                input_resolution=(patches_resolution[0],
                                                                  patches_resolution[1]),
                                                depth=depth,
                                                num_heads=num_heads,
                                                window_size=window_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer)

        self.norm_A = norm_layer(self.embed_dim)
        if fusion_type == 'double_fuse':
            self.conv_fusion = BasicBlock(256, 128, 1)
            self.norm_B = norm_layer(self.embed_dim)

    def forward(self, x):
        # * Divide image into patches
        x, y = x
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        x_size = (x.shape[2], x.shape[3])

        # * Add position embedding
        x = (x + self.absolute_pos_embed_A).flatten(2).transpose(1, 2)
        y = (y + self.absolute_pos_embed_B).flatten(2).transpose(1, 2)

        x = self.pos_drop(x)
        y = self.pos_drop(y)
        if self.fusion_type == "x_main":
            x, _ = self.cross_attention(x, y, x_size)
            # x = self.norm_A(x)  # B L C
            x = self.patch_unembed(x, x_size)

        elif self.fusion_type == "double_fuse":
            x, y = self.cross_attention(x, y, x_size)
            x = self.norm_Fusion_A(x)  # B L C
            x = self.patch_unembed(x, x_size)

            y = self.norm_Fusion_B(y)  # B L C
            y = self.patch_unembed(y, x_size)

            x = torch.cat([x, y], 1)
            x = self.conv_fusion(x)

        else:
            raise NotImplementedError(
                f"Not support fusion type {self.fusion_type}.")

        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1),
                                  nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
                                  nn.ReLU(inplace=True))

        pytorch_total_params = sum(p.numel() for p in self.proj.parameters())
        print(pytorch_total_params)
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
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


if __name__ == "__main__":
    import time
    model = Fusion(20, use_att=False, fusion_scale="main_late").cuda()
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
        input_rgb = torch.rand(2, 3, 64, 512).cuda()
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

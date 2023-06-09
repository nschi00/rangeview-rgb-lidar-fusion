import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())
sys.path.append("modules")
from overfit_test import overfit_test

from torch.nn import functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from third_party.SwinFusion.models.network_swinfusion1 import RSTB, CRSTB, PatchUnEmbed, PatchEmbed, Upsample, UpsampleOneStep
from ResNet import BasicBlock, BasicConv2d, conv1x1
import torch
import torch.nn as nn


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
                 norm_layer=None, groups=1, width_per_group=64, use_skip=True):

        super(Fusion, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        """BASEMODEL"""
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux
        self.use_skip = use_skip

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

        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_heads = nn.ModuleDict()
            for i in range(2, 5):
                self.aux_heads["layer{}".format(i)] = nn.Conv2d(128, nclasses, 1)

        """FUSION LAYERS"""
        self.conv_before_fusion_lidar = BasicConv2d(640, 128, kernel_size=1, padding=0)
        self.conv_before_fusion_rgb = BasicConv2d(640, 128, kernel_size=1, padding=0)
        upscale = 4
        window_size = 8
        height = (64 // upscale // window_size + 1) * window_size
        width = (512 // upscale // window_size + 1) * window_size
        self.fusion_layer = SwinFusion(upscale=upscale, img_size=(height, width),
                    window_size=window_size, embed_dim=128, Fusion_num_heads=[8, 8, 8],
                    Re_num_heads=[8], mlp_ratio=2, upsampler='', in_chans=128, ape=True,
                    drop_path_rate=0.)
        if self.use_skip:
            self.end_conv = BasicConv2d(256, 128, kernel_size=1)

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

        for i in range(2, 5):
            x_lidar_features[str(i)] = F.interpolate(
                x_lidar_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)
        for i in range(0, 5):
            x_rgb_features[str(i)] = F.interpolate(
                x_rgb_features[str(i)], size=proj_size, mode='bilinear', align_corners=True)
            
        x_lidar_features_cat = self.conv_before_fusion_lidar(torch.cat(list(x_lidar_features.values()), dim=1))
        x_rgb_features_cat = self.conv_before_fusion_rgb(torch.cat(list(x_rgb_features.values()), dim=1))
        out = self.fusion_layer(x_lidar_features_cat, x_rgb_features_cat)
        if self.use_skip:
            out = self.end_conv(torch.cat([out, x_lidar_features_cat], dim=1))

        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            out = [out]
            for i in range(2, 5):
                out.append(self.aux_heads["layer{}".format(i)](x_lidar_features["{}".format(i)]))
                out.append(self.aux_heads["layer{}".format(i)](x_rgb_features["{}".format(i)]))
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


class SwinFusion(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Fusion_depths=[2, 2], Re_depths=[4], 
                 Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinFusion, self).__init__()
        num_out_ch = in_chans
        num_feat = 64
        embed_dim_temp = int(embed_dim / 2)

        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.Fusion_num_layers = len(Fusion_depths)
        self.Re_num_layers = len(Re_depths)

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        # num_patches = 32768  ###TODO: change to image size multiplication
        num_patches = 12288
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)

        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]  # stochastic depth decay rule
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]  # stochastic depth decay rule
        
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.num_features)
        self.norm_Fusion_B = norm_layer(self.num_features)
        
        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Re_depths[i_layer],
                         num_heads=Re_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body_Fusion = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)

        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
            self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features_Fusion(self, x, y):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)
            # y = layer(y, x, x_size)
            

        x = self.norm_Fusion_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)  # B L C
        y = self.patch_unembed(y, x_size)
        x = torch.cat([x, y], 1)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_after_body_Fusion(x))
        
        return x

    def forward_features_Re(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x, x_size)

        x = self.norm_Re(x)  # B L C
        x = self.patch_unembed(x, x_size)
        # Convolution 
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x

    def forward(self, x, y):
        # print("Initializing the model")

        x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)

        _, _, H, W = x.shape            
        
        return x[:, :, :H*self.upscale, :W*self.upscale]

    # def flops(self):
    #     flops = 0
    #     H, W = self.patches_resolution
    #     flops += H * W * 3 * self.embed_dim * 9
    #     flops += self.patch_embed.flops()
    #     for i, layer in enumerate(self.layers_Fusion):
    #         flops += layer.flops()
    #     for i, layer in enumerate(self.layers_Re):
    #         flops += layer.flops()
    #     flops += H * W * 3 * self.embed_dim * self.embed_dim
    #     flops += self.upsample.flops()
    #     return flops    


if __name__ == "__main__":
    model = Fusion(20, True, use_skip=True).cuda()
    overfit_test(model, 1, (3, 64, 192), (5, 64, 192))

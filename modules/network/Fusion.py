import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.getcwd())
sys.path.append("modules")
from overfit_test import overfit_test

from torch.nn import functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from third_party.SwinFusion.models.network_swinfusion1 import RSTB, CRSTB, PatchUnEmbed, PatchEmbed, Upsample, UpsampleOneStep
from new_cenet import CENet
from Mask2Former_RGB import Backbone_RGB
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

    def __init__(self, nclasses, img_prop):

        super(Fusion, self).__init__()

        W, H = img_prop["width"], img_prop["height"]

        if W == 512:
            self.fusion_w = 192
            self.fusion_h = 64
        elif W == 1024:
            self.fusion_w = 384
            self.fusion_h = 64
        else:
            NotImplementedError("You have to design a proper fusion module on your own.")

        inplanes = 128

        self.Fusion_num_heads=[8, 8]
        self.Fusion_depths = [2, 2]

        self.vis_att = False


        """RGB Backbone"""
        self.rgb_backbone = Backbone_RGB(nclasses)

        if W == 512:
            w_dict = torch.load(
                "logs/mask2former_rgb_samelidarview_flip_100/SENet_valid_best",
                map_location=lambda storage, loc: storage)
        elif W == 1024:
            w_dict = torch.load(
                "logs/mask2former_rgb_samelidarview_flip_100_384width/SENet_valid_best",
                map_location=lambda storage, loc: storage)
        else:
            NotImplementedError("There is no pretrained model for this range-view resolution.")

        self.rgb_backbone.load_state_dict(w_dict['state_dict'], strict=True)

        for param in self.rgb_backbone.parameters():
            param.requires_grad = True

        for param in self.rgb_backbone.backbone.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False

        self.rgb_backbone.backbone.class_predictor = nn.Linear(256, inplanes)

        self.bn_rgb = nn.BatchNorm2d(inplanes)

        for module in self.rgb_backbone.backbone.model.pixel_level_module.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
                module.track_running_stats = False

        """Lidar Backbone"""
        self.cenet = CENet(nclasses)

        if W == 512:
            w_dict = torch.load(
            "logs/cenet_100_rangeaugs/SENet_valid_best",
                                    map_location=lambda storage, loc: storage)
        elif W == 1024:
            w_dict = torch.load(
            "logs/cenet_100_1024_rangeaugs/SENet_valid_best",
                                    map_location=lambda storage, loc: storage)
        else:
            NotImplementedError("There is no pretrained model for this range-view resolution.")

        self.cenet.load_state_dict(w_dict['state_dict'], strict=True)

        for param in self.cenet.parameters():
            param.requires_grad = False

        self.cenet.eval()

        for module in self.cenet.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
                module.track_running_stats = False

        # self.cenet.semantic_output = nn.Conv2d(128, 128, 1)

        self.bn_lidar = nn.BatchNorm2d(inplanes)

        """Fusion"""
        upscale = 4
        window_size = 8
        height = (64 // upscale // window_size + 1) * window_size
        width = (512 // upscale // window_size + 1) * window_size
        self.fusion_layer = SwinFusion(upscale=upscale, img_size=(height, width), Fusion_depths=self.Fusion_depths,
                    window_size=window_size, embed_dim=128, Fusion_num_heads=self.Fusion_num_heads,
                    Re_num_heads=[8], Re_depths=[4], mlp_ratio=2, upsampler='', in_chans=128, ape=True,
                    drop_path_rate=0., img_h=self.fusion_h, img_w=self.fusion_w)
        
        self.bn_fusion = nn.BatchNorm2d(inplanes)

        """Final layers"""
        self.semantic_output = nn.Conv2d(128, nclasses, 1)


    def forward(self, lidar, rgb):

        assert lidar.shape[1] == 7, "Lidar input must have 7 channels include a mask channel and a empty mask channel"
        lidar, _, mask_front = lidar[:, :5, :, :], lidar[:, 5, :, :].bool(), lidar[:, 6, :, :].bool()
        bs = lidar.shape[0]

        """Pre-trained Feature Extraction"""
        with torch.no_grad():
            lidar_out, x_lidar = self.cenet(lidar, rgb)
            lidar_out = lidar_out[0]

        out = lidar_out.clone().detach()
        lidar_features = x_lidar.clone().detach()

        del x_lidar, lidar_out

        lidar_features = self.bn_lidar(lidar_features)
        x_lidar_fusion_list = self.bbox2(lidar_features * mask_front.unsqueeze(dim=1))

        x_lidar_fusion = []
        x_lidar_fusion_size = []

        for i in range(bs):
            x_lidar_fusion_size.append(x_lidar_fusion_list[i].shape[2:])
            x_lidar_fusion.append(F.interpolate(x_lidar_fusion_list[i],
                                       size=[self.fusion_h, self.fusion_w], mode='bilinear', align_corners=False))
        x_lidar_fusion = torch.cat(x_lidar_fusion, dim=0)

        x_rgb = self.rgb_backbone(x_lidar_fusion, rgb)
        x_rgb = self.bn_rgb(x_rgb)

        """Fusion Module"""
        fusion_out, _ = self.fusion_layer(x_lidar_fusion, x_rgb)

        x_fusion = torch.zeros_like(lidar_features)

        for i in range(bs):
            current_fusion_out = F.interpolate(fusion_out[i].unsqueeze(0), size=x_lidar_fusion_size[i], mode='bilinear', align_corners=False)
            ymin, ymax, xmin, xmax = self.indices[i]
            x_fusion[i, :, ymin:ymax+1, xmin:xmax+1] = current_fusion_out
        x_fusion = self.bn_fusion(x_fusion)
        del x_lidar_fusion, fusion_out

        mask_fusion = mask_front.unsqueeze(dim=1).repeat(1, out.shape[1], 1, 1)
        x_fusion = self.semantic_output(x_fusion)
        x_fusion = F.softmax(x_fusion, dim=1)
        out[mask_fusion] = x_fusion[mask_fusion]

        return out
    
    def bbox2(self, batch):
        out = []
        self.indices = []
        for i in range(batch.shape[0]):
            rows = torch.any(batch[i, 0], dim=1) #TODO: Check if first channel is enough for that
            cols = torch.any(batch[i, 0], dim=0)
            ymin, ymax = torch.where(rows)[0][[0, -1]]
            xmin, xmax = torch.where(cols)[0][[0, -1]]
            out.append(batch[i, :, ymin:ymax+1, xmin:xmax+1].unsqueeze(0))
            self.indices.append([ymin, ymax, xmin, xmax])
        # out = torch.cat(out, dim=0)
        return out


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
                 img_w=192, img_h=64, **kwargs):
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
        num_patches = img_h * img_w
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
        
        x_aux = []
        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)
            # y = layer(y, x, x_size)
            ## Auxiliary outputs:
            x_temp = self.norm_Fusion_A(x)
            x_temp = self.patch_unembed(x_temp, x_size)
            y_temp = self.norm_Fusion_A(y)
            y_temp = self.patch_unembed(y_temp, x_size)
            x_aux.append(torch.cat([x_temp, y_temp], 1))
            

        x = self.norm_Fusion_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)  # B L C
        y = self.patch_unembed(y, x_size)
        x = torch.cat([x, y], 1)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_after_body_Fusion(x))
        
        return x, x_aux

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

        x, x_aux = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)

        _, _, H, W = x.shape            
        
        return x[:, :, :H*self.upscale, :W*self.upscale], x_aux

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
    model = Fusion(20).cuda()
    overfit_test(model, 1, (3, 64, 192), (7, 64, 192))

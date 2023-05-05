import torch
import torch.nn as nn
import sys
sys.path.append('/home/son/project/praktikum/CENet-fusion')
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from ResNet import BasicBlock, BasicConv2d, conv1x1, conv3x3
from third_party.SwinFusion.models.network_swinfusion import Cross_BasicLayer, PatchUnEmbed
from third_party.Mask2Former.mask2former.modeling.backbone.swin import PatchEmbed 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F
from transformers import AutoImageProcessor, Mask2FormerModel

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
        brach_type: semantic, instance or panoptic
        only_enc: whether to only use the encoder
        in_size: input size of the image
    """

    def __init__(self, name, use_att = False, fuse_all = True, branch_type = "semantic", only_enc = False, in_size = (64,512)) -> None:
        super().__init__()
        assert name in ["resnet50", "mask2former"], "Backbone name must be either resnet50 or mask2former" 
        assert branch_type in ["semantic", "instance", "panoptic"], "Branch type must be either semantic, instance or panoptic"
        
        def get_smaller(in_size, scale):
            """
            Returns a tuple of the smaller size given an input size and scale factor.
            """
            return (in_size[0]//scale, in_size[1]//scale)
        
        self.name = name
        self.in_size = in_size
        output_res = [1,2,4,8] #* Define output resolution 
        if use_att and fuse_all:
            output_res = [1, 1, 4, 4] #* only fuse with attention in 2 resolutions to save resources
        if only_enc:
            hidden_dim = [96 , 192, 384, 768]
            self.layer_list = ["encoder_hidden_states"] * 4
        else:
            hidden_dim = [256] * 4
            self.layer_list = ["decoder_hidden_states"] * 3 + ["decoder_last_hidden_state"]  
           
        
        if name == "resnet50":
            hidden_dim = [2048] + [1024] * 3
            self.layer_list = ["out"] + ["aux"] * 3
            self.processor = None
            self.backbone = deeplabv3_resnet50(weights='DEFAULT').backbone
            
            if fuse_all:
                self.upscale_layers = nn.ModuleList([
                    Upscale_head(hidden_dim[0], 128, (8,16), get_smaller(in_size, output_res[0])),
                    Upscale_head(hidden_dim[1], 128, (8,16), get_smaller(in_size, output_res[1])),
                    Upscale_head(hidden_dim[2], 128, (8,16), get_smaller(in_size, output_res[2])),
                    BasicConv2d(hidden_dim[3],128,1)
                ])
            else:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128, (8,16), in_size)])

            
        elif name == "mask2former":
            weight = "facebook/mask2former-swin-tiny-cityscapes-{}".format(branch_type)
            self.backbone = Mask2FormerModel.from_pretrained(weight).pixel_level_module
    
            if fuse_all:
                self.upscale_layers = nn.ModuleList([
                    Upscale_head(hidden_dim[0], 128, get_smaller(in_size, 4), get_smaller(in_size, output_res[0])),
                    Upscale_head(hidden_dim[1], 128, get_smaller(in_size, 8), get_smaller(in_size, output_res[1])),
                    Upscale_head(hidden_dim[2], 128, get_smaller(in_size, 16), get_smaller(in_size, output_res[2])),
                    Upscale_head(hidden_dim[3], 128, get_smaller(in_size, 32), get_smaller(in_size, output_res[3]))
                ])
            else:
                self.upscale_layers = nn.ModuleList([Upscale_head(hidden_dim[0], 128, get_smaller(in_size, 4), in_size)])
              
        else:
            raise NotImplementedError
        
        for p in self.backbone.parameters():
            p.requires_grad = False
        

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x, output_hidden_states=True) if self.name != "resnet50" else self.backbone(x)
        outputs = []
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

class Fusion_with_resnet(nn.Module):
    """
    All scale fusion with resnet50 backbone
    Basic fusion with torch.cat + conv1x1
    use_att: use cross attention or not
    fusion_scale: "all" or "main_late" or "main_early"
    """
    def __init__(self, nclasses, aux=True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64, use_att = False, fusion_scale='all'):
        super(Fusion_with_resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_att = use_att
        self.fusion_scale = fusion_scale

        self.backbone = BackBone(name="resnet50", use_att=use_att, fuse_all=aux, only_enc=False, brach_type="semantic")

        """BASEMODEL"""
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

        """FUSION LAYERS"""
        if not use_att:
            self.fusion = BasicConv2d(256, 128, kernel_size=1, padding=0)
        else:
            self.fusion = Cross_SW_Attention()


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
        #* get RGB features
        
        rgb_out = self.backbone(rgb)
        

        #* get projection features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_1 = self.layer1(x)  # 1

        """FUSION MAIN SCALES"""
        if self.fusion_scale == "main_early":
            if not self.use_att:
                x = self.fusion(torch.cat([x, rgb_out[0]], dim=1))
                x_1 = self.fusion(torch.cat([x_1, rgb_out[0]], dim=1))
            else:
                x = self.fusion(x, rgb_out[0])
                x_1 = self.fusion(x_1, rgb_out[0])
    

        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        """FUSION ALL(AUX) SCALES"""
        if self.fusion_scale == "all":
            if not self.use_att:
                x = self.fusion(torch.cat([x, rgb_out[0]], dim=1))
                x_1 = self.fusion(torch.cat([x_1, rgb_out[0]], dim=1))
                x_2 = self.fusion(torch.cat([x_2, rgb_out[1]], dim=1))
                x_3 = self.fusion(torch.cat([x_3, rgb_out[2]], dim=1))
                x_4 = self.fusion(torch.cat([x_4, rgb_out[3]], dim=1))

        #TODO: consider implement "all" early fusion for cross attention: very F***ing expensive

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)

        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)

        """LATE FUSION"""
        if self.fusion_scale == "main_late":
            if not self.use_att:
                out = self.fusion(torch.cat([out, rgb_out[0]], dim=1))
            else:
                out = self.fusion(out, rgb_out[0])
                #! Size reduction due to patch size: patch_size = 1 can be heavy to calculate
                if out.shape != rgb_out[0].shape:
                    out = F.interpolate(out, size=rgb_out[0].shape[2:], mode='bilinear', align_corners=True)

        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)

            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)
        if self.aux:
            return [out, res_2, res_3, res_4]
        else:
            return out



class Upscale_head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, size_in: tuple, size_out: tuple) -> None:
        super(Upscale_head, self).__init__()
        inter_channels = in_channels // 4
        h_in, w_in = size_in
        h_out, w_out = size_out
        self.size_inter = ((h_in+h_out)//2, (w_in+w_out)//2)
        self.size_out = size_out
        
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, out_channels, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_inter = self.conv1(F.interpolate(x, size=self.size_inter, mode='bilinear', align_corners=True))
        x_inter = self.relu(self.bn1(x_inter))

        x_out = self.conv2(F.interpolate(x_inter, size=self.size_out, mode='bilinear', align_corners=True))
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
    def __init__(self, pe_type = "abs", fusion_type="x_main", embed_dim=128, input_size =(64, 512), patch_size=2, depth=3, num_heads = 8, window_size=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.pe_type = pe_type
        self.fusion_type = fusion_type
        self.window_size = window_size

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=128, embed_dim=embed_dim,
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
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            self.absolute_pos_embed_B = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed_A, std=0.02)
            trunc_normal_(self.absolute_pos_embed_B, std=0.02)
        elif self.pe_type == 'adaptive':
            pass
        ##TODO: add adaptive PE
        else:
            raise NotImplementedError(f"Not support pe type {self.pe_type}.")
        
        self.pos_drop = nn.Dropout(p=drop)

        """Input resolution = Patch's resolution"""
        self.fusion_layers = Cross_BasicLayer(dim=embed_dim,
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
            self.conv_fusion = BasicBlock(256,128,1)
            self.norm_B = norm_layer(self.embed_dim)

    
    def forward(self,x,y):
        

        #* Divide image into patches
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        x_size = (x.shape[2], x.shape[3])

        #* Add position embedding
        x = (x + self.absolute_pos_embed_A).flatten(2).transpose(1, 2)
        y = (y + self.absolute_pos_embed_B).flatten(2).transpose(1, 2)

        x = self.pos_drop(x)
        y = self.pos_drop(y)
        if self.fusion_type == "x_main":
            x, _= self.fusion_layers(x, y, x_size)
            x = self.norm_A(x)  # B L C
            x = self.patch_unembed(x, x_size)

        elif self.fusion_type == "double_fuse":
            x, y= self.fusion_layers(x, y, x_size)
            x = self.norm_Fusion_A(x)  # B L C
            x = self.patch_unembed(x, x_size)

            y = self.norm_Fusion_B(y)  # B L C
            y = self.patch_unembed(y, x_size)

            x = torch.cat([x, y], 1)
            x = self.conv_fusion(x)

        else:
            raise NotImplementedError(f"Not support fusion type {self.fusion_type}.")
        
        return x
        
    

if __name__ == "__main__":
    import time
    model = Fusion_with_resnet(20).cuda()
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
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
        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)
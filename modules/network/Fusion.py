import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from modules.network.ResNet import BasicBlock, BasicConv2d, conv1x1, conv3x3
#from third_party.SwinFusion.models.network_swinfusion
from torch.nn import functional as F

""" This is a torch way of getting the intermediate layers  
 return_layers = {"layer4": "out"}
 if aux:
      return_layers["layer3"] = "aux" 
 backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
"""

class Fusion_with_resnet(nn.Module):
    """
    All scale fusion with resnet50 backbone
    Basic fusion with torch.cat + conv1x1
    Late/Early fusion
    fusion_type: "cat+conv" or "cross_attention"
    fusioN_scale: "all" or "two"
    """
    def __init__(self, nclasses, aux=True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64, early_fusion=True, fusion_type='cat+conv', fusion_scale='all'):
        super(Fusion_with_resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.early_fusion = early_fusion
        self.fusion_type = fusion_type
        self.fusion_scale = fusion_scale

        """BACKBONE AND UPSAMPLING HEADS"""
        self.backbone = deeplabv3_resnet50(weights='DEFAULT', aux_loss = aux).backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.upscale_main = Upscale_head(2048, 128, (8,16), (64,512))
        if early_fusion:
            self.upscale_aux_1 = Upscale_head(1024, 128, (8,16), (32,256))
            self.upscale_aux_2 = Upscale_head(1024, 128, (8,16), (16,128))
            self.upscale_aux_3 = BasicConv2d(1024,128,1)

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
        if fusion_type == 'cat+conv':
            self.fusion = BasicConv2d(256, 128, kernel_size=1, padding=0)
        
        elif fusion_type == 'cross_attention':
            self.fusion = Cross_SW_Attention()

        else:
            raise NotImplementedError

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
        with torch.no_grad():
            rgb = self.backbone(rgb)
        rgb_main = self.upscale_main(rgb['out'])
        

        #* get projection features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        """FUSION ALL SCALES (early)"""
        if self.early_fusion:
            rgb_aux_1 = self.upscale_aux_1(rgb['aux'])
            rgb_aux_2 = self.upscale_aux_2(rgb['aux'])
            rgb_aux_3 = self.upscale_aux_3(rgb['aux'])
            if self.fusion_type == "cat+conv":
                x = self.fusion(torch.cat([x, rgb_main], dim=1))
                x_1 = self.fusion(torch.cat([x_1, rgb_main], dim=1))
                x_2 = self.fusion(torch.cat([x_2, rgb_aux_1], dim=1))
                if self.fusion_scale == "all":
                    x_3 = self.fusion(torch.cat([x_3, rgb_aux_2], dim=1))
                    x_4 = self.fusion(torch.cat([x_4, rgb_aux_3], dim=1))


        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)

        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)

        """LATE FUSION"""
        if not self.early_fusion:
            if self.fusion_type == "cat+conv":
                out = self.fusion(torch.cat([out, rgb_main], dim=1))

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
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,y):
        pass
        


if __name__ == "__main__":
    import time
    model = Fusion_with_resnet(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        input_3D = torch.randn(2, 5, 64, 512).cuda()
        input_rgb = torch.randn(2, 3, 64, 512).cuda()
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
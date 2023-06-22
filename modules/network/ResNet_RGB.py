import sys
sys.path.append("modules")
import torch.nn as nn
import torch
from torch.nn import functional as F
from overfit_test import overfit_test


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet_tfbu(nn.Module):
    def __init__(self,
                 nclasses, aux=True,
                 block=BasicBlock,
                 layers=[3, 4, 6, 3],
                 if_BN=True, dilation=1,
                 norm_layer=nn.BatchNorm2d,
                 groups=1, inplanes=128,
                 width_per_group=64) -> None:
        super().__init__()
        self.if_BN = if_BN
        self.groups = groups
        self.aux = aux
        self.base_width = width_per_group
        self._norm_layer = norm_layer
        self.dilation = dilation
        self.inplanes = inplanes
        
        self.initial_conv = nn.Sequential(BasicConv2d(5, 64, kernel_size=3, padding=1),
                                          BasicConv2d(64, 128, kernel_size=3, padding=1),
                                          BasicConv2d(128, 128, kernel_size=3, padding=1))

        self.unet_layers = nn.ModuleList()
        for i, j in enumerate([1, 2, 2, 2]):
            self.unet_layers.append(self._make_layer(block, 128, layers[i], stride=j))
        
        #self.upscale_conv = [BasicConv2d(256, 128, kernel_size=3, padding=1)]*4
        self.upscale_conv = nn.ModuleList()
        for i in range(4):
            self.upscale_conv.append(BasicConv2d(256, 128, kernel_size=3, padding=1))

        if self.aux:
            self.aux_heads = nn.ModuleList()
            for i in range(4):
                self.aux_heads.append(BasicConv2d(128, nclasses, kernel_size=3, padding=1))
            
        self.semantic_output = nn.Conv2d(128, nclasses, 1)    
            
    def forward(self, x, _):
        x = self.initial_conv(x)
        x_features = {'0': x}
        del x
        for i in range(0, 4):
            x_features[str(i + 1)] = self.unet_layers[i](x_features[str(i)])
        
        for i in reversed(range(1,5)):
            target_size = x_features[str(i-1)].size()[2:]
            current_feat = F.interpolate(x_features[str(i)], size=target_size, mode='bilinear', align_corners=True)
            if i == 1:
                # * x_features['0'] and ['1'] are the same size so no need interpolation
                current_feat = x_features[str(i)]
            x_features[str(i-1)] = self.upscale_conv[i-1](torch.cat([x_features[str(i-1)], current_feat], dim=1))
        
        out = [F.softmax(self.semantic_output(x_features['0']), dim=1)]
        if self.aux:
            out.append(F.softmax(self.aux_heads[0](x_features['1']), dim=1))
            for i in range(1,4):
                upscaled_feats = F.interpolate(x_features[str(i+1)], size=x_features['0'].size()[2:], mode='bilinear', align_corners=True)
                out.append(F.softmax(self.aux_heads[i](upscaled_feats), dim=1))

        return out

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
    

class ResNet_34_RGB(nn.Module):
    def __init__(self, nclasses, aux=False, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34_RGB, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(3, 64, kernel_size=3, padding=1)
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

    def forward(self, lidar, x):

        target_size = lidar.shape[2:4]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x_1 = F.interpolate(x_1, size=target_size, mode='bilinear', align_corners=True)
        res_2 = F.interpolate(x_2, size=target_size, mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=target_size, mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=target_size, mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)
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

class ResNet_34BB(ResNet_34_RGB):
    def forward(self,pixel_values, output_hidden_states=False):
        x  = pixel_values
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)

        class Output():
            pass
            
        output = Output()
        setattr(output, 'decoder_last_hidden_state', out)
        setattr(output, 'decoder_hidden_states', [x_2, x_3, x_4])
        setattr(output, 'encoder_last_hidden_state', None)
        setattr(output, 'encoder_hidden_states', None)

        return output
        
if __name__ == "__main__":
    model = ResNet_34_RGB(20, True).cuda()
    overfit_test(model, 6, False)
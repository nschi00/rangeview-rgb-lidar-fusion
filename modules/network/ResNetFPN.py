import sys
sys.path.append("modules")
import torch
import torch.nn as nn
from torch.nn import functional as F
from overfit_test import overfit_test

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    

class RGB_Net(nn.Module):
    def __init__(self, nclasses):
        super(RGB_Net, self).__init__()

        self.backbone = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        self.backbone.classifier[4] = nn.Conv2d(512, nclasses, kernel_size=(1, 1), stride=(1, 1))
        self.backbone.aux_classifier[4] = nn.Conv2d(256, nclasses, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, lidar, x):

        assert lidar.shape[1] == 7, "Lidar input must have 7 channels including a mask channel and an empty mask channel"
        lidar, _, mask_front = lidar[:, :5, :, :], lidar[:, 5, :, :].bool(), lidar[:, 6, :, :].bool()
        bs = lidar.shape[0]

        x = self.backbone(x)

        x_out = F.interpolate(x['out'], (64, 192), mode="bilinear", align_corners=True)
        x_aux = F.interpolate(x['aux'], (64, 192), mode="bilinear", align_corners=True)

        out = F.softmax(x_out, dim=1)
        aux = F.softmax(x_aux, dim=1)

        out = [out, aux]

        return out
    
        
if __name__ == "__main__":
    model = RGB_Net(20).cuda()
    overfit_test(model, 200, (3, 64, 192), (5, 64, 192))
import sys
sys.path.append("modules")
import torch.nn as nn
import torch
from torch.nn import functional as F
from overfit_test import overfit_test

from transformers import Mask2FormerModel
    

class Backbone_RGB(nn.Module):
    def __init__(self, nclasses, aux=False):
        super(Backbone_RGB, self).__init__()

        weight = "facebook/mask2former-swin-tiny-cityscapes-semantic"

        self.backbone = Mask2FormerModel.from_pretrained(weight)
        # self.backbone.pixel_level_module.requires_grad = False
        self.class_predictor = nn.Linear(256, nclasses)

    def forward(self, lidar, x):

        target_size = lidar.shape[2:4]

        with torch.no_grad():
            x = self.backbone(**x, output_hidden_states=True)

        class_queries_logits = self.class_predictor(x.transformer_decoder_last_hidden_state)
        masks_queries_logits = x.masks_queries_logits[-1]
        del x

        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=target_size, mode="bilinear", align_corners=False
        )

        masks_classes = class_queries_logits.softmax(dim=-1)
        # [batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        out = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        out = F.softmax(out, dim=1)

        return out 
        
if __name__ == "__main__":
    model = Backbone_RGB(20).cuda()
    overfit_test(model, 200, (3, 64, 192), (5, 64, 192))

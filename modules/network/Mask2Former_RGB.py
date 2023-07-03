import sys
sys.path.append("modules")
import torch.nn as nn
import torch
from torch.nn import functional as F
from overfit_test import overfit_test

from transformers import Mask2FormerForUniversalSegmentation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
    

class Backbone_RGB(nn.Module):
    def __init__(self, nclasses, aux=False):
        super(Backbone_RGB, self).__init__()

        weight = "logs/lidar_front_25_rgb_mask2former_retrainTransPixDec_rgbfullsizeNormalize_noDropPoints/models--facebook--mask2former-swin-tiny-cityscapes-semantic/snapshots/414e5f2219df2e2e56703b856bbce4cc7b7046b9"

        self.backbone = Mask2FormerForUniversalSegmentation.from_pretrained(weight)

        for param in self.backbone.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False
        
        self.backbone.class_predictor = nn.Linear(256, nclasses)

    def forward(self, lidar, x):

        target_size = lidar.shape[2:4]
        bs = lidar.shape[0]

        # with torch.no_grad():
        x = self.backbone(x, output_hidden_states=True)

        out = self.post_process_semantic_segmentation(x, [target_size for _ in range(bs)])

        out = F.softmax(out, dim=1)

        return out 
    
    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        masks_classes = class_queries_logits.softmax(dim=-1)
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_segmentation.append(resized_logits[0])
        else:
            semantic_segmentation = [semantic_segmentation[i] for i in range(segmentation.shape[0])]

        return torch.stack(semantic_segmentation, dim=0)
        
if __name__ == "__main__":
    model = Backbone_RGB(20).cuda()
    overfit_test(model, 200, (3, 64, 192), (5, 64, 192))
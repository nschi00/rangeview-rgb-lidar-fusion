import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerConfig, Mask2FormerModel
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


# load Mask2Former fine-tuned on Cityscapes instance segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-instance")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

## INPUT SIZE = 1x5x64x512 (1024 or 2048)

class ExtractFeatureBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(5, 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(5, 3, kernel_size=5, padding=2)
        self.last_conv = nn.Conv2d(9, 3, kernel_size=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x1, x2, x3 = self.batchnorm(x1), self.batchnorm(x2), self.batchnorm(x3)
        x1, x2, x3 = self.activation(x1), self.activation(x2), self.activation(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.last_conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x



class Mask2FormerAdaptive(nn.Module):
    def __init__(self, nclasses, aux):
        super(Mask2FormerAdaptive, self).__init__()
        self.aux = aux
        self.nclasses = nclasses
        self.feature_extractor = ExtractFeatureBlock()
        self.configuration = Mask2FormerConfig().from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.configuration.num_labels = self.nclasses
        self.configuration.ignore_value = 0
        self.configuration.id2label ={0:"unlabeled",
                                    1:"car",
                                    2:"bicycle",
                                    3:"motorcycle",
                                    4:"truck",
                                    5:"other-vehicle",
                                    6:"person",
                                    7:"bicyclist",
                                    8:"motorcyclist",
                                    9:"road",
                                    10:"parking",
                                    11:"sidewalk",
                                    12:"other-ground",
                                    13:"building",
                                    14:"fence",
                                    15:"vegetation",
                                    16:"trunk",
                                    17:"terrain",
                                    18:"pole",
                                    19:"traffic-sign"}
        
        self.configuration.label2id ={
                                        "unlabeled": 0,
                                        "car": 1,
                                        "bicycle": 2,
                                        "motorcycle": 3,
                                        "truck": 4,
                                        "other-vehicle": 5,
                                        "person": 6,
                                        "bicyclist": 7,
                                        "motorcyclist": 8,
                                        "road": 9,
                                        "parking": 10,
                                        "sidewalk": 11,
                                        "other-ground": 12,
                                        "building": 13,
                                        "fence": 14,
                                        "vegetation": 15,
                                        "trunk": 16,
                                        "terrain": 17,
                                        "pole": 18,
                                        "traffic-sign": 19
                                    }

        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.model = Mask2FormerForUniversalSegmentation(self.configuration)
        config = self.model.config

        print(config)

    def forward(self, x, rgb):
        x = self.feature_extractor(x)
        x = list(x)
        input_mask2former = self.processor(images=x, return_tensors="pt")
        input_mask2former.data['pixel_values'] = input_mask2former.data['pixel_values'].cuda()
        input_mask2former.data['pixel_mask'] = input_mask2former.data['pixel_mask'].cuda()

        output_mask2former = self.model(**input_mask2former)

        class_queries_logits = output_mask2former["class_queries_logits"]
        masks_queries_logits = output_mask2former["masks_queries_logits"]

        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]
        
        target_sizes = []
        for image in x:
            target_sizes.append(image.shape[1::])
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
                semantic_map = F.softmax(resized_logits[0],dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        temp = torch.stack(semantic_segmentation)
        return temp

if __name__ == "__main__":
    import time
    model = Mask2FormerAdaptive(20, aux = False).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        input_3D = torch.randn(2, 5, 64, 512).cuda()
        input_rgb = torch.randn(2, 3, 452, 1032).cuda()
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
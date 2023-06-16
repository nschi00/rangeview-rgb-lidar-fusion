# import requests
# import torch
# from PIL import Image
# from transformers import AutoImageProcessor, Mask2FormerModel


# # load Mask2Former fine-tuned on Cityscapes semantic segmentation
# processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
# model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
# model = model.pixel_level_module

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(images=image, return_tensors="pt")
# a = inputs["pixel_values"]
# with torch.no_grad():
#     outputs = model(a, output_hidden_states=True)

# # model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# class_queries_logits = outputs.class_queries_logits
# masks_queries_logits = outputs.masks_queries_logits

# # you can pass them to processor for postprocessing
# predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)

from torchvision.models import resnet50
import torch
from torchvision.models._utils import IntermediateLayerGetter
return_layers = {"layer4": "out"}
return_layers["layer3"] = "aux"

model = resnet50(pretrained=False)
a = torch.load('resnet50-19c8e357.pth')
model.load_state_dict(torch.load('resnet50-19c8e357.pth'))
model = IntermediateLayerGetter(model, return_layers=return_layers)
a = torch.randn(1, 3, 224, 224)
out = model(a)
print(out)
print(model)
import requests
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerModel, Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerTransformerModule, Mask2FormerMaskedAttentionDecoder
#from third_party.Mask2Former.mask2former.modeling.criterion import SetCriterion, HungarianMatcher




# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
config = model.config
model = model.pixel_level_module


image = Image.open("aachen_000000_000019_leftImg8bit.png")
semantic_map = Image.open("aachen_000000_000019_gtFine_labelTrainIds.png")
inputs = processor(images, segmentation_maps=semantic_map, return_tensors="pt")
a = inputs["pixel_values"]
with torch.no_grad():
    outputs = model(a, output_hidden_states=True)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)




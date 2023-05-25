import requests
import torch
from PIL import Image
from modules.ioueval import iouEval
from transformers import Mask2FormerImageProcessor, Mask2FormerModel, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerTransformerModule, Mask2FormerMaskedAttentionDecoder
#from third_party.Mask2Former.mask2former.modeling.criterion import SetCriterion, HungarianMatcher
import evaluate

metric = evaluate.load("mean_iou")



# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
#processor = processor.cuda()

config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")

model = Mask2FormerForUniversalSegmentation(config)


image = Image.open("aachen_000000_000019_leftImg8bit.png")

label = Image.open("aachen_000000_000019_gtFine_labelTrainIds.png")

inputs = processor(image, segmentation_maps=label, return_tensors="pt", device='cuda')


optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
pytorch_total_params = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
print("Number of parameters: ", pytorch_total_params / 1000000, "M")
time_train = []

evaluator = iouEval(20, device='cuda', ignore=19)
import numpy as np
for i in range(20000):
    model.train()
    #with torch.no_grad():

    outputs = model(**inputs)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    evaluator.addBatch(predicted_semantic_map.cuda(), np.asarray(label))
    accuracy = evaluator.getacc()
    jaccard, class_jaccard = evaluator.getIoUMissingClass()
    print("Loss: ", loss.item())
    print("Accuracy: ", accuracy.item())
    print("IoU: ", jaccard.item())


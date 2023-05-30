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

# def visualize(self, img_list: list):
#     plot_length = len(img_list)
#     for i, img in enumerate(img_list):
#       if img.shape[0] == 3:
#         img_list[i] = img.permute(1, 2, 0)
#       else:
#         print(img.max(), img.min())
#         # img_list[i] = self.map(img, self.learning_map_inv)
#         # img_list[i] = self.map(img_list[i], self.color_map)
#       # if max(img) != 255 and min(img) != 0:
#       #   img_list[i] = self.map(img, self.learning_map_inv)
#       #   img_list[i] = self.map(img_list[i], self.color_map)
#       # elif type(img) == torch.Tensor:
#       #   img_list[i] = img.permute(1, 2, 0).numpy().astype(np.uint8) * 255.0
    
    
#     fig, axs = plt.subplots(plot_length, 1)
#     for i in range(plot_length):
#       axs[i].imshow(img_list[i])
#       axs[i].axis('off')
      
#     # Adjust spacing between subplots
#     plt.subplots_adjust(hspace=0.1)

#     # Show the plot
#     plt.show()
    
# def get_division_angles(self, division):
#         # Calculate the angle per division
#         angle_per_division = 360.0 / division

#         # Calculate the start angle for the first division
#         first_start_angle = -angle_per_division / 2.0
#         first_end_angle = angle_per_division / 2.0

#         # Create a list to store the start and end angles for each division
#         division_angles = [(first_start_angle, first_end_angle)]

#         # Calculate the start and end angles for the remaining divisions
#         for i in range(1, division):
#             start_angle = division_angles[i-1][1]
#             if start_angle < 0.0:
#                 end_angle = start_angle - angle_per_division
#             else:
#                 end_angle = start_angle + angle_per_division
#             if end_angle > 180.0:
#                 end_angle -= 360.0
#             division_angles.append((start_angle, end_angle))

#         return division_angles
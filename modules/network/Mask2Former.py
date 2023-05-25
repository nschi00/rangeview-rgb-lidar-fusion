import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'Mask2Former'))
sys.path.append(os.getcwd())
from ioueval import *
import warnings
warnings.simplefilter("ignore", UserWarning)


from network.ResNet import BasicConv2d, ResNet_34BB, BasicBlock
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import Tensor
from typing import List
from detectron2.config import get_cfg

from transformers import Mask2FormerModel, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
#from third_party.Mask2Former.mask2former.modeling.transformer_decoder import MultiScaleMaskedTransformerDecoder,  Mask2FormerSinePositionEmbedding, Mask2FormerMaskedAttentionDecoder, Mask2FormerMaskedAttentionDecoderOutput
from third_party.Mask2Former.mask2former import config as m2f_config
from third_party.Mask2Former.mask2former.modeling.matcher import HungarianMatcher
from third_party.Mask2Former.mask2former.modeling.criterion import SetCriterion
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from image_processing_mask2former import Mask2FormerImageProcessor



# class Mask2FormerBasePrototype(nn.Module):
#     """
#     All scale fusion with resnet50 backbone
#     Basic fusion with torch.cat + conv1x1
#     use_att: use cross attention or not
#     fusion_scale: "all_early" or "all_late" or "main_late" or "main_early"
#     name_backbone: backbone for RGB, only "resnet50" at the moment possible
#     branch_type: semantic, instance or panoptic
#     stage: whether to only use the enc, pixel_decoder or combined pixel/transformer decoder output (combination)
#     """

#     def __init__(self, nclasses, aux=True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
#                  norm_layer=None, groups=1, width_per_group=64,
#                  backbone_3d_ptr="best_pretrained/CENet_64x512_67_6",
#                  config_path="third_party/Mask2Former/configs/cityscapes/semantic-segmentation/"
#                  "swin/maskformer2_swin_tiny_bs16_90k.yaml",
#                  freeze_bb=True,
#                  l_weight=None):
#         super(Mask2FormerBasePrototype, self).__init__()

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.backbone_3d = ResNet_34(nclasses, True, block,
#                                      layers, if_BN,
#                                      width_per_group=width_per_group,
#                                      groups=groups)

#         if backbone_3d_ptr is not None:
#             w_dict = torch.load(backbone_3d_ptr, map_location=lambda storage, loc: storage)
#             self.backbone_3d.load_state_dict(w_dict['state_dict'], strict=True)

#         if freeze_bb:
#             for param in self.backbone_3d.parameters():
#                 param.requires_grad = False

#         self.l_weight = l_weight
#         self.n_classes = nclasses
#         self.cfg = self.prepare_cfg(config_path)
#         self.aux = aux
#         self.decoder = MultiScaleMaskedTransformerDecoder(self.cfg, in_channels=128, mask_classification=True)
#         self.mask_conv = BasicConv2d(128, 256, 1)
#         self.class_predictor = nn.Linear(256, nclasses)
#         self.mask_pred = BasicConv2d(100, nclasses, 1)
#         self.criteria = self.get_criteror()
#         self.processor = self.prepare_processor()

#     def forward(self, x, _):
#         # * get RGB features
#         x = self.backbone_3d.feature_extractor(x)
#         out = self.decoder(x=x[1:], mask_features=self.mask_conv(x[0]))
#         return out

#     def prepare_cfg(self, config_path):
#         cfg = get_cfg()
#         add_deeplab_config(cfg)
#         m2f_config.add_maskformer2_config(cfg)
#         cfg.merge_from_file(config_path)
#         cfg.MODEL.MASK_FORMER.DEC_LAYERS = 4
#         return cfg
    
#     def prepare_processor(self):
#         processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
#         processor.do_normalize = False
#         processor.do_resize = False
#         processor.do_rescale = False
#         processor.size_divisor = 0
#         return processor
    
#     def get_criteror(self):
#         deep_supervision = self.cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
#         #no_object_weight = self.cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

#         # loss weights
#         class_weight = self.cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
#         dice_weight = self.cfg.MODEL.MASK_FORMER.DICE_WEIGHT
#         mask_weight = self.cfg.MODEL.MASK_FORMER.MASK_WEIGHT

#         # building criterion
#         matcher = HungarianMatcher(
#             cost_class=class_weight,
#             cost_mask=mask_weight,
#             cost_dice=dice_weight,
#             num_points=self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
#         )

#         weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

#         if deep_supervision:
#             dec_layers = self.cfg.MODEL.MASK_FORMER.DEC_LAYERS
#             aux_weight_dict = {}
#             for i in range(dec_layers - 1):
#                 aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
#             weight_dict.update(aux_weight_dict)

#         losses = ["labels", "masks"]

#         self.criterion = SetCriterion(
#             self.n_classes-1,
#             matcher=matcher,
#             weight_dict=weight_dict,
#             eos_coef=self.l_weight,
#             losses=losses,
#             num_points=self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
#             oversample_ratio=self.cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
#             importance_sample_ratio=self.cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
#         )
    
#     def loss(self, outputs, targets):
#         H, W = targets.shape[1:]
#         targets = list(targets)
#         bs = len(targets)
#         # generate random list of dummy inputs
#         inputs = [torch.rand(3, H, W) for _ in range(bs)]
        
#         targets = self.processor(inputs, segmentation_maps=targets, return_tensors="pt")
#         labels = []
#         for label, mask in zip(targets["class_labels"], targets["mask_labels"]):
#             labels.append({"labels": label.cuda(), "masks": mask.cuda()})
        
#         loss_dict = self.criterion(outputs, labels)
#         for k in list(loss_dict.keys()):
#             if k in self.criterion.weight_dict:
#                 loss_dict[k] *= self.criterion.weight_dict[k]
#             else:
#                 # remove this loss if not specified in `weight_dict`
#                 loss_dict.pop(k)
#         return loss_dict
    
#     def eval_seg(self, logits, mask):
#         masks_classes = logits.softmax(dim=-1)[..., :-1]
#         masks_probs = mask.sigmoid()
#         segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
#         batch_size = logits.shape[0]
#         semantic_segmentation = segmentation.argmax(dim=1)
#         semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
#         semantic_segmentation = torch.stack(semantic_segmentation, dim=0)
#         assert batch_size == semantic_segmentation.shape[0]
#         return semantic_segmentation


class Mask2FormerBasePrototype(nn.Module):
    def __init__(self,
                 norm_layer=None, 
                 backbone_3d_ptr="best_pretrained/CENet_64x512_67_6",
                 freeze_bb=True,
                 l_weight=None):
        super(Mask2FormerBasePrototype, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        block=BasicBlock 
        layers=[3, 4, 6, 3]
        if_BN=True,
        groups=1 
        width_per_group=64
        nclasses = 20
        backbone_3d = ResNet_34BB(nclasses, True, block,
                                     layers, if_BN,
                                     width_per_group=width_per_group,
                                     groups=groups)
        self.prepare_config()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        #temp = self.model.config
        
        #self.model.criterion.empty_weight = l_weight if l_weight is not None else self.model.criterion.weight_dict
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.processor.do_normalize = False
        self.processor.do_resize = False
        self.processor.do_rescale = False
        self.processor.size_divisor = 1
        self.processor.size = (64, 512)
        self.conv1 = nn.Conv2d(5, 3, 1)
        # if backbone_3d_ptr is not None:
        #     w_dict = torch.load(backbone_3d_ptr, map_location=lambda storage, loc: storage)
        #     backbone_3d.load_state_dict(w_dict['state_dict'], strict=True)
        # self.model.model.pixel_level_module = backbone_3d
        # if freeze_bb:
        #     for param in self.backbone_3d.parameters():
        #         param.requires_grad = False
                
    def forward(self, lidar, _, label=None):
        # * get RGB features
        B, _, H, W = lidar.shape
        inputs_dummy = torch.rand(B, 3, H, W)
        inputs = self.processor(list(inputs_dummy),segmentation_maps=label, return_tensors="pt")
        inputs["pixel_values"] = self.conv1(lidar)
        inputs["pixel_mask"] = inputs["pixel_mask"].cuda()
        for i in range(B):
            inputs["mask_labels"][i] = inputs["mask_labels"][i].cuda()
            inputs["class_labels"][i] = inputs["class_labels"][i].cuda()
            
        x = self.model(**inputs)
        return x
    
    def seg_eval(self, outputs, target_sizes=None):
        
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]


        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
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
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            mini = torch.min(semantic_segmentation)
            maxi = torch.max(semantic_segmentation)
            import pandas as pd
            s = pd.DataFrame(semantic_segmentation.cpu().numpy()[0])
            p = s.describe()
            print(p)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return torch.stack(semantic_segmentation, dim=0)
    
    @classmethod
    def prepare_config(self):
        self.model_config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
        self.model_config.hidden_dim = 128
        self.model_config.feature_size = 128
        self.model_config.mask_feature_size = 128
        self.model_config.num_labels = 20
        self.model_config.no_object_weight = 0.0
        
    
class Input_Dummy():
    def __init__(self, pixel_values, pixel_mask):
        self.pixel_values = pixel_values
        self.pixel_mask = pixel_mask
        
    def shape(self):
        return self.pixel_values.shape

if __name__ == "__main__":
    import time
    l_weight = torch.ones(20).cuda()
    #l_weight[0] = 0.0
    model = Mask2FormerBasePrototype(20, l_weight=l_weight).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    
    evaluator = iouEval(21, device='cuda', ignore=20)
    
    input_3D = torch.randn(2, 5, 64, 512).cuda()
    input_rgb = torch.rand(2, 3, 452, 1032).cuda()
    label = torch.randint(0, 20, (2, 64, 512)).cuda()
    for i in range(20000):
        
        model.train()
        #with torch.no_grad():
        start_time = time.time()
        outputs = model(input_3D, input_rgb, label)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp = [(input_3D.shape[2], input_3D.shape[3])] * 2
        predicted_semantic_map = model.seg_eval(outputs, target_sizes=temp)
        evaluator.addBatch(predicted_semantic_map, label)
        accuracy = evaluator.getacc()
        
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)
        fwt = time.time() - start_time
        time_train.append(fwt)
        # print("Forward time per img: %.3f (Mean: %.3f)" % (
        #     fwt / 1, sum(time_train) / len(time_train) / 1))
        # time.sleep(0.15)

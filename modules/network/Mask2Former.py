import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'modules', 'network'))
sys.path.append(os.path.join(os.getcwd(), 'third_party', 'Mask2Former'))
sys.path.append(os.getcwd())

import warnings
warnings.simplefilter("ignore", UserWarning)


from ResNet import BasicConv2d, ResNet_34, BasicBlock
import torch.nn.functional as F
import torch.nn as nn
import torch
from detectron2.config import get_cfg
from typing import List
from torch import Tensor
from third_party.Mask2Former.mask2former.modeling.transformer_decoder import MultiScaleMaskedTransformerDecoder
from third_party.Mask2Former.mask2former import config as m2f_config
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

class Mask2FormerBasePrototype(nn.Module):
    """
    All scale fusion with resnet50 backbone
    Basic fusion with torch.cat + conv1x1
    use_att: use cross attention or not
    fusion_scale: "all_early" or "all_late" or "main_late" or "main_early"
    name_backbone: backbone for RGB, only "resnet50" at the moment possible
    branch_type: semantic, instance or panoptic
    stage: whether to only use the enc, pixel_decoder or combined pixel/transformer decoder output (combination)
    """

    def __init__(self, nclasses, aux=True, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 norm_layer=None, groups=1, width_per_group=64,
                 backbone_3d_ptr="best_pretrained/CENet_64x512_67_6",
                 config_path="third_party/Mask2Former/configs/cityscapes/semantic-segmentation/"
                 "swin/maskformer2_swin_tiny_bs16_90k.yaml",
                 freeze_bb=True):
        super(Mask2FormerBasePrototype, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.backbone_3d = ResNet_34(nclasses, True, block,
                                     layers, if_BN,
                                     width_per_group=width_per_group,
                                     groups=groups)

        if backbone_3d_ptr is not None:
            w_dict = torch.load(backbone_3d_ptr, map_location=lambda storage, loc: storage)
            self.backbone_3d.load_state_dict(w_dict['state_dict'], strict=True)

        if freeze_bb:
            for param in self.backbone_3d.parameters():
                param.requires_grad = False

        cfg = get_cfg()
        add_deeplab_config(cfg)
        m2f_config.add_maskformer2_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.MASK_FORMER.DEC_LAYERS = 4
        self.aux = aux
        self.decoder = MultiScaleMaskedTransformerDecoder(cfg, in_channels=128, mask_classification=True)
        self.mask_conv = BasicConv2d(128, 256, 1)
        self.class_predictor = nn.Linear(256, nclasses)
        self.mask_pred = BasicConv2d(100, nclasses, 1)

    def forward(self, x, _):
        # * get RGB features
        x = self.backbone_3d.feature_extractor(x)

        out = self.decoder(x=x[1:], mask_features=self.mask_conv(x[0]))
        outputs = [F.softmax(self.mask_pred(out['pred_masks']), dim=1)]
        if not self.aux:
            return outputs
        for aux_out in out['aux_outputs']:
            outputs.append(F.softmax(self.mask_pred(aux_out['pred_masks']), dim=1))

        return outputs

# class Mask2FormerTransformerModule(nn.Module):
#     """
#     The Mask2Former's transformer module.
#     """

#     def __init__(self, in_features: int, config: Mask2FormerConfig):
#         super().__init__()
#         hidden_dim = config.hidden_dim
#         self.num_feature_levels = 3
#         self.position_embedder = Mask2FormerSinePositionEmbedding(
#             num_pos_feats=hidden_dim // 2, normalize=True)
#         self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
#         self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
#         self.input_projections = []

#         for _ in range(self.num_feature_levels):
#             if in_features != hidden_dim or config.enforce_input_projection:
#                 self.input_projections.append(
#                     nn.Conv2d(in_features, hidden_dim, kernel_size=1))
#             else:
#                 self.input_projections.append(nn.Sequential())

#         self.input_projections = nn.ModuleList(self.input_projections)
#         self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
#         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

#     def forward(
#         self,
#         multi_scale_features: List[Tensor],
#         mask_features: Tensor,
#         output_hidden_states: bool = False,
#         output_attentions: bool = False,
#     ) -> Mask2FormerMaskedAttentionDecoderOutput:
#         multi_stage_features = []
#         multi_stage_positional_embeddings = []
#         size_list = []

#         for i in range(self.num_feature_levels):
#             size_list.append(multi_scale_features[i].shape[-2:])
#             multi_stage_positional_embeddings.append(
#                 self.position_embedder(multi_scale_features[i], None).flatten(2))
#             multi_stage_features.append(
#                 self.input_projections[i](multi_scale_features[i]).flatten(2)
#                 + self.level_embed.weight[i][None, :, None]
#             )

#             # Flatten (batch_size, num_channels, height, width) -> (height*width, batch_size, num_channels)
#             multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(
#                 2, 0, 1)
#             multi_stage_features[-1] = multi_stage_features[-1].permute(
#                 2, 0, 1)

#         _, batch_size, _ = multi_stage_features[0].shape

#         # [num_queries, batch_size, num_channels]
#         query_embeddings = self.queries_embedder.weight.unsqueeze(
#             1).repeat(1, batch_size, 1)
#         query_features = self.queries_features.weight.unsqueeze(
#             1).repeat(1, batch_size, 1)

#         decoder_output = self.decoder(
#             inputs_embeds=query_features,
#             multi_stage_positional_embeddings=multi_stage_positional_embeddings,
#             pixel_embeddings=mask_features,
#             encoder_hidden_states=multi_stage_features,
#             query_position_embeddings=query_embeddings,
#             feature_size_list=size_list,
#             output_hidden_states=output_hidden_states,
#             output_attentions=output_attentions,
#             return_dict=True,
#         )

#         return decoder_output


if __name__ == "__main__":
    import time
    model = Mask2FormerBasePrototype(20).cuda()
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
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
        print("Forward time per img: %.3f (Mean: %.3f)" % (
            fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)

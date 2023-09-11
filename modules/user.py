#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from tqdm import tqdm   
from modules.network.Fusion import Fusion
from modules.network.RangePreprocessFusion import RangePreprocessFusion
from modules.network.ResNet import ResNet_34
from modules.network.new_cenet import CENet
from modules.network.Mask2Former_RGB import Backbone_RGB
from modules.network.RangeFormer import RangeFormer
from postproc.KNN import KNN

def convert_relu_to_softplus(model, act):
    for child_name, child in model.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, act)
        else:
            convert_relu_to_softplus(child, act)

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split

    # get the data
    from dataset.kitti.parser import Parser
    self.parser = Parser(root=self.datadir,
                        train_sequences=self.DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=self.DATA["split"]["valid"],
                        test_sequences=None,
                        labels=self.DATA["labels"],
                        color_map=self.DATA["color_map"],
                        learning_map=self.DATA["learning_map"],
                        learning_map_inv=self.DATA["learning_map_inv"],
                        sensor=self.ARCH["dataset"]["sensor"],
                        max_points=self.ARCH["dataset"]["max_points"],
                        batch_size=1,
                        workers=0,
                        gt=True,
                        shuffle_train=False,
                        only_RGB=self.ARCH["train"]["model"] == "mask2former")

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        activation = eval("nn." + self.ARCH["train"]["act"] + "()")
        if self.ARCH["train"]["pipeline"] == "res":
            self.model = ResNet_34(self.parser.get_n_classes())
            convert_relu_to_softplus(self.model, activation)
        elif self.ARCH["train"]["pipeline"] == "rangeformer":
            self.model = RangeFormer(self.parser.get_n_classes(), self.parser.get_resolution())
        elif self.ARCH["train"]["pipeline"] == "fusion":
            if self.ARCH["train"]["model"] == "cenet":
                self.model = CENet(self.parser.get_n_classes())
            elif self.ARCH["train"]["model"] == "swinfusion":
                self.model = Fusion(self.parser.get_n_classes(), self.ARCH["dataset"]["sensor"]["img_prop"])
            elif self.ARCH["train"]["model"] == "mask2former":
                self.model = Backbone_RGB(self.parser.get_n_classes())
            else:
                raise SyntaxError("Invalid name chosen. Choose one of 'cenet', 'swinfusion', or 'mask2former'.")
#     print(self.model)
    w_dict = torch.load(modeldir + "/SENet_valid_best",
                        map_location=lambda storage, loc: storage)
    self.model.load_state_dict(w_dict['state_dict'], strict=True)
    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    print(self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    self.range_preprocess = RangePreprocessFusion([0,0,0,0])
    self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      for i, (proj_data, rgb_data) in tqdm(enumerate(loader), total=len(loader)):
        # first cut to rela size (batch size one allows it)
        proj_in, proj_mask, _ ,query_mask, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints= proj_data
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
            proj_in = proj_in.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()
            proj_mask = proj_mask.cuda()
            rgb_data = rgb_data.cuda()
            query_mask = query_mask.cuda()
            if self.post:
                proj_range = proj_range.cuda()
                unproj_range = unproj_range.cuda()
        end = time.time()

        with torch.cuda.amp.autocast(enabled=True):
            if self.ARCH["train"]["pipeline"] == "rangeformer":
                proj_in, proj_mask, _= self.range_preprocess(proj_in, 
                                                            [proj_mask, None], 
                                                            _, 
                                                            False)
            elif self.ARCH["train"]["pipeline"] == "fusion":
                proj_in, proj_mask, _ = self.range_preprocess(proj_in, 
                                                            [proj_mask, query_mask], 
                                                            _,
                                                            False)
            else:
                proj_in, _, _ = self.range_preprocess(proj_in, 
                                                    [None, None], 
                                                    _,
                                                    False)
            proj_output = self.model(proj_in, rgb_data)
        if type(proj_output) != list:
            proj_output = [proj_output]
        if proj_output[0].dim() == 4:
            proj_output[0] = proj_output[0].squeeze(0)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
#             # nla postproc
#             proj_unfold_range, proj_unfold_pre = NN_filter(proj_range, proj_argmax)
#             proj_unfold_range=proj_unfold_range.cpu().numpy()
#             proj_unfold_pre=proj_unfold_pre.cpu().numpy()
#             unproj_range = unproj_range.cpu().numpy()
#             #  Check this part. Maybe not correct (Low speed caused by for loop)
#             #  Just simply change from
#             #  https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI/blob/7f90b45a765b8bba042b25f642cf12d8fccb5bc2/semantic_inference.py#L177-L202
#             for jj in range(len(p_x)):
#                 py, px = p_y[jj].cpu().numpy(), p_x[jj].cpu().numpy()
#                 if unproj_range[jj] == proj_range[py, px]:
#                     unproj_argmax = proj_argmax[py, px]
#                 else:
#                     potential_label = proj_unfold_pre[0, :, py, px]
#                     potential_range = proj_unfold_range[0, :, py, px]
#                     min_arg = np.argmin(abs(potential_range - unproj_range[jj]))
#                     unproj_argmax = potential_label[min_arg]

        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)

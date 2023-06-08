import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random

class RangePreprocess():
    def __init__(self, aug_prob = None) -> None:
        if aug_prob is None:
            self.aug_prob = [0.9, 0.2, 0.9, 1.0] #Mix, Paste, Union, Shift
        else:
            self.aug_prob = aug_prob
    
    def __call__(self, data, mask, label, training=False):
        data = torch.cat([data, mask.unsqueeze(1)], dim=1)
        if not training:
            return data, mask, label.long()
        bs = data.shape[0]
        out_scan = []
        out_label = []
        for i in range(bs):
            j = random.randint(0, bs-1)
            while j == i:
                j = random.randint(0, bs-1)
            scan_a, scan_b = data[i].clone(), data[j]
            label_a, label_b = label[i].clone(), label[j]

            if torch.rand(1) < self.aug_prob[0]:
                scan_a, label_a = self.RangeMix(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[1]:
                scan_a, label_a = self.RangePaste(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[2]:
                scan_a, label_a = self.RangeUnion(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[3]:
                scan_a, label_a = self.RangeShift(scan_a, label_a)
            out_scan.append(scan_a)
            out_label.append(label_a)
                
        out_scan = torch.stack(out_scan)
        out_label = torch.stack(out_label)
        out_mask = out_scan[:, -1, :, :]
        return out_scan, out_mask, out_label.long()
    
    def RangeUnion(self, scan_a, label_a, scan_b, label_b, k_union=0.5):
        mask = scan_a[-1, :, :]
        void = mask == 0
        mask_temp = torch.rand(void.shape, device=void.device) <= k_union
    # * Only fill 50% of the void points
        void = void.logical_and(mask_temp)
        scan_a[:, void], label_a[void] = scan_b[:, void], label_b[void]
        return scan_a, label_a

    def RangePaste(self, scan_a, label_a, scan_b, label_b, tail_classes=None):
        if tail_classes is None:
            tail_classes = [ 2,  3,  4,  5,  6,  7,  8, 16, 18, 19]
        for tail_class in tail_classes:
            pix = label_b == tail_class
            scan_a[:, pix] = scan_b[:, pix]
            label_a[pix] = label_b[pix]
        return scan_a, label_a
    
    def RangeShift(self, scan_a, label_a):
        _, h, w = scan_a.shape
        p = torch.randint(low=int(0.25*w), high=int(0.75*w), size=(1,))
        scan_a = torch.cat([scan_a[:, :, p:], scan_a[:, :, :p]], dim=2)
        label_a = torch.cat([label_a[:, p:], label_a[:, :p]], dim=1)
        # scan_a = torch.cat(scan_a[:, p:, :], scan_a[:, :p, :], dim = 1)
        # label_a = torch.cat(label_a[p:, :], label_a[:p, :], dim = 1)
        return scan_a, label_a
    
    
    def RangeMix(self, scan_a, label_a, scan_b, label_b, mix_strategies= [2, 3, 4, 5, 6]):
        _, h, w = scan_a.shape
        k_mix = random.choice(mix_strategies)
        index = random.choice(range(k_mix))
        mix_h_s = int(h / k_mix) * (index)
        mix_h_e = int(h / k_mix) * (index + 1)
        scan_a[:, mix_h_s:mix_h_e, :] = scan_b[:, mix_h_s:mix_h_e, :]
        label_a[mix_h_s:mix_h_e, :] = label_b[mix_h_s:mix_h_e, :]
        return scan_a, label_a
        
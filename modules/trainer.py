#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from common.avgmeter import *
from torch.utils.tensorboard import SummaryWriter
from common.sync_batchnorm.batchnorm import convert_model
from modules.scheduler.warmupLR import *
from modules.ioueval import *
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.scheduler.cosine import CosineAnnealingWarmUpRestarts
from dataset.kitti.parser import Parser
from modules.network.ResNet import ResNet_34, ResNet_tfbu
from modules.network.Fusion import Fusion, FusionDouble
from modules.network.Mask2Former import Mask2FormerBasePrototype
from tqdm import tqdm
from modules.losses.boundary_loss import BoundaryLoss
from collections import defaultdict

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=".pth"):
    # Save the weights
    torch.save(to_save, logdir + "/SENet" + suffix)


def convert_relu_to_softplus(model, act):
    for child_name, child in model.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(model, child_name, act)
        else:
            convert_relu_to_softplus(child, act)


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.datadir = datadir
        self.log = logdir
        self.path = path
        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        self.parser = Parser(root=self.datadir,
                             # DATA["split"]["valid"] + DATA["split"]["train"] if finetune with valid
                             train_sequences=DATA["split"]["train"],
                             valid_sequences=DATA["split"]["valid"],
                             test_sequences=None,
                             labels=DATA["labels"],
                             color_map=DATA["color_map"],
                             learning_map=DATA["learning_map"],
                             learning_map_inv=DATA["learning_map_inv"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.ARCH["train"]["batch_size"],
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=True,
                             overfit=self.ARCH["train"]["overfit"],
                             share_subset_train=self.ARCH["train"]["share_subset_train"])

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            # map actual class to xentropy class
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights

        # ignore the ones necessary to ignore
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)
        F_config = defaultdict(lambda: None)
        with torch.no_grad():
            activation = eval("nn." + self.ARCH["train"]["act"] + "()")
            if self.ARCH["train"]["pipeline"] == "res":
                self.model = ResNet_tfbu(self.parser.get_n_classes(),
                                       self.ARCH["train"]["aux_loss"])
                convert_relu_to_softplus(self.model, activation)
            elif self.ARCH["train"]["pipeline"] == "fusion":
                F_config = ARCH["fusion"]
                # self.model = Fusion(nclasses=self.parser.get_n_classes(),
                #                     aux=self.ARCH["train"]["aux_loss"],
                #                     use_att=F_config["use_att"],
                #                     fusion_scale=F_config["fuse_all"],
                #                     name_backbone=F_config["name_backbone"],
                #                     branch_type=F_config["branch_type"],
                #                     stage=F_config["stage"])
                self.model = FusionDouble(nclasses=self.parser.get_n_classes())
                #convert_relu_to_softplus(self.model, activation)
            else:
                self.model = Mask2FormerBasePrototype(nclasses=self.parser.get_n_classes(),
                                                      aux=self.ARCH["train"]["aux_loss"])


        save_to_log(self.log, 'model.txt', str(self.model))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params / 1000000, "M")
        print("Overfitting samples: ", self.ARCH["train"]["overfit"])

        # if F_config["name_backbone"]:
        #     print("{}_{}_{}_{}". format(F_config["name_backbone"],
        #                                 "ca" if F_config["use_att"] else "conv",
        #                                 F_config["fuse_all"],

        #                                 "" if self.ARCH["train"]["aux_loss"] else "noaux"))
        #     if F_config["name_backbone"] == "mask2former":
        #         print("{}_{}".format(F_config["branch_type"], F_config["stage"]))

        #     print("Please verify your settings before continue.")
        #     time.sleep(7)

        save_to_log(self.log, 'model.txt', "Number of parameters: %.5f M" % (
            pytorch_total_params / 1000000))
        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(
                self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()

        """
        TODO: create different optimizers and schedulers for different part of the model
        Example:
        backbone can be retrained with a lower learning rate
        main resnet model prefer cosine annealing with max lr = 0.01
        swin fusion prefer adam with MULTISTEPLR (need to check document)
        """

        scheduler_type = self.ARCH["train"]["scheduler"]
        scheduler_config = self.ARCH["train"][scheduler_type]
        steps_per_epoch = self.parser.get_train_size()
        lr = scheduler_config["lr"] if scheduler_type == "decay" else scheduler_config["min_lr"]
        momentum = self.ARCH["train"]["momentum"]

        # * Create Adam optimizer for cross attention fusion module
        self.att_optimizer = None
        self.att_scheduler = None
        if F_config["use_att"]:
            fusion_params = self.model.fusion_layer.parameters()
            rest_params = [p for n, p in self.model.named_parameters() if "fusion_layer" not in n]

            self.att_optimizer = optim.AdamW(fusion_params,
                                            lr=F_config["lr"],
                                            weight_decay=F_config["w_decay"])

            # self.att_scheduler = optim.lr_scheduler.MultiStepLR(self.att_optimizer,
            #                                                     milestones=F_config["scheduler_milestones"],
            #                                                     gamma=F_config["scheduler_gamma"])

        else:
            rest_params = self.model.parameters()

        if ARCH["train"]["pipeline"] == "m2f":
            lr = ARCH["train"]["adamw"]["lr"]
            w_decay = ARCH["train"]["w_decay"]
            self.clip_grad = ARCH["train"]["adamw"]["clip_grad"]
            self.optimizer = optim.AdamW(rest_params, lr=lr, weight_decay=w_decay)
        else:
            self.optimizer = optim.SGD(rest_params,
                                       lr=lr,  # min_lr
                                       momentum=momentum,
                                       weight_decay=self.ARCH["train"]["w_decay"])

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=scheduler_config["first_cycle"] * steps_per_epoch,
                                                           T_mult=scheduler_config["cycle"],  # cycle
                                                           eta_max=scheduler_config["max_lr"],  # max_lr
                                                           T_up=scheduler_config["wup_epochs"] * steps_per_epoch,
                                                           gamma=scheduler_config["gamma"])  # gamma

        elif scheduler_type == "decay":
            up_steps = int(scheduler_config["wup_epochs"] * steps_per_epoch)  # wup_epochs
            final_decay = scheduler_config["lr_decay"] ** (1 / steps_per_epoch)  # lr_decay
            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=lr,  # lr
                                      warmup_steps=up_steps,
                                      momentum=momentum,
                                      decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            assert "fusion" not in self.ARCH["train"]["pipeline"], "no pretrained for fusion"
            try:
                w_dict = torch.load(path + "/SENet_valid_best",
                                    map_location=lambda storage, loc: storage)
            except:
                w_dict = torch.load(path,
                                    map_location=lambda storage, loc: storage)
            else:
                print("Loading model from: {}".format(path))
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
#             self.optimizer.load_state_dict(w_dict['optimizer'])
#             self.epoch = w_dict['epoch'] + 1
#             self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
#             self.info = w_dict['info']
            print("info", w_dict['info'])

    def calculate_estimate(self, epoch, iter):
        steps_per_epoch = self.parser.get_train_size()
        max_epochs = self.ARCH['train']['max_epochs']
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * (steps_per_epoch * max_epochs - (iter + 1 +
                       epoch * steps_per_epoch))) + int(self.batch_time_e.avg * self.parser.get_valid_size() *
                                                        (max_epochs - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if self.path is not None:  # *validate model if loaded from checkpoint
            acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                     evaluator=self.evaluator,
                                                     class_func=self.parser.get_xentropy_class_string,
                                                     color_fn=self.parser.to_color,
                                                     save_scans=self.ARCH["train"]["save_scans"])

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # train for 1 epoch

            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                              epoch=epoch,
                                              evaluator=self.evaluator,
                                              color_fn=self.parser.to_color,
                                              report=self.ARCH["train"]["report_batch"],
                                              show_scans=self.ARCH["train"]["show_scans"])

            # update info
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["lr"] = self.optimizer.param_groups[0]["lr"]
            self.info["att_lr"] = self.att_optimizer.param_groups[0]["lr"] if self.att_optimizer is not None else np.nan

            # remember best iou and save checkpoint
            # TODO: save attention optim and scheduler if necessary
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="_latest")

            if self.info['train_iou'] > self.info['best_train_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in training set so far, save model!")
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in validation so far, save model!")
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            Trainer.save_to_log(logdir=self.log,
                                logger=self.tb_logger,
                                info=self.info,
                                epoch=epoch,
                                w_summary=self.ARCH["train"]["save_summary"],
                                model=self.model_single,
                                img_summary=self.ARCH["train"]["save_scans"],
                                imgs=rand_img)
            save_to_log(self.log, 'log.txt', time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()))
        print('Finished Training')
        save_to_log(self.log, 'log.txt', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()))
        return

    def train_epoch(self, train_loader, epoch, evaluator, color_fn, report=10, show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        bd = AverageMeter()
        learning_rate = AverageMeter()
        attention_lr = AverageMeter()

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        scaler = torch.cuda.amp.GradScaler()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (proj_data, rgb_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            in_vol, proj_mask, proj_labels = proj_data[0:3]
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
            rgb_data = rgb_data.cuda()
            # compute output
            with torch.cuda.amp.autocast():
                out = self.model(in_vol, rgb_data)
                lamda = self.ARCH["train"]["lamda"]
                if type(out) is not list:  # IN CASE OF SINGLE OUTPUT
                    out = [out]

                # SUM POSITION LOSSES
                for j in range(len(out)):
                    if j == 0:
                        bdlosss = self.bd(out[j], proj_labels.long())
                        loss_mn = self.criterion(torch.log(out[j].clamp(
                            min=1e-8)), proj_labels) + 1.5 * self.ls(out[j], proj_labels.long())
                    else:
                        bdlosss += lamda * self.bd(out[j], proj_labels.long())
                        loss_mn += lamda * self.criterion(torch.log(out[j].clamp(
                            min=1e-8)), proj_labels) + 1.5 * self.ls(out[j], proj_labels.long())

                loss_m = loss_mn + bdlosss
                output = out[0]

            # * Compute attention gradient if exsits
            self.optimizer.zero_grad()
            self.att_optimizer.zero_grad() if self.att_optimizer is not None else None
            scaler.scale(loss_m.sum()).backward()
            # if self.clip_grad is not None or self.clip_grad == 0:
            #     torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad)
            scaler.step(self.optimizer)
            scaler.step(self.att_optimizer) if self.att_optimizer is not None else None
            scaler.update()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                if self.ARCH["train"]["overfit"]:
                    jaccard, class_jaccard = evaluator.getIoUMissingClass()
                else:
                    jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd.update(bdlosss.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            learning_rate.update(lr, 1)

            if self.att_optimizer is not None:
                att_lr = self.att_optimizer.param_groups[0]["lr"]
                attention_lr.update(att_lr, 1)
            else:
                att_lr = np.nan

            if show_scans:
                if i % self.ARCH["train"]["save_batch"] == 0:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(
                        depth_np, mask_np, pred_np, gt_np, color_fn)

                    directory = os.path.join(self.log, "train-predictions")
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = os.path.join(directory, str(i) + ".png")
                    cv2.imwrite(name, out)

            if i % self.ARCH["train"]["report_batch"] == 0:
                print('Lr: {lr:.3e} | '
                      'Att_lr: {att_lr:.0e} | '
                      'Epoch: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                      'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                      'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                          epoch, i, len(train_loader), batch_time=self.batch_time_t,
                          data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                          att_lr=att_lr, estim=self.calculate_estimate(epoch, i)))

                save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                            'Att_lr: {att_lr:.0e} |'
                            'Epoch: [{0}][{1}/{2}] | '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                            'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                            'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                            'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                                epoch, i, len(train_loader), batch_time=self.batch_time_t,
                                data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                                att_lr=att_lr, estim=self.calculate_estimate(epoch, i)))

            # * step scheduler
            self.scheduler.step()
            self.att_scheduler.step() if self.att_scheduler is not None else None

        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        self.model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (proj_data, rgb_data) in tqdm(enumerate(val_loader), total=len(val_loader)):
                in_vol, proj_mask, proj_labels = proj_data[0:3]
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()
                rgb_data = rgb_data.cuda()
                # compute output
                output = self.model(in_vol, rgb_data)

                if type(output) is list:
                    output = output[0]

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = self.criterion(log_out, proj_labels)
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))

                wces.update(wce.mean().item(), in_vol.size(0))

                if save_scans:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np,
                                               mask_np,
                                               pred_np,
                                               gt_np,
                                               color_fn)
                    rand_imgs.append(out)

                # measure elapsed time
                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            print('Validation set:\n'
                  'Time avg per batch {batch_time.avg:.3f}\n'
                  'Loss avg {loss.avg:.4f}\n'
                  'Jaccard avg {jac.avg:.4f}\n'
                  'WCE avg {wces.avg:.4f}\n'
                  'Acc avg {acc.avg:.3f}\n'
                  'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                 loss=losses,
                                                 jac=jaccs,
                                                 wces=wces,
                                                 acc=acc, iou=iou))

            save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                             'Time avg per batch {batch_time.avg:.3f}\n'
                                             'Loss avg {loss.avg:.4f}\n'
                                             'Jaccard avg {jac.avg:.4f}\n'
                                             'WCE avg {wces.avg:.4f}\n'
                                             'Acc avg {acc.avg:.3f}\n'
                                             'IoU avg {iou.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                            loss=losses,
                                                                            jac=jaccs,
                                                                            wces=wces,
                                                                            acc=acc, iou=iou))
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                self.info["valid_classes/" + class_func(i)] = jacc

        return acc.avg, iou.avg, losses.avg, rand_imgs

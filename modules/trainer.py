#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import cv2
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
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import OneCycleLR
from modules.network.Fusion import Fusion
from modules.network.new_cenet import CENet
from modules.network.Mask2Former_RGB import Backbone_RGB
from modules.network.RangePreprocessFusion import RangePreprocessFusion
from modules.network.RangePreprocessLidar import RangePreprocessLidar
from modules.network.ResNet import ResNet_34
from modules.network.RangeFormer import RangeFormer
from tqdm import tqdm

def get_Optim(model, config, iter_per_epoch = None):
    optimizer_cfg = config["optimizer"]
    scheduler_cfg = config["scheduler"]
    optim_name = optimizer_cfg["Name"]
    optimizer = eval("optim." + optim_name)
    try:
        fusion_params = model.fusion_layer.parameters()
    except:
        fusion_params = []
    rest_params = [p for n, p in model.named_parameters() if "fusion_layer" not in n]

    total_iter = iter_per_epoch * config["train"]["max_epochs"]
    # * F&B_mutiplier set to 1.0 will train all parameters with the same learning rate
    optimizer = optimizer(
            [{"params":fusion_params, "lr": optimizer_cfg[optim_name]["lr"] *
                                            optimizer_cfg["F&B_mutiplier"]}, 
             {"params": rest_params}], **optimizer_cfg[optim_name])

    # * Set up scheduler if needed
    if scheduler_cfg is not None and scheduler_cfg["Name"] != "None":
        name = scheduler_cfg["Name"]
        if name == "CosineAnnealingWarmupRestarts":
            scheduler_cfg[name]["max_lr"] = optimizer_cfg[optim_name]["lr"]
            scheduler_cfg[name]["first_cycle_steps"] = iter_per_epoch * scheduler_cfg[name]["first_cycle_steps"]
            scheduler_cfg[name]["warmup_steps"] = iter_per_epoch * scheduler_cfg[name]["warmup_steps"]
        elif name == "OneCycleLR":
            scheduler_cfg[name]["max_lr"] = optimizer_cfg[optim_name]["lr"]
            scheduler_cfg[name]["total_steps"] = total_iter
        scheduler = eval(name)
        scheduler = scheduler(optimizer, **scheduler_cfg[name])
    else:
        # * If no scheduler is needed, set up a dummy scheduler
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)

    return optimizer, scheduler


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=".pth"):
    # Save the weights
    torch.save(to_save, logdir +
               "/SENet" + suffix)

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
        self.DATA = DATA
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
                     "train_iou_front": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "valid_iou_front": 0,
                     "best_train_iou": 0,
                     "best_train_iou_front": 0,
                     "best_val_iou": 0,
                     "best_val_iou_front": 0}

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
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True,
                                          subset_ratio=self.ARCH["train"]["subset_ratio"],
                                          old_aug=True,
                                          only_RGB=(self.ARCH["train"]["model"]=="mask2former"))

        #self.range_preprocess = RangePreprocess([0.,0.,0.,.8]) #Mix, Paste, Union, Shift
        if self.ARCH["train"]["model"] == "cenet":
            self.range_preprocess = RangePreprocessLidar([0.5,0.2,0.5,.8]) #Mix, Paste, Union, Shift
        else:
            self.range_preprocess = RangePreprocessFusion([0.,0.,0.,0.]) #Mix, Paste, Union, Shift

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights


        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            activation = eval("nn." + self.ARCH["train"]["act"] + "()")
            if self.ARCH["train"]["pipeline"] == "res":
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])
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

        save_to_log(self.log, 'model.txt', str(self.model))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params/1000000, "M")
        print("Number of workers: ", self.ARCH["train"]["workers"])
        print("Batch size: ", self.ARCH["train"]["batch_size"])
        print("Subset ratio: ", self.ARCH["train"]["subset_ratio"])
        print("RangeAug prob: ", self.range_preprocess.aug_prob)

        save_to_log(self.log, 'model.txt', "Number of parameters: %.5f M" %(pytorch_total_params/1000000))
        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        from modules.losses.boundary_loss import BoundaryLoss
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()

        self.optimizer, self.scheduler = get_Optim(
                                        self.model, 
                                        self.ARCH, 
                                        self.parser.get_train_size())
        print(self.optimizer)
        print(self.scheduler)
        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SENet_valid_best",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.optimizer.load_state_dict(w_dict['optimizer'])
            self.epoch = w_dict['epoch'] + 1
            self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            #             self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
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
        self.front_evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.path is not None:
            acc, iou, loss, rand_img, iou_front = self.validate(val_loader=self.parser.get_valid_set(),
                                                    model=self.model,
                                                    criterion=self.criterion,
                                                    evaluator=self.evaluator,
                                                    front_evaluator=self.front_evaluator,
                                                    class_func=self.parser.get_xentropy_class_string,
                                                    color_fn=self.parser.to_color,
                                                    save_scans=self.ARCH["train"]["save_scans"])
            
            path = os.path.join(self.log, "valid-predictions")
            for i, img in enumerate(rand_img):
                if not os.path.isdir(path):
                    os.makedirs(path)
                img_path = os.path.join(path, str(i) + ".png")
                cv2.imwrite(img_path, img)
                

        # train for n epochs
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # train for 1 epoch

            acc, iou, loss, iou_front = self.train_epoch(train_loader=self.parser.get_train_set(),
                                            model=self.model,
                                            criterion=self.criterion,
                                            optimizer=self.optimizer,
                                            epoch=epoch,
                                            evaluator=self.evaluator,
                                            front_evaluator=self.front_evaluator,
                                            scheduler=self.scheduler,
                                            color_fn=self.parser.to_color,
                                            show_scans=self.ARCH["train"]["show_scans"])


            # update info
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou
            self.info["train_iou_front"] = iou_front

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")
            # save_checkpoint(state, self.log, suffix=""+str(epoch))

            if self.info['train_iou'] > self.info['best_train_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in training set so far, save model!")
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if self.info['train_iou_front'] > self.info['best_train_iou_front']:
                save_to_log(self.log, 'log.txt', "Best mean iou front in training set so far, save model!")
                print("Best mean iou front in training set so far, save model!")
                self.info['best_train_iou_front'] = self.info['train_iou_front']

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img, iou_front = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         front_evaluator=self.front_evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou
                self.info["valid_iou_front"] = iou_front

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                save_to_log(self.log, 'log.txt', "Best mean iou in validation so far, save model!")
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']
                self.info['best_val_iou_front'] = self.info['valid_iou_front']
                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            if self.info['valid_iou_front'] > self.info['best_val_iou_front']:
                save_to_log(self.log, 'log.txt', "Best mean iou front in validation so far, save model!")
                print("Best mean iou front in validation so far, save model!")
                self.info['best_val_iou_front'] = self.info['valid_iou_front']

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
            save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('Finished Training')
        save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, front_evaluator, scheduler, color_fn,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        bd = AverageMeter()
        iou_front = AverageMeter()
        learning_rate = AverageMeter()
        train=True
        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        scaler = torch.cuda.amp.GradScaler()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (proj_data, rgb_data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            in_vol, proj_mask, proj_labels, query_mask = proj_data[0:4]
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
                proj_mask = proj_mask.cuda()
                rgb_data = rgb_data.cuda()
                query_mask = query_mask.cuda()
            # compute output
            with torch.cuda.amp.autocast():
                if self.ARCH["train"]["pipeline"] == "rangeformer":
                    in_vol, proj_mask, proj_labels= self.range_preprocess(in_vol, 
                                                                          [proj_mask, None], 
                                                                          proj_labels, 
                                                                          training=train)
                elif self.ARCH["train"]["pipeline"] == "fusion":
                    if self.ARCH["train"]["model"] == "mask2former":
                        in_vol, proj_mask, proj_labels = self.range_preprocess(in_vol, 
                                                                          [proj_mask, query_mask], 
                                                                          proj_labels,
                                                                          training=False)
                    else:
                        in_vol, proj_mask, proj_labels = self.range_preprocess(in_vol, 
                                                                            [proj_mask, query_mask], 
                                                                            proj_labels,
                                                                            training=train)
                else:
                    in_vol, _, proj_labels = self.range_preprocess(in_vol, 
                                                                   [None, None], 
                                                                   proj_labels,
                                                                   False)
                out = model(in_vol, rgb_data)
                lamda = self.ARCH["train"]["lamda"]

                if type(out) is not list: # IN CASE OF SINGLE OUTPUT
                    out = [out]

                ## SUM POSITION LOSSES
                for j in range(len(out)):
                    if j == 0:
                        bdlosss = self.bd(out[j], proj_labels)
                        loss_mn = criterion(torch.log(out[j].clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(out[j], proj_labels)
                    else:
                        bdlosss += lamda*self.bd(out[j], proj_labels)
                        loss_mn += lamda*criterion(torch.log(out[j].clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(out[j], proj_labels)

                loss_m = loss_mn + bdlosss
                output = out[0]

            optimizer.zero_grad()
            scaler.scale(loss_m.sum()).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            loss = loss_m.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoUMissingClass()
                #! IoU for camera FoV
                query_mask = in_vol[:, 6, :, :].bool()
                proj_labels_front = proj_labels * query_mask
                if self.gpu:
                    proj_labels_front = proj_labels_front.cuda().long()
                front_evaluator.reset()
                front_evaluator.addBatch(argmax*query_mask, proj_labels_front)
                jaccard_front, class_jaccard = front_evaluator.getIoUMissingClass()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            iou_front.update(jaccard_front.item(), in_vol.size(0))
            bd.update(bdlosss.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            for g in self.optimizer.param_groups:
                lr = g["lr"]
            learning_rate.update(lr, 1)


            if show_scans:
                if i % self.ARCH["train"]["save_batch"] == 0:
                    # get the first scan in batch and project points
                    mask_np = proj_mask[0].cpu().numpy()
                    depth_np = in_vol[0][0].cpu().numpy()
                    pred_np = argmax[0].cpu().numpy()
                    gt_np = proj_labels[0].cpu().numpy()
                    out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

                    directory = os.path.join(self.log, "train-predictions")
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = os.path.join(directory, str(i) + ".png")
                    cv2.imwrite(name, out)


            if i % 10 == 0:
                print('Lr: {lr:.3e} | '
                      'Epoch: [{0}][{1}/{2}] | '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                      'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                      'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                      'IoU {iou.val:.3f} ({iou.avg:.3f}) | '
                      'IoU front {iou_front.val:.3f} ({iou_front.avg:.3f}) | [{estim}]'.format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    iou_front=iou_front, estim=self.calculate_estimate(epoch, i)))

                save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
                                                 'Epoch: [{0}][{1}/{2}] | '
                                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                                 'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                                                 'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                                                 'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                                                 'IoU {iou.val:.3f} ({iou.avg:.3f}) | '
                                                 'IoU front {iou_front.val:.3f} ({iou_front.avg:.3f}) | [{estim}]'.format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    iou_front=iou_front, estim=self.calculate_estimate(epoch, i)))
            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, iou_front.avg

    def validate(self, val_loader, model, criterion, evaluator, front_evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        iou_front = AverageMeter()
        rand_imgs = []
        train=False
        # switch to evaluate mode
        model.eval()
        evaluator.reset()
        front_evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (proj_data, rgb_data) in tqdm(enumerate(val_loader), total=len(val_loader)):
                in_vol, proj_mask, proj_labels, query_mask = proj_data[0:4]
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                    query_mask = query_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True)
                rgb_data = rgb_data.cuda()
                # compute output
                if self.ARCH["train"]["pipeline"] == "rangeformer":
                    in_vol, proj_mask, proj_labels= self.range_preprocess(in_vol, 
                                                                          [proj_mask, None], 
                                                                          proj_labels, 
                                                                          training=train)
                elif self.ARCH["train"]["pipeline"] == "fusion":
                    in_vol, proj_mask, proj_labels = self.range_preprocess(in_vol, 
                                                                          [proj_mask, query_mask], 
                                                                          proj_labels,
                                                                          training=train)
                else:
                    in_vol, _, proj_labels = self.range_preprocess(in_vol, 
                                                                   [None, None], 
                                                                   proj_labels,
                                                                   False)
                with torch.cuda.amp.autocast():
                    output = model(in_vol,rgb_data)
                if self.ARCH["train"]["aux_loss"]:
                    output = output[0]

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                # measure accuracy and record loss
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                query_mask = in_vol[:, 6, :, :].bool()
                proj_labels_front = proj_labels * query_mask
                if self.gpu:
                    proj_labels_front = proj_labels_front.cuda()
                front_evaluator.addBatch(argmax*query_mask, proj_labels_front)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(),in_vol.size(0))


                wces.update(wce.mean().item(),in_vol.size(0))

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
            jaccard, class_jaccard = evaluator.getIoUMissingClass()
            #! IoU for camera FoV
            jaccard_front, class_jaccard_front = front_evaluator.getIoUMissingClass()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            iou_front.update(jaccard_front.item(), in_vol.size(0))

            f_accuracy = front_evaluator.getacc()
            f_jaccard, f_class_jaccard = front_evaluator.getIoUMissingClass()
            iou_front.update(f_jaccard.item(), in_vol.size(0))
            print('Validation set:\n'
                  'Time avg per batch {batch_time.avg:.3f}\n'
                  'Loss avg {loss.avg:.4f}\n'
                  'Jaccard avg {jac.avg:.4f}\n'
                  'WCE avg {wces.avg:.4f}\n'
                  'Acc avg {acc.avg:.3f}\n'
                  'IoU avg {iou.avg:.3f}\n'
                  'IoU front avg {iou_front.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                 loss=losses,
                                                 jac=jaccs,
                                                 wces=wces,
                                                 acc=acc, iou=iou, iou_front=iou_front))

            save_to_log(self.log, 'log.txt', 'Validation set:\n'
                                             'Time avg per batch {batch_time.avg:.3f}\n'
                                             'Loss avg {loss.avg:.4f}\n'
                                             'Jaccard avg {jac.avg:.4f}\n'
                                             'WCE avg {wces.avg:.4f}\n'
                                             'Acc avg {acc.avg:.3f}\n'
                                             'IoU avg {iou.avg:.3f}\n'
                                             'IoU front avg {iou_front.avg:.3f}'.format(batch_time=self.batch_time_e,
                                                                            loss=losses,
                                                                            jac=jaccs,
                                                                            wces=wces,
                                                                            acc=acc, iou=iou, iou_front=iou_front))
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                save_to_log(self.log, 'log.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                self.info["valid_classes/" + class_func(i)] = jacc

            for i, jacc in enumerate(class_jaccard_front):
                print('Front IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                save_to_log(self.log, 'log.txt', 'Front IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=class_func(i), jacc=jacc))
                self.info["valid_classes_front/" + class_func(i)] = jacc
        return acc.avg, iou.avg, losses.avg, rand_imgs, iou_front.avg
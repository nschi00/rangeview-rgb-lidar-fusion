import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision.transforms as TF
import copy
import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings
from matplotlib import pyplot as plt
import pandas as pd


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_RGB = ['.png']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_rgb(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_RGB)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask,proj_labels

class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               only_lidar_front,
               rgb_resize,
               max_points=150000,   # max number of points present in dataset
               gt=True,
               transform=False,
               division=4):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.only_lidar_front = only_lidar_front
    self.transform = transform
    self.sensor_img_W = round((self.sensor_img_W // division) / 64) * 64
    self.division = division
    self.img_flip = TF.RandomHorizontalFlip(p=1.0)
    if rgb_resize:
      self.img_transform = TF.Compose([TF.ToTensor(), TF.Resize((self.sensor_img_H, self.sensor_img_W))])
    else:
      self.img_transform = TF.Compose([TF.ToTensor()])

    
    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []
    self.rgb_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")
      rgb_path = os.path.join(self.root, seq, "image_2")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]
      rgb_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(rgb_path)) for f in fn if is_rgb(f)]

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)
      self.rgb_files.extend(rgb_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()
    self.rgb_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    rgb_data = self.img_transform(Image.open(self.rgb_files[index])) # 3 x H x W

    if self.gt:
      label_file = self.label_files[index]
      # index_next = index + 1
      # next_file = self.scan_files[index_next]
      # split_scan = scan_file.split("/")
      # next_scan = next_file.split("/")
      # # * Test whether the next scan is from the same sequence
      # if split_scan[-3] != next_scan[-3]: 
      #   index_next = index - 1
      # del split_scan
      # del next_scan
      # next_file = self.scan_files[index_next]
      #label_file_next = self.label_files[index_next]

    # open a semantic laserscan
    DA = False
    flip_sign = False
    rot = False
    drop_points = False
    # if self.transform:
    #     if random.random() > 0.5:
    #         # if random.random() > 0.5:
    #         #     DA = True
    #         if random.random() > 0.5:
    #             flip_sign = True
    #             rgb_data = self.img_flip(rgb_data)
            # if random.random() > 0.5:
            #     rot = False
            # drop_points = random.uniform(0, 0.5)

    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down,
                          DA=DA,
                          flip_sign=flip_sign,
                          rot=rot,
                          drop_points=drop_points)
      
      # scan_next = copy.deepcopy(scan)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down,
                       DA=DA,
                       flip_sign=flip_sign,
                       rot=rot,
                       drop_points=drop_points)

    # open and obtain scan
    division_angles = self.get_division_angles(self.division)
    scan.open_scan(scan_file, self.only_lidar_front, division_angles)
    if self.gt:
      scan.open_label(label_file)
      # projected_data = self.prepare_output(scan, scan_file)
      # map unused classes to used classes (also for projection)
      
      # scan_next.open_scan(next_file, self.only_lidar_front, division_angles)
      # scan_next.open_label(label_file_next)
      # projected_data_next = self.prepare_output(scan_next, next_file)
      
      # projected_data = self.RangePaste(projected_data, projected_data_next)
      # projected_data_2 = self.RangeUnion(projected_data, projected_data_next)
      # self.visualize([projected_data_next[2], projected_data[2], rgb_data])
      return [scan.points, scan.remissions, scan.sem_label, scan.inst_label], rgb_data
      
    projected_data = self.prepare_output(scan, scan_file)
    # return
    return projected_data, rgb_data

  def RangeUnion(self, scan, scan_next, k_union=0.5):
    proj, proj_mask, proj_labels = copy.deepcopy(scan[0:3])
    proj_next, proj_mask_next, proj_labels_next = scan_next[0:3]
    void = proj_mask <= 0
    mask_temp = torch.rand(void.shape) <= k_union
    # * Only fill 50% of the void points
    void = void.logical_and(mask_temp)
    proj[:, void], proj_labels[void] = proj_next[:, void], proj_labels_next[void]
    proj_mask[void] = proj_mask_next[void]
    scan[0:3] = [proj, proj_mask, proj_labels]
    return scan

  def RangePaste(self, scan, scan_next, tail_classes=None):
    proj, proj_mask, proj_labels = copy.deepcopy(scan[0:3])
    proj_next, proj_mask_next, proj_labels_next = scan_next[0:3]
    if tail_classes is None:
      tail_classes = [ 2,  3,  4,  5,  6,  7,  8, 12, 16, 18, 19]
    for tail_class in tail_classes:
      pix = proj_labels_next == tail_class
      proj[:, pix] = proj_next[:, pix]
      proj_mask[pix] = proj_mask_next[pix]
      proj_labels[pix] = proj_labels_next[pix]
    
    return scan
  
  def RangeShift(scan):
    proj, proj_mask, proj_labels = copy.deepcopy(scan[0:3])
    _, h, w = proj_labels.shape
    p = torch.randint(int(0.25*w), int(0.75*w))
    proj = torch.cat(proj[:, p:, :], proj[:, :p, :], dim = 1)
    proj_labels = torch.cat(proj_labels[p:, :], proj_labels[:p, :], dim = 1)
    proj_mask = torch.cat(proj_mask[p:, :], proj_mask[:p, :], dim = 1)
    scan[0:3] = [proj, proj_mask, proj_labels]
    return scan
  
  def RangeMix(scan, scan_next, mix_strategies):
    proj, proj_mask, proj_labels = copy.deepcopy(scan[0:3])
    proj_next, proj_mask_next, proj_labels_next = scan_next[0:3]
    _, h, w = proj_labels.shape
    phi, theta = mix_strategies
    mix_h, mix_w = int(h / phi), int(w / theta)
    for i in range(1, mix_h):
      for j in range(1, mix_w):
        proj[:, i-1:i, j-1:j] = proj_next[:, i-1:i, j-1:j]
        proj_labels[:, i-1:i, j-1:j] = proj_labels_next[:, i-1:i, j-1:j]
        proj_mask[:, i-1:i, j-1:j] = proj_mask_next[:, i-1:i, j-1:j]
    scan[0:3] = [proj, proj_mask, proj_labels]
    return scan
  
  def visualize(self, img_list: list):
    plot_length = len(img_list)
    for i, img in enumerate(img_list):
      if img.shape[0] == 3:
        img_list[i] = img.permute(1, 2, 0)
      else:
        img_list[i] = self.map(img, self.learning_map_inv)
        img_list[i] = self.map(img_list[i], self.color_map)
        
    fig, axs = plt.subplots(plot_length, 1)
    for i in range(plot_length):
      axs[i].imshow(img_list[i])
      axs[i].axis('off')
      
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1)

    # Show the plot
    plt.show()

  def get_division_angles(self, division):
        # Calculate the angle per division
        angle_per_division = 360.0 / division

        # Calculate the start angle for the first division
        first_start_angle = -angle_per_division / 2.0
        first_end_angle = angle_per_division / 2.0

        # Create a list to store the start and end angles for each division
        division_angles = [(first_start_angle, first_end_angle)]

        # Calculate the start and end angles for the remaining divisions
        for i in range(1, division):
            start_angle = division_angles[i-1][1]
            if start_angle < 0.0:
                end_angle = start_angle - angle_per_division
            else:
                end_angle = start_angle + angle_per_division
            if end_angle > 180.0:
                end_angle -= 360.0
            division_angles.append((start_angle, end_angle))

        return division_angles

  def prepare_output(self, scan, scan_file):
    scan.sem_label = self.map(scan.sem_label, self.learning_map)
    scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range)
    proj_xyz = torch.from_numpy(scan.proj_xyz)
    proj_remission = torch.from_numpy(scan.proj_remission)

#     proj_normal = torch.from_numpy(scan.normal_image)

    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label)
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []

    del scan
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    
    del proj_range, proj_xyz, proj_remission


    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]

    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    projected_data = [proj, 
                      proj_mask, 
                      proj_labels, 
                      path_seq, 
                      path_name
]
    
    return projected_data
    
  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               rgb_resize,
               gt=True,           # get gt?
               shuffle_train=True,
               subset_ratio=1.0,
               only_lidar_front=False,
               division=1):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       transform=True,
                                       gt=self.gt,
                                       only_lidar_front=only_lidar_front,
                                       rgb_resize=rgb_resize,
                                       division=division)
    
    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt,
                                       only_lidar_front=only_lidar_front,
                                       rgb_resize=rgb_resize,
                                       division=division)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(1024)

    if subset_ratio < 1:
      samples_step = np.rint(1 / subset_ratio).astype('int')
      self.train_dataset = torch.utils.data.Subset(self.train_dataset, np.arange(0, len(self.train_dataset), samples_step))


    def collate_fn(batch):
      batch_zip = list(zip(*batch))
      rgb = list(zip(*batch_zip[1]))
      lidar = list(zip(*batch_zip[0]))
      pcd, remission, sem_label, ins_label = lidar
      n_pcd, n_remission, n_sem_label, n_ins_label = [], [], [], []

      min_length = min(len(arr) for arr in pcd)
      for i in range(len(pcd)):
        current_length = len(pcd[i])
        index = np.random.choice(range(current_length), min_length, replace=False)
        # new_point_cloud = [arr[random_indices] for arr in point_cloud]
        # new_label = [arr[random_indices] for arr in label]
        #pcd[i] = pcd[i][index]
        
        n_pcd.append(pcd[i][index,:])
        n_remission.append(remission[i][index])
        n_sem_label.append(sem_label[i][index])
        n_ins_label.append(ins_label[i][index])
        
      del pcd, remission, sem_label, ins_label
      
      n_pcd = np.stack(n_pcd, axis=0)
      n_remission = np.stack(n_remission, axis=0)
      n_sem_label = np.stack(n_sem_label, axis=0)
      n_ins_label = np.stack(n_ins_label, axis=0)
      
      n_pcd = torch.from_numpy(n_pcd)
      n_remission = torch.from_numpy(n_remission)
      n_sem_label = torch.from_numpy(n_sem_label)
      n_ins_label = torch.from_numpy(n_ins_label)

      return [n_pcd, n_remission, n_sem_label, n_ins_label], rgb
    
    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   worker_init_fn=seed_worker,
                                                   generator=g,
                                                   drop_last=True,
                                                   collate_fn=collate_fn)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)
    
      
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    drop_last=True)
#       assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses
  
  def get_reso(self):
    try:
      H = self.train_dataset.dataset.sensor_img_H
      W = self.train_dataset.dataset.sensor_img_W
    except:
      H = self.train_dataset.sensor_img_H
      W = self.train_dataset.sensor_img_W
    
    return (H,W)

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)
  
  
  
  
  
  
  
  
  
  
  
  
  ######### TEMP FOR VISUALIZATION #########
  # proj_prev, proj_mask_prev, proj_labels_prev = self.prepare_output(scan)
    # proj_next, proj_mask_next, proj_labels_next = self.prepare_output(scan_next)
    # projected_data = self.RangeUnion(self.prepare_output(scan), self.prepare_output(scan_next))
    # del scan_next, scan
    # label = SemanticKitti.map(proj_labels, self.learning_map_inv)
    # label = SemanticKitti.map(label, self.color_map)
    
    # label_pre = SemanticKitti.map(proj_labels_prev, self.learning_map_inv)
    # label_pre = SemanticKitti.map(label_pre, self.color_map)
    
    # label_next = SemanticKitti.map(proj_labels_next, self.learning_map_inv)
    # label_next = SemanticKitti.map(label_next, self.color_map)
    
    # fig, axs = plt.subplots(3, 1)

    # Display the top image
    # axs[0].imshow(label_pre)
    # axs[0].axis('off')
    
    # axs[2].imshow(label)
    # axs[2].axis('off')

    # # Display the bottom image
    # axs[1].imshow(label_next)
    # axs[1].axis('off')

    # # Adjust spacing between subplots
    # plt.subplots_adjust(hspace=0.1)

    # # Show the plot
    # plt.show()

  
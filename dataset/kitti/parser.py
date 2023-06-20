import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
import torchvision.transforms as TF
from collections import defaultdict
import torch
import random
from PIL import Image
import numpy as np


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
               max_points=150000,   # max number of points present in dataset
               gt=True,
               transform=False,
               old_aug=True):            # send ground truth?
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
    self.transform = transform
    
    if self.transform:
      if old_aug:
        self.aug_prob = {"scaling": 0.0,
                         "rotation": 0.5,
                         "jittering": 0.5,
                         "flipping": 0.5,
                         "point_dropping": [0.5, 0.5],
                         "type": "old"}
      else:
        self.aug_prob = {"scaling": 1.0,
                        "rotation": 1.0,
                        "jittering": 1.0,
                        "flipping": 1.0,
                        "point_dropping": [0.9, 0.1],
                        "type": "new"}
        
      self.img_aug_prob = {"scaling": 0.0,
                           "C_jittering": 0.5,
                           "H_flip": 0.5,
                           "V_flip": 0.5,}
                           
      ColorJitter = TF.RandomApply(torch.nn.ModuleList([TF.ColorJitter(0.2, 0.2, 0.2, 0.2)]), p=self.img_aug_prob["C_jittering"])
      self.img_transform = TF.Compose([TF.ToTensor(),
                                      ColorJitter,
                                      TF.RandomHorizontalFlip(p=self.img_aug_prob["H_flip"]),
                                      TF.RandomVerticalFlip(p=self.img_aug_prob["V_flip"]),
                                     TF.Resize((256, 768)),
                                     TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
      self.aug_prob = defaultdict(lambda: -1.0)
      self.aug_prob["point_dropping"] = [-1.0, -1.0]
      self.img_transform = TF.Compose([TF.ToTensor(),
                                     TF.Resize((256, 768)),
                                     TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
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
    rgb_data = Image.open(self.rgb_files[index])

    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down,
                          aug_prob=self.aug_prob)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down,
                       aug_prob=self.aug_prob)

    # open and obtain scan
    scan.open_scan(scan_file, rgb_data)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    unproj_n_points = scan.points.shape[0]

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_idx = torch.from_numpy(scan.proj_idx).clone()
    query_mask = torch.from_numpy(scan.query_mask).clone()
    # query_idx = np.full((self.sensor_img_H * self.sensor_img_W), -1, dtype=np.int16)
    # query_idx[:scan.point_idx_rv_camera_fov.shape[0]] = scan.point_idx_rv_camera_fov
    # query_idx = torch.from_numpy(query_idx).clone()

#     proj_normal = torch.from_numpy(scan.normal_image).clone()

    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])

#     proj = torch.cat([proj_range.unsqueeze(0).clone(),
#                       proj_xyz.clone().permute(2, 0, 1),
#                       proj_remission.unsqueeze(0).clone(),
#                       proj_normal.unsqueeze(0).clone()])

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
                      query_mask, 
                      path_seq, 
                      path_name, 
                      proj_idx]
    
    rgb_data = self.img_transform(rgb_data)
    # return
    return projected_data, rgb_data

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
               gt=True,           # get gt?
               shuffle_train=True,
               subset_ratio=1.0,
               old_aug = True):
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
                                       old_aug=old_aug)
    
    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(1024)

    if subset_ratio < 1:
      samples_step = np.rint(1 / subset_ratio).astype('int')
      self.train_dataset = torch.utils.data.Subset(self.train_dataset, np.arange(0, len(self.train_dataset), samples_step))

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   worker_init_fn=seed_worker,
                                                   generator=g,
                                                   drop_last=True)
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
  
  def get_resolution(self):
    try:
      H = self.train_dataset.sensor_img_H
      W = self.train_dataset.sensor_img_W
    except:
      H = self.train_dataset.dataset.sensor_img_H
      W = self.train_dataset.dataset.sensor_img_W
    return H, W
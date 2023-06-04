#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,DA=False,flip_sign=False,rot=False,drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename, only_lidar_front, division_angle=None):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        self.filename = filename

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz

        remissions = scan[:, 3]  # get remission

        self.mask_front = None

        # if only_lidar_front:
        #     self.mask_front = self.points_basic_filter(points, [-40, 40], [-14, 30])
        #     points = points[self.mask_front]
        #     remissions = remissions[self.mask_front]
        self.division_angle = division_angle
        self.points_map_lidar2rgb = points
        fov_mask = self.points_basic_filter(points, self.division_angle[0], [-90, 90])
        self.fov_mask = np.where(fov_mask == 0)[0]
        self.points = points
        self.remissions = remissions
    # self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute

        self.aug_prob = {"scaling": 1.0,
                         "rotation": 1.0,
                         "jittering": 1.0,
                         "flipping": 1.0,
                         "point_dropping": 0.9}
        rand_num = random.random()

        if rand_num < self.aug_prob["point_dropping"]:
            self.points_to_drop = self.RandomDropping(points)
            self.points_to_drop = np.unique(np.concatenate((self.points_to_drop, self.fov_mask)))
        else:
            self.points_to_drop = self.fov_mask

        if len(self.points_to_drop) > 0:
            self.points = np.delete(points, self.points_to_drop, axis=0)
            remissions = np.delete(remissions, self.points_to_drop)
        else:
            self.points = points
        if rand_num < self.aug_prob["scaling"]:
            self.points = self.RandomScaling(self.points)
        if rand_num < self.aug_prob["rotation"]:
            self.points = self.GlobalRotation(self.points)
        if rand_num < self.aug_prob["jittering"]:
            self.points = self.RandomJittering(self.points)
        if rand_num < self.aug_prob["flipping"]:
            self.points = self.RandomFlipping(self.points)
        # if self.flip_sign:
        #     self.points[:, 1] = -self.points[:, 1]
        # if self.DA:
        #     jitter_x = random.uniform(-5, 5)
        #     jitter_y = random.uniform(-3, 3)
        #     jitter_z = random.uniform(-1, 0)
        #     self.points[:, 0] += jitter_x
        #     self.points[:, 1] += jitter_y
        #     self.points[:, 2] += jitter_z
        # if self.rot:
        #     euler_angle = np.random.normal(0, 90, 1)[0]  # 40
        #     r = np.array(R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix())
        #     r_t = r.transpose()
        #     self.points = self.points.dot(r_t)
        #     self.points = np.squeeze(self.points)
        if remissions is not None:
            self.remissions = remissions  # get remission
            #if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()


    def RandomScaling(self, scan, r_s=0.05):
        scale = np.random.uniform(1, 1+r_s)
        if np.random.random() < 0.5:
            scale = 1 / scale
            scan[:, :2] *= scale

        return scan

    def GlobalRotation(self, scan):
        rotate_rad = np.deg2rad(np.random.random() * 360)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        scan[:, :2] = np.dot(scan[:, :2], j)
        return scan

    def RandomJittering(self, scan, r_j=0.3):
        jitter = np.clip(np.random.normal(0, r_j, 3), -r_j, r_j)
        scan += jitter
        return scan

    def RandomFlipping(self, scan):
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            scan[:, 0] = -scan[:, 0]
        elif flip_type == 2:
            scan[:, 1] = -scan[:, 1]
        elif flip_type == 3:
            scan[:, :2] = -scan[:, :2]
        return scan

    def RandomDropping(self, scan, r_d=0.1):
        drop = int(len(scan) * r_d)
        drop = np.random.randint(low=0, high=drop)
        to_drop = np.random.randint(low=0, high=len(scan)-1, size=drop)
        to_drop = np.unique(to_drop)
        return to_drop

    def do_fd_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
          Function takes no arguments because it can be also called externally
          if the value of the constructor was not set (in case you change your
          mind about wanting the projection)
      """
        # laser parameters
        depth = np.linalg.norm(self.points, 2, axis=1)
        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1

        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        proj_y = np.cumsum(proj_y)
        proj_x = proj_x * self.proj_W - 0.001

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order
        # stope a copy in original order

        self.unproj_range = np.copy(depth)  # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        # proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        #self.fov_vert = [-14, 5]
        #fov_vert = np.asarray(self.fov_vert) / 180.0 * np.pi
        proj_x = abs((yaw) - (yaw.min())) / abs((yaw.min()) - (yaw.max())) # using yaw due to varying values for fov_hor
        #proj_y = 1.0 - (pitch + abs(fov_vert[0])) / abs(fov_vert[0]-fov_vert[1])  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

    def fill_spherical(self, range_image):
        # fill in spherical image for calculating normal vector
        height, width = np.shape(range_image)[:2]
        value_mask = np.asarray(1.0 - np.squeeze(range_image) > 0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)

        with_value = np.squeeze(range_image) > 0.1

        depth_list = np.squeeze(range_image)[with_value]

        label_list = np.reshape(lbl, [1, height * width])
        depth_list_all = depth_list[label_list - 1]

        depth_map = np.reshape(depth_list_all, (height, width))

        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)
        depth_map = range_image * with_value + depth_map * (1 - with_value)
        return depth_map

    def calculate_normal(self, range_image):

        one_matrix = np.ones((self.proj_H, self.proj_W))
        # img_gaussian =cv2.GaussianBlur(range_image,(3,3),0)
        img_gaussian = range_image
        # prewitt
        kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        self.partial_r_theta = img_prewitty / (np.pi * 2.0 / self.proj_W) / 6
        self.partial_r_phi = img_prewittx / (((self.fov_up - self.fov_down) / 180.0 * np.pi) / self.proj_H) / 6

        partial_vector = [1.0 * one_matrix, self.partial_r_theta / (range_image * np.cos(self.phi_channel)),
                          self.partial_r_phi / range_image]
        partial_vector = np.asarray(partial_vector)
        partial_vector = np.transpose(partial_vector, (1, 2, 0))
        partial_vector = np.reshape(partial_vector, [self.proj_H, self.proj_W, 3, 1])
        normal_vector = np.matmul(self.R_theta_phi, partial_vector)
        normal_vector = np.squeeze(normal_vector)
        normal_vector = normal_vector / np.reshape(np.max(np.abs(normal_vector), axis=2),
                                                   (self.proj_H, self.proj_W, 1))
        normal_vector_camera = np.zeros((self.proj_H, self.proj_W, 3))
        normal_vector_camera[:, :, 0] = normal_vector[:, :, 1]
        normal_vector_camera[:, :, 1] = -normal_vector[:, :, 2]
        normal_vector_camera[:, :, 2] = normal_vector[:, :, 0]
        return normal_vector_camera

    def project_lidar_into_image(self, rgb):
        self.filename = self.filename.rsplit('/', 2)[0] + "/calib.txt"

        # Open the file for reading
        with open(self.filename, 'r') as file:
            # Read the contents of the file
            contents = file.read()

        # Split the contents by lines and process each line
        lines = contents.split('\n')
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('P2:'):
                numbers_str = line.split(':')[1].strip()  # Extract the numbers after the colon and remove whitespace
                numbers = list(map(float, numbers_str.split()))  # Convert the numbers to a list of floats

                # Reshape the list into a 3x4 matrix
                P2 = np.array([numbers[i:i+4] for i in range(0, len(numbers), 4)])
            elif line.startswith('Tr:'):
                numbers_str = line.split(':')[1].strip()  # Extract the numbers after the colon and remove whitespace
                numbers = list(map(float, numbers_str.split()))  # Convert the numbers to a list of floats

                # Reshape the list into a 3x4 matrix
                Tr = np.array([numbers[i:i+4] for i in range(0, len(numbers), 4)])

        # Transform LiDAR to left camera coordinates and projection to pixel space as described in KITTI Odometry Readme
        hom_points = np.ones((np.shape(self.points_map_lidar2rgb)[0], 4))
        hom_points[:, 0:3] = self.points_map_lidar2rgb
        trans_points = (Tr @ hom_points.T).T
        hom_points[:, 0:3] = trans_points
        proj_points_im = (P2 @ hom_points.T).T
        proj_points_im[:, 0] /= proj_points_im[:, 2]
        proj_points_im[:, 1] /= proj_points_im[:, 2]
        proj_points_im = proj_points_im[:, 0:2]

        rgb_image = Image.fromarray(np.transpose((rgb.numpy()*255).astype('uint8'), (1, 2, 0)))
        image_with_points = rgb_image.copy()

        draw = ImageDraw.Draw(image_with_points)

        # Draw the points on the image
        for point in proj_points_im:
            draw.point(point, fill="red")

        # Save the modified image with the points
        image_with_points.save("image_with_points.jpg")

    ### Code from https://github.com/Jiang-Muyun/Open3D-Semantic-KITTI-Vis.git
    def hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """
        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def points_basic_filter(self, points, h_fov, v_fov):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            return a bool array
        """
        assert points.shape[1] == 3, points.shape # [N,3]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))

        # extract in-range fov points
        h_points = self.hv_in_range(x, y, h_fov, fov_type='h')
        v_points = self.hv_in_range(d, z, v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)

        return combined

    def extract_points(self):
        # filter in range points based on fov, x,y,z range setting
        combined = self.points_basic_filter(self.points)
        points = self.points[combined]
        label = self.sem_label[combined]


class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300,DA=False,flip_sign=False,rot=False,drop_points=False):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down,DA=DA,flip_sign=flip_sign,rot=rot,drop_points=drop_points)
        self.reset()

        # make semantic colors
        if sem_color_dict:
            # if I have a dict, make it
            max_sem_key = 0
            for key, data in sem_color_dict.items():
                if key + 1 > max_sem_key:
                    max_sem_key = key + 1
            self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
            for key, value in sem_color_dict.items():
                self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
        else:
            # otherwise make random
            max_sem_key = max_classes
            self.sem_color_lut = np.random.uniform(low=0.0,
                                                   high=1.0,
                                                   size=(max_sem_key, 3))
            # force zero to a gray-ish color
            self.sem_color_lut[0] = np.full((3), 0.1)

        # make instance colors
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

    def reset(self):
        """ Reset scan members. """
        super(SemLaserScan, self).reset()

        # semantic labels
        self.sem_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # instance labels
        self.inst_label = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: label
        self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                       dtype=np.int32)  # [H,W]  label
        self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                       dtype=np.float)  # [H,W,3] color

        # projection color with instance labels
        self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                        dtype=np.int32)  # [H,W]  label
        self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                        dtype=np.float)  # [H,W,3] color

    def open_label(self, filename):
        """ Open raw scan and fill in attributes
        """
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        label = np.fromfile(filename, dtype=np.int32)
        label = label.reshape((-1))

        # label = np.delete(label,self.points_to_drop) if len(self.points_to_drop) > 0 else label
        # set it
        self.set_label(label)

    def set_label(self, label):
        """ Set points for label not from file but from np
        """
        # check label makes sense
        if not isinstance(label, np.ndarray):
            raise TypeError("Label should be numpy array")

        # only fill in attribute if the right size
        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # sanity check
        assert ((self.sem_label + (self.inst_label << 16) == label).all())

        #if self.project:
        # self.do_label_projection()

    def colorize(self):
        """ Colorize pointcloud with the color of each semantic label
        """
        self.sem_label_color = self.sem_color_lut[self.sem_label]
        self.sem_label_color = self.sem_label_color.reshape((-1, 3))

        self.inst_label_color = self.inst_color_lut[self.inst_label]
        self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    def do_label_projection(self):
        # only map colors to labels that exist
        mask = self.proj_idx >= 0

        # semantics
        self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
        self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

        # instances
        self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
        self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]


class Preprocess(nn.Module):
    def __init__(self,
                 aug_params=None,
                 sensor=None,
                 learning_map=None,
                 learning_map_inv=None,
                 color_map=None,
                 fov_up=3.0,
                 fov_down=-25.0,
                 division=1,
                 old_aug=True,
                 seed=1024) -> None:
        super(Preprocess, self).__init__()

        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.sensor = sensor
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.proj_W = sensor["img_prop"]["width"]
        self.proj_H = sensor["img_prop"]["height"]
        self.proj_W = round((self.proj_W // division) / 64) * 64
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float,
                                         device="cuda")
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float,
                                        device="cuda")
        self.color_map = color_map
        super(Preprocess, self).__init__()
        self.division = division
        self.old = old_aug
        if self.old:
            self.aug_prob = {"scaling": 0.5,
                            "rotation": 0.5,
                            "jittering": 0.5,
                            "flipping": 0.5,
                            "point_dropping": 0.5}

        else:
            self.aug_prob = {"scaling": 1.0,
                            "rotation": 1.0,
                            "jittering": 1.0,
                            "flipping": 1.0,
                            "point_dropping": 0.9}

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        #torch.set_default_device('cuda')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def augmentation(self, pcd, remissions, sem_label, inst_label):
        if random.random() < self.aug_prob["scaling"]:
            pcd = self.RandomScaling(pcd)
        if random.random() < self.aug_prob["rotation"]:
            pcd = self.GlobalRotation(pcd)
        if random.random() < self.aug_prob["jittering"]:
            pcd = self.RandomJittering(pcd)
        if random.random() < self.aug_prob["flipping"]:
            pcd = self.RandomFlipping(pcd)
        self.mask = self.fov_cal(pcd)
        if random.random() < self.aug_prob["point_dropping"]:
            mask_drop = self.RandomDropping(pcd,
                                            r_d=0.1)
            self.mask = self.mask & mask_drop
        return pcd, remissions, sem_label, inst_label

    def old_augmentation(self, pcd, remissions, sem_label, inst_label):
        self.mask = self.fov_cal(pcd)
        if random.random() < self.aug_prob["point_dropping"]:
            mask_drop = self.RandomDropping(pcd,
                                        r_d=random.uniform(0, 0.5))
        self.mask = self.mask & mask_drop
        if random.random() < self.aug_prob["rotation"]:
            pcd = self.old_rot(pcd)
        if random.random() < self.aug_prob["jittering"]:
            pcd = self.old_DA(pcd)
        if random.random() < self.aug_prob["flipping"]:
            pcd = self.old_flip(pcd)
       
        pcd = torch.where(self.mask.unsqueeze(-1).expand(-1, -1, 3), pcd, torch.nan)
        remissions = torch.where(self.mask, remissions, torch.nan)
        sem_label = torch.where(self.mask, sem_label, torch.nan)
        inst_label = torch.where(self.mask, inst_label, torch.nan)
        return pcd, remissions, sem_label, inst_label

    def projection_points(self, pcd, remissions, sem_label):
        bs = pcd.shape[0]
        # projected range image - [B,H,W] range (-1 is no data)
        proj_range = torch.full((bs, self.proj_H, self.proj_W), -1,
                                  dtype=torch.float32, device="cuda")

        # projected point cloud xyz - [B,3,H,W] xyz coord (-1 is no data)
        proj_xyz = torch.full((bs, 3, self.proj_H, self.proj_W), -1,
                                dtype=torch.float32, device="cuda")

        # projected remission - [B,H,W] intensity (-1 is no data)
        proj_remission = torch.full((bs, self.proj_H, self.proj_W), -1,
                                      dtype=torch.float32, device="cuda")

        proj_idx = torch.full((bs, self.proj_H, self.proj_W), -1,
                                dtype=torch.int64, device="cuda")

        # mask containing for each pixel, if it contains a point or not
        proj_mask = torch.zeros((bs, self.proj_H, self.proj_W),
                                  dtype=torch.int32, device="cuda")  # [H,W] mask

        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov_vert = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = torch.linalg.norm(pcd, 2, dim=2)

        # get scan components
        scan_x = pcd[:, :, 0]
        scan_y = pcd[:, :, 1]
        scan_z = pcd[:, :, 2]

        # get angles of all points
        yaw = -torch.arctan2(scan_y, scan_x)
        pitch = torch.arcsin(scan_z / depth)

        # get projections in image coords
        # proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov_vert  # in [0.0, 1.0]
        #self.fov_vert = [-14, 5]
        #fov_vert = np.asarray(self.fov_vert) / 180.0 * np.pi

        yaw_max = torch.max(torch.nan_to_num(yaw, nan=-float('inf')), dim=1)[0]
        yaw_min = torch.min(torch.nan_to_num(yaw, nan=float('inf')), dim=1)[0]
        
        proj_x = abs((yaw) - yaw_min.unsqueeze(1)) / abs(yaw_min - yaw_max).unsqueeze(1) # using yaw due to varying values for fov_hor
        #proj_y = 1.0 - (pitch + abs(fov_vert[0])) / abs(fov_vert[0]-fov_vert[1])  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = torch.floor(proj_x)
        proj_x[proj_x >= self.proj_W - 1] = self.proj_W - 1
        proj_x[proj_x<0] = 0  # in [0,W-1]
        #proj_x = proj_x.long()
        
        proj_y = torch.floor(proj_y)
        proj_y[proj_y >= self.proj_H - 1] = self.proj_H - 1
        proj_y[proj_y<0] = 0  # in [0,H-1]
        #proj_y = proj_y.long()

        # order in decreasing depth


        proj_sem_label = torch.zeros((bs, self.proj_H, self.proj_W), device="cuda", dtype=torch.int64)
        # assign to images
        #points = []
        # start_time = time.time()
        for i in range(bs):
            non_nan_indices = torch.nonzero(~torch.isnan(depth[i])).squeeze()
            depth_without_nan = depth[i, non_nan_indices]
            order = torch.argsort(depth_without_nan, descending=True)
            depth_without_nan_sorted = depth_without_nan[order]
            non_nan_indices_sorted = non_nan_indices[order]
            
            remission = remissions[i, non_nan_indices_sorted]
            points_x = scan_x[i, non_nan_indices_sorted]
            points_y = scan_y[i, non_nan_indices_sorted]
            points_z = scan_z[i, non_nan_indices_sorted]
            points = torch.stack((points_x, points_y, points_z), dim=0)
            proj_y_i = proj_y[i, non_nan_indices_sorted].long()
            proj_x_i = proj_x[i, non_nan_indices_sorted].long()
            
            proj_range[i, proj_y_i, proj_x_i] = depth_without_nan_sorted
            proj_xyz[i,:, proj_y_i, proj_x_i] = points
            proj_remission[i, proj_y_i, proj_x_i] = remission
            proj_idx[i, proj_y_i, proj_x_i] = non_nan_indices_sorted
            
            # ! I dont't understand why this works, but it does
            proj_mask = (proj_idx[i] > 0).bool()
            proj_sem_label[i][proj_mask] = sem_label[i][proj_idx[i][proj_mask]].long()
            proj_sem_label[i] = self.map(proj_sem_label[i], self.learning_map)
            
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(execution_time)
        proj_mask = (proj_idx > 0).long()

        proj = torch.cat([proj_range.unsqueeze(1),
                      proj_xyz,
                      proj_remission.unsqueeze(1)],
                      dim=1)

        proj = (proj - self.sensor_img_means[None, :, None, None]
                ) / self.sensor_img_stds[None, :, None, None]

        proj = proj * proj_mask.unsqueeze(1).repeat_interleave(5, dim=1).float()
        proj_sem_label *= proj_mask
        return proj, proj_mask,  proj_sem_label

    def RandomScaling(self, pcd, r_s=0.05):
        scale = torch.rand(1, device="cuda") * r_s + 1
        if np.random.random() < 0.5:
            scale = 1 / scale
            pcd *= scale

        return pcd

    def GlobalRotation(self, pcd):
        rotate_rad = np.deg2rad(np.random.random() * 360)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        j = torch.from_numpy(j).float().to(self.device)
        pcd[:, :, :2] = torch.matmul(pcd[:, :, :2], j)
        return pcd

    def RandomJittering(self, pcd, r_j=0.3):
        B = pcd.shape[0]
        jitter = torch.clip(torch.randn((B, 1, 3), device="cuda") * 3, -r_j, r_j)
        pcd += jitter
        return pcd

    def RandomFlipping(self, pcd):
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            pcd[:, 0] = -pcd[:, 0]
        elif flip_type == 2:
            pcd[:, 1] = -pcd[:, 1]
        elif flip_type == 3:
            pcd[:, :2] = -pcd[:, :2]
        return pcd

    def RandomDropping(self, pcd, r_d=0.1):
        bs, L = pcd.shape[0:2]
        mask = torch.bernoulli(torch.ones(bs, L) * (1 - r_d)).bool()
        return mask.cuda()

    def fov_cal(self, pcd):
        # Calculate the angle per division
        angle_per_division = 360.0 / self.division

        # Calculate the start angle for the first division
        first_start_angle = -angle_per_division / 2.0
        first_end_angle = angle_per_division / 2.0

        # Create a list to store the start and end angles for each division
        division_angles = [(first_start_angle, first_end_angle)]

        # Calculate the start and end angles for the remaining divisions
        for i in range(1, self.division):
            start_angle = division_angles[i-1][1]
            if start_angle < 0.0:
                end_angle = start_angle - angle_per_division
            else:
                end_angle = start_angle + angle_per_division
            if end_angle > 180.0:
                end_angle -= 360.0
            division_angles.append((start_angle, end_angle))

        index = 0
        # index = torch.randint(0, len(division_angles), (1,)).item()

        fov_mask = self.points_basic_filter(pcd, division_angles[index], [-90, 90])

        return fov_mask

    def points_basic_filter(self, points, h_fov, v_fov):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            return a bool array
        """
        assert points.shape[2] == 3, points.shape # [N,3]
        x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        # temp = torch.sqrt(x[0][0] ** 2 + y[0][0] ** 2 + z[0][0] ** 2)
        d = torch.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))

        # extract in-range fov points
        h_points = self.hv_in_range(x, y, h_fov, fov_type='h')
        v_points = self.hv_in_range(d, z, v_fov, fov_type='v')
        combined = torch.logical_and(h_points, v_points)

        return combined

    ### Code from https://github.com/Jiang-Muyun/Open3D-Semantic-KITTI-Vis.git
    def hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """
        #a = torch.atan2(n, m) > (-fov[1] * np.pi / 180)
        if fov_type == 'h':
            return torch.logical_and(torch.atan2(n, m) > (-fov[1] * np.pi / 180), \
                                    torch.atan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return torch.logical_and(torch.atan2(n, m) < (fov[1] * np.pi / 180), \
                                    torch.atan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def old_flip(self, pcd):
        pcd[:, :, 1] = -pcd[:, :, 1]
        return pcd

    def old_DA(self, pcd):
        B = pcd.shape[0]
        jitter_x = torch.randn((B,1), device="cuda") * 5
        jitter_y = torch.randn((B,1), device="cuda") * 3
        jitter_z = torch.rand((B,1), device="cuda") * -1
        pcd[:, :, 0] += jitter_x
        pcd[:, :, 1] += jitter_y
        pcd[:, :, 2] += jitter_z
        return pcd

    def old_rot(self, pcd):
        B = pcd.shape[0]
        euler_angles = np.random.normal(0, 90, B)
        r = [np.array(R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix())
             for euler_angle in euler_angles]
        r_t = torch.from_numpy(np.stack(r)).transpose(1, 2).float().squeeze().to(self.device)  
        #r_t = torch.from_numpy(r.transpose()).float().to(self.device).squeeze()
        pcd = torch.matmul(pcd, r_t)
        return pcd

    def visualize(self, proj, proj_mask, proj_labels):
        image = self.make_log_img(proj.cpu().numpy(), proj_mask.cpu().numpy(), proj_labels.cpu().numpy(), self.to_color)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

    def forward(self, pcd, remission, sem_label, inst_label, train=True):
        if train:
            if self.old:
                pcd, remission, sem_label, inst_label = self.old_augmentation(
                    pcd, remission, sem_label, inst_label)
            else:
                pcd, remission, sem_label, inst_label = self.augmentation(
                    pcd, remission, sem_label, inst_label)
        proj, proj_mask, proj_labels = self.projection_points(pcd, remission, sem_label)

        # self.visualize(proj[0][0], proj_mask[0], proj_labels[0])
        # self.visualize(proj[1][0], proj_mask[1], proj_labels[1])
        # self.visualize(proj[2][0], proj_mask[2], proj_labels[2])
        # self.visualize(proj[3][0], proj_mask[3], proj_labels[3])
        # self.visualize(proj[4][0], proj_mask[4], proj_labels[4])
        return proj, proj_mask, proj_labels

    def make_log_img(self, depth, mask, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, self.get_mpl_colormap('viridis')) * mask[..., None]
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    def to_color(self, label):
        # put label in original values
        label = self.map_old(label, self.learning_map_inv)
        # put label in color
        return self.map_old(label, self.color_map)

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
            lut = torch.zeros((maxkey + 100, nel), dtype=torch.int32, device='cuda')
        else:
            lut = torch.zeros((maxkey + 100), dtype=torch.int32, device='cuda')
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    @staticmethod
    def map_old(label, mapdict):
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

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

if __name__ == "__main__":
    a = [torch.ones((1,20))*0.9] * 10
    a = torch.bernoulli(input=a)
    none_zero = torch.sum(a)
    prepo = Preprocess(3)
    a = prepo(2)

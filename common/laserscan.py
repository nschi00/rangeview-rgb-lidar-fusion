#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import vispy
from vispy.scene import visuals, SceneCanvas

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, aug_prob=None):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.aug_prob = aug_prob
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

    def open_scan(self, filename, rgb_data):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        self.rgb_data = rgb_data
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

        
        # self.point_idx_camera_fov = np.argwhere(mask_camera_fov == True).squeeze(1)

        if random.random() < self.aug_prob["point_dropping"][0]:
            if self.aug_prob["type"] == "new":
                self.drop_points = random.uniform(0.0, self.aug_prob["point_dropping"][1])
            else:
                self.drop_points = self.aug_prob["point_dropping"][0]
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)
            remissions = np.delete(remissions,self.points_to_drop)
        else:
            self.drop_points = False
            
        fov_hor = [-90, 90]
        fov_vert = [-90, 90]
        mask_camera_fov = self.points_basic_filter(points, fov_hor, fov_vert)
        self.point_idx_camera_fov = self.get_lidar_points_in_image_plane(points, mask_camera_fov)

        self.set_points(points, remissions)

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
        self.aug_prob["flipped"] = False
        # put in attribute
        self.points = points  # get
        if random.random() < self.aug_prob["scaling"]:
            self.RandomScaling()
        if random.random() < self.aug_prob["rotation"]:
            self.GlobalRotation()
        if random.random() < self.aug_prob["jittering"]:
            self.RandomJittering()
        if random.random() < self.aug_prob["flipping"]:
            self.aug_prob["flipped"] = True
            self.RandomFlipping(mode=self.aug_prob["type"])

        if remissions is not None:
            self.remissions = remissions  # get remission
            #if self.DA:
            #    self.remissions = self.remissions[::-1].copy()
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def RandomScaling(self, r_s=0.05):
        scale = np.random.uniform(1, 1+r_s)
        if np.random.random(1) < 0.5:
            scale = 1 / scale
            self.points[:, :2] *= scale
    
    def GlobalRotation(self):
        euler_angle = np.random.normal(0, 90, 1)[0]  # 40
        r = np.array(R.from_euler('zyx', [[euler_angle, 0, 0]], degrees=True).as_matrix())
        r_t = r.transpose()
        self.points = self.points.dot(r_t)
        self.points = np.squeeze(self.points)
        
    def RandomJittering(self):
        r_j = 0.3
        jitter = np.clip(np.random.normal(0, r_j, 3), -r_j, r_j)
        self.points += jitter
            
    def RandomFlipping(self, mode):
        if mode == "old":
            flip_type = 2
        elif mode == "new":
            flip_type = np.random.choice([1,2,3], 1)[0]
        if flip_type == 1:
            self.points[:, 0] = -self.points[:, 0]
        elif flip_type == 2:
            self.points[:, 1] = -self.points[:, 1]
        elif flip_type == 3:
            self.points[:, :2] = -self.points[:, :2]
    
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
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

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
        self.query_mask = np.isin(self.proj_idx, self.point_idx_camera_fov)

        # self.point_idx_rv = self.proj_idx[self.proj_idx != -1]
        # self.point_idx_rv_camera_fov = np.intersect1d(self.point_idx_rv, self.point_idx_camera_fov)
        # test = self.proj_idx[self.query_mask]
        

        # ! Comment in if visualization is needed
        # self.start_visualization(indices, "All points")
        # self.start_visualization(self.point_idx_rv, "Only Range View")
        # self.start_visualization(self.point_idx_camera_fov, "Only Camera FoV")
        # self.start_visualization(self.point_idx_rv_camera_fov, "Combination of RV and Camera FoV")
        # print("End")

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
    
    def get_lidar_points_in_image_plane(self, points, mask_fov):
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

                # Reshape the list into a 4x4 matrix
                Tr = np.eye(4)
                Tr[0:3, 0:4] = np.array([numbers[i:i+4] for i in range(0, len(numbers), 4)])
        
        # Transform LiDAR to left camera coordinates and projection to pixel space as described in KITTI Odometry Readme
        hom_points = np.ones((np.shape(points)[0], 4))
        hom_points[:, 0:3] = points
        proj_points_im = (P2 @ Tr @ hom_points.T).T
        proj_points_im[:, 0] /= proj_points_im[:, 2]
        proj_points_im[:, 1] /= proj_points_im[:, 2]
        proj_points_im = proj_points_im[:, 0:2]

        condition_col1 = (proj_points_im[:, 0] >= 0) & (proj_points_im[:, 0] < self.rgb_data._size[0])
        condition_col2 = (proj_points_im[:, 1] >= 0) & (proj_points_im[:, 1] < self.rgb_data._size[1])

        combined_condition = condition_col1 & condition_col2 & mask_fov

        indices = np.nonzero(combined_condition)[0]

        return indices

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
    
    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0
    
    def start_visualization(self, indices, title):
        self.action = "no"  # no, next, back, quit are the possibilities
        self.indices = indices
        self.viz_title = title

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        self.update_scan()
        self.run()

    def update_scan(self):
        # then change names
        title = self.viz_title
        self.canvas.title = title

        # plot scan
        power = 16
        # print()
        range_data = np.copy(self.unproj_range[self.indices])
        # range_data = np.copy(self.unproj_range)
        # print(range_data.max(), range_data.min())
        range_data = range_data**(1 / power)
        # print(range_data.max(), range_data.min())
        viridis_range = ((range_data - range_data.min()) /
                        (range_data.max() - range_data.min()) *
                        255).astype(np.uint8)
        viridis_map = self.get_mpl_colormap("viridis")
        viridis_colors = viridis_map[viridis_range]
        self.scan_vis.set_data(self.points[self.indices],
                                face_color=viridis_colors[..., ::-1],
                                edge_color=viridis_colors[..., ::-1],
                                size=1)

      # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.update_scan()
        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1
            self.update_scan()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        if self.images:
            self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()

class SemLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
    EXTENSIONS_LABEL = ['.label']

    def __init__(self, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, max_classes=300, aug_prob=None):
        super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down, aug_prob=aug_prob)
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

        if self.drop_points is not False:
            label = np.delete(label,self.points_to_drop)
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

        if self.project:
            self.do_label_projection()

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
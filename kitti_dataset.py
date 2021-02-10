import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import utils

class KittiDataset(Dataset):
    def __init__(self, root, use_image, use_calib, use_velo, use_label):
        self.root = root

        # min, max size
        self.image_size_dict = {'LONG_SIDE': 1333, 'SHORT_SIDE': 800}

        self.images_path = root + '/image_2'
        self.calibs_path = root + '/calib'
        self.velos_path = root + '/velodyne'
        self.labels_path = root + '/label_2'

        self.images = list(sorted(os.listdir(path=self.images_path)))
        self.calibs = list(sorted(os.listdir(path=self.calibs_path)))
        self.velos = list(sorted(os.listdir(path=self.velos_path)))
        self.labels = list(sorted(os.listdir(path=self.labels_path)))

        self.use_image = use_image
        self.use_calib = use_calib
        self.use_velo = use_velo
        self.use_label = use_label

        self.label_encoder = {"Car": 1, "Van": 2, "Truck": 3, "Pedestrian": 4, 
                              "Person_sitting": 5, "Cyclist": 6, "Tram": 7, 
                              "Misc": 8, "DontCare": 9}

    #TODO: image_fv, image_bev, boxes_2d_bev, boxes_3d => transform
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images[idx])
        calib_path = os.path.join(self.calibs_path, self.calibs[idx])
        velo_path = os.path.join(self.velos_path, self.velos[idx])
        label_path = os.path.join(self.labels_path, self.labels[idx])

        image = Image.open(image_path).convert("RGB")
        image, scale_factor, target_size = self.image_transform(image)

        calib = utils.read_calib(calib_path)
        velo = utils.read_velo(velo_path)
        label = utils.read_label(label_path)
        label = self.remove_dontcare(label)

        boxes = [[x.xmin, x.ymin, x.xmax, x.ymax] for x in label]
        boxes = np.asarray(boxes) * scale_factor

        labels = [self.label_encoder[x.type] for x in label]
        # print('Data Load Finished .....')

        image_fv = self.front_view_velodyne(velo[:, :3], calib, target_size[1], target_size[0])
        image_bev = self.bird_eye_view_velodyne(velo)
        # print('Image preprocessing Finished .....')

        boxes_2d_bev = self.boxes2d_in_bev(label, calib)
        boxes_3d = self.boxes3d_projection_in_image(calib, label)
        # print('Boxes preprocessing Finished ......')

        # ToTensor
        # image = torchvision.transforms.ToTensor()(image)
        image_fv = torchvision.transforms.ToTensor()(image_fv)
        image_bev = torchvision.transforms.ToTensor()(image_bev)

        boxes_2d = torch.as_tensor(boxes, dtype=torch.float32)
        boxes_2d_bev = torch.as_tensor(boxes_2d_bev, dtype=torch.float32)
        boxes_3d = torch.as_tensor(boxes_3d, dtype=torch.float32)
         
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # print('Data To Tensor finished ......')
        # velo = torch.as_tensor(velo, dtype=torch.float32)

        target = {}
        target['image'] = image
        # target['image_fv'] = image_fv
        # target['image_bev'] = image_bev

        target['boxes_2d'] = boxes_2d
        # target['boxes_2d_bev'] = boxes_2d_bev
        # target['boxes_3d'] = boxes_3d

        target['labels'] = labels
        
        return target, target_size, scale_factor
    
    def remove_dontcare(self, labels):
        x = []
        for t in labels:
            if t.type != 'DontCare':
                x.append(t)
        
        return x

    def image_transform(self, image):
        w_ori, h_ori = image.width, image.height
        if w_ori > h_ori:   
            target_size = (self.image_size_dict.get('SHORT_SIDE'), self.image_size_dict.get('LONG_SIDE'))
        else:
            target_size = (self.image_size_dict.get('LONG_SIDE'), self.image_size_dict.get('SHORT_SIDE'))
        h_t, w_t = target_size
        scale_factor = min(w_t/w_ori, h_t/h_ori)
        target_size = (round(scale_factor*h_ori), round(scale_factor*w_ori))

        means_norm = (0.485, 0.456, 0.406)
        stds_norm = (0.229, 0.224, 0.225)
        # transform = transforms.Compose([transforms.Resize(target_size),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(mean=means_norm, std=stds_norm)])

        transform = transforms.Compose([transforms.Resize(target_size),
                                transforms.ToTensor()])

        image = transform(image)
        return image, scale_factor, target_size

    ################## 2DBox on Bird Eye View ##################
    def inverse_rigid_transform(self, velo_to_cam):
        velo_to_cam = np.reshape(velo_to_cam, [3, 4])

        inverse = np.zeros_like(velo_to_cam)
        inverse[0:3, 0:3] = np.transpose(velo_to_cam[0:3, 0:3])
        inverse[0:3, 3] = np.dot(-np.transpose(inverse[0:3, 0:3]), velo_to_cam[0:3, 3])
        return inverse

    def cartesian_to_homogeneous(self, point3d):
        n = point3d.shape[0]
        point3d_hom = np.hstack((point3d, np.ones((n, 1))))
        return point3d_hom

    def project_reference_to_velo(self, point3d_reference, V2C):
        point3d_reference = self.cartesian_to_homogeneous(point3d_reference)
        return np.dot(point3d_reference, np.transpose(self.inverse_rigid_transform(V2C)))

    def project_rect_to_reference(self, box3d, R0):
        return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(box3d)))

    def project_rect_to_velo(self, box3d, R0, V2C):
        point3d_reference = self.project_rect_to_reference(box3d, R0)
        return self.project_reference_to_velo(point3d_reference, V2C)

    def boxes2d_in_bev(self, label, calib):
        P = np.reshape(calib['P2'], [3, 4])
        R0 = np.reshape(calib['R0_rect'], [3, 3])
        V2C = calib['Tr_velo_to_cam']
        boxes = []
        for t in label:
            box3d = np.transpose(self.boxes3d_in_image(t))
            box2d_in_bev = self.project_rect_to_velo(box3d, R0, V2C)
            boxes.append(box2d_in_bev)

        return boxes
    #########################################################

    ################## Bird Eye View Lidar ##################
    def rescale(self, pixel, min, max, scale=511, dtype=np.uint8):
        return (((pixel - min) / float(max - min)) * scale).astype(dtype)

    def bird_eye_view_velodyne(self, velo, res=0.1, side_range=(-35.2, 35.2), fwd_range=(0., 60.), height_range=(-2., 2.)):
        x_points = velo[:, 0]
        y_points = velo[:, 1]
        z_points = velo[:, 2]

        f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
        s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filter = np.logical_and(f_filt, s_filt)
        valid_index= np.argwhere(filter).flatten()

        x_points = x_points[valid_index]
        y_points = y_points[valid_index]
        z_points = z_points[valid_index]

        x_img = (-y_points / res).astype(np.int32)
        y_img = (-x_points / res).astype(np.int32)

        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.ceil(fwd_range[1] / res))

        pixel_values = np.clip(a=z_points,
                               a_min=height_range[0],
                               a_max=height_range[1])

        pixel_values = self.rescale(pixel_values,
                               min=height_range[0],
                               max=height_range[1])

        x_max = 1 + int((side_range[1] - side_range[0]) / res)
        y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
        image = np.zeros([y_max, x_max], dtype=np.uint8)
        image[y_img, x_img] = pixel_values
        
        return image
    #########################################################

    ################## Front Of View Lidar ##################
    def project_velodyne_to_camera(self, calib):
        P_velodnye_to_camera = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))
        R_reference_to_rect = np.eye(4)
        R0_rect = calib['R0_rect'].reshape(3, 3)
        R_reference_to_rect[:3, :3] = R0_rect
        P_rect_to_camera = calib['P2'].reshape((3, 4))
        project_matrix = P_rect_to_camera @ R_reference_to_rect @ P_velodnye_to_camera

        return project_matrix

    def project_to_image(self, velo, project_matrix):
        num_velo = velo.shape[1]

        points = np.vstack((velo, np.ones((1, num_velo))))
        points = project_matrix @ points
        points[:2, :] /= points[2, :]
        return points[:2, :]


    def front_view_velodyne(self, velo, calib, width, height):
        project_velodyne_to_camera = self.project_velodyne_to_camera(calib)
        point2d = self.project_to_image(velo.transpose(), project_velodyne_to_camera)

        valid_index = np.where((point2d[0, :] < width) & (point2d[0, :] >= 0) &
                               (point2d[1, :] < height) & (point2d[1, :] >= 0) &
                               (velo[:, 0] > 0))[0]

        fv_image = point2d[:, valid_index]

        fv_velo = velo[valid_index, :]
        fv_velo = np.hstack((fv_velo, np.ones((fv_velo.shape[0], 1))))
        fv_cam = project_velodyne_to_camera @ fv_velo.transpose()

        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        black_image = np.zeros((height, width, 3), np.uint8)
        black_image[:, 0:width//2] == (255, 0, 0)

        for i in range(fv_image.shape[1]):
            depth = fv_cam[2, i]
            color = cmap[int(640.0 / depth), :]
            cv2.circle(black_image, 
                       (int(np.round(fv_image[0, i])), int(np.round(fv_image[1, i]))),
                       2, 
                       color=tuple(color), 
                       thickness=-1)
        
        return black_image
    #########################################################


    ################## 3D Box On Image ##################
    # [x1, x2, x3, ..., x8, y1, y2, y3, ... y8, z1, z2, z3, ..., z8]
    def boxes3d_projection_in_image(self, calib, label):
        P_rect_to_cam = calib['P2'].reshape((3, 4))
        point3d = []
        for t in label:
            point3d.append(self.boxes3d_in_image(t))

        return point3d

    def boxes3d_in_image(self, label):
        l = label.l
        w = label.w
        h = label.h

        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        R = utils.rotation_y(label.ry)
        point3d = R @ box_coord

        point3d[0, :] = point3d[0, :] + label.t[0]
        point3d[1, :] = point3d[1, :] + label.t[1]
        point3d[2, :] = point3d[2, :] + label.t[2]

        return point3d
    #####################################################

    def __len__(self):
        return len(self.images)
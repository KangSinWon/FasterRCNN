import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
from PIL import Image

import utils

class KittiDataset(Dataset):
    def __init__(self, root, use_image, use_calib, use_velo, use_label):
        self.root = root

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

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images[idx])
        calib_path = os.path.join(self.calibs_path, self.calibs[idx])
        velo_path = os.path.join(self.velos_path, self.velos[idx])
        label_path = os.path.join(self.labels_path, self.labels[idx])

        image = Image.open(image_path).convert("RGB")
        calib = utils.read_calib(calib_path)
        velo = utils.read_velo(velo_path)
        label = utils.read_label(label_path)

        boxes = [[x.xmin, x.xmin, x.xmin, x.xmin] for x in label]
        labels = [self.label_encoder[x.type] for x in label]

        #TODO: boxes_2d_bird_eye_view, image_of_velodyne, image_of_bird_eye_view
        image = torchvision.transforms.ToTensor()(image)
        # image_fv = 
        # image_bev = 

        boxes_2d = torch.as_tensor(boxes, dtype=torch.float32)
        # boxes_2d_bev = 
        boxes_3d = self.boxes3d_projection_in_image(calib, label)

        labels = torch.as_tensor(labels, dtype=torch.int64)
        velo = torch.as_tensor(velo, dtype=torch.float32)

        # print(calib)
        # print(velo)
        # print(labels)
        # print(boxes)

    # [x1, x2, x3, ..., x8, y1, y2, y3, ... y8, z1, z2, z3, ..., z8]
    def boxes3d_projection_in_image(self, calib, label):
        P_rect_to_cam = calib['P2'].reshape((3, 4))
        point3ds = []
        for t in label:
            point3ds.append(self.boxes3d_in_image(t))

        return point3ds

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

    def __len__(self):
        return len(self.images)
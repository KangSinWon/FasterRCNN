import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

from anchor import Anchor
from rpn import RegionProposalNet
from kitti_dataset import KittiDataset
from faster_rcnn import FasterRCNN
from vgg16 import VGG16

def collate_fn(batch):
    image_list = []
    box_list = []
    label_list = []

    for i in range(batch_size):
        image_list.append(batch[i][0].numpy())
        box_list.append(batch[i][1].numpy())
        label_list.append(batch[i][2].numpy())

    return torch.as_tensor(image_list), box_list, label_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Learning Machine:', device)
classes = ["Car", "Van", "Truck", 
            "Pedestrian", "Person_sitting", "Cyclist", 
            "Tram", "Misc"]

n_epochs = 100
batch_size = 4
stride = 16

kitti = KittiDataset('/home/ksw/Data/Kitti/training', True, True, True, True)
torch.manual_seed(1)
train_data_loader = DataLoader(
    kitti,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

faster_rcnn = VGG16(classes)
faster_rcnn.zero_grad()
faster_rcnn.create_architecture()

params = [p for p in faster_rcnn.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# TEST
# data_iter = iter(train_data_loader)
# data = next(data_iter)

for epoch in range(n_epochs):
    for data in train_data_loader:
        rpn_cls_loss, rpn_bbox_loss, \
        roi_cls_loss, roi_bbox_loss, \
        rpn_loss, loss = faster_rcnn(data[0], data[1], data[2])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('-------------- LOSS --------------')
        print('rpn_cls_loss', rpn_cls_loss)
        print('rpn_bbox_loss', rpn_bbox_loss)
        print('roi_cls_loss', roi_cls_loss)
        print('roi_bbox_loss', roi_bbox_loss)
        print('loss', loss, '\n')
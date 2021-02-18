import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from faster_rcnn import FasterRCNN

class VGG16(FasterRCNN):
        def __init__(self, classes, pretrained=True, class_agnostic=False):
            # vgg16 = torchvision.models.vgg16(pretrained=True)
            # self.vgg16_features = nn.Sequential(vgg16.features)
            self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
            self.pretrained = pretrained
            self.class_agnostic = class_agnostic

            FasterRCNN.__init__(self, classes, class_agnostic)

        def _init_modules(self):
            vgg = models.vgg16()

            if self.pretrained:
                print("Loading pretrained weights VGG16")
                state_dict = torch.load(self.model_path)
                vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

            vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

            # VGG16 Feature extractor
            self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

            for layer in range(10):
                for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

            self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

            if self.class_agnostic:
                self.RCNN_bbox_pred = nn.Linear(4096, 4)
            else:
                self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
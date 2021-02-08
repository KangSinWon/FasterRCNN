import torch
import torch.nn as nn
import torchvision

class VGG16(nn.Module):
        def __init__(self):
            super(VGG16, self).__init__()

            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_features = nn.Sequential(vgg16.features)
            
        def forward(self, x):
            return self.vgg16_features(x)
import torch
import torch.nn as nn
import numpy as np


class ProposalLayer(nn.Module):
    def __init__(self):
        super(ProposalLayer, self).__init__()

        self.NMS_thresh = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000


    def forward(self, anchor_boxes, rpn_cls_score, rpn_bbox_pred):
        print('')

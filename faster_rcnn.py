import torch
import torch.nn as nn

class FasterRCNN(nn.Module):

    def __init__(self):
        super(FasterRCNN, self).__init__()

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0        

        # rpn
        # self.rpn = rpn()
        # self.proposal_target =
        # self.roi_pool = 
        # self.roi_align 

    # def forward(self):
        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
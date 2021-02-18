import torch
import torch.nn as nn

from rpn import RegionProposalNet

class FasterRCNN(nn.Module):

    def __init__(self, classes, class_agnostic):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0        

        # rpn
        self.rpn = RegionProposalNet(512, 512)
        
        # self.proposal_target =
        # self.roi_pool = 
        # self.roi_align 

    def forward(self, image, gt_boxes, labels):
        batch_size = len(image)

        base_feature = self.RCNN_base(image)

        rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feature, gt_boxes)
        return base_feature


    # def _init_weights(self):
    #     def normal_init(m, mean, stddev, truncated=False):

    #         if truncated:
    #             m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    #         else:
    #             m.weight.data.normal_(mean, stddev)
    #             m.bias.data.zero_()
        
    #     normal_init(self.RCNN_rpn.)

    def create_architecture(self):
        self._init_modules()
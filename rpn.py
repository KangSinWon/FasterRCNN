import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor import Anchor
from anchor_target_layer import AnchorTargetLayer
from proposal_layer import ProposalLayer


class RegionProposalNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegionProposalNet, self).__init__()

        # TODO: parameter
        self.anchor_boxes = Anchor(1333, 402, 16)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]
        self.stride = 16
        
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv1.bias.data.zero_()
        
        self.rpn_regressor_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.bounding_box_regressor = nn.Conv2d(self.out_channels, self.rpn_regressor_out, 1, 1, 0)
        # self.bounding_box_regressor.weight.data.normal_(0, 0.01)
        # self.bounding_box_regressor.bias.data.zero_()

        self.rpn_classifier_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.classifier = nn.Conv2d(self.out_channels, self.rpn_classifier_out, 1, 1, 0)
        # self.classifier.weight.data.normal_(0, 0.01)
        # self.classifier.bias.data.zero_()

        self.rpn_proposal = ProposalLayer()
        self.rpn_anchor_target = AnchorTargetLayer()
        
    def forward(self, feature, gt_boxes):
        batch_size = feature.size(0)
        rpn_conv1 = F.relu(self.conv1(feature), inplace=True)
        # [4, 18, 25, 83]
        rpn_cls_score = self.classifier(rpn_conv1)
        # [4, 18, 25, 83]
        rpn_bbox_pred = self.bounding_box_regressor(rpn_conv1)

        # RPN PROPOSAL LAYER TODO
        # rois = self.rpn_proposal(self.anchor_boxes.inside_anchor_boxes, rpn_cls_score, rpn_bbox_pred)

        ############## if training calculate loss ##############

        # RPN ANCHOR TARGET LAYER TODO
        rpn_data = self.rpn_anchor_target(rpn_cls_score, self.anchor_boxes, gt_boxes)

        # cls loss
        # [4, 18675, 2]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        # Gt Label

        ########################################################


    # TODO: Multi-task loss
    def multi_task_loss(rpn_box_reg, rpn_cls, gt_box_reg, gt_cls):
        rpn_cls_loss = F.cross_entropy(rpn_cls, gt_cls.long(), ignore_index=-1)

        pos = gt_cls > 0
        mask = pos.unsqueeze(1).expand_as(rpn_box_reg)

        mask_box_preds = rpn_box_reg[mask].view(-1, 4)
        mask_box_targets = gt_box_reg[mask].view(-1, 4)

        x = torch.abs(mask_box_targets.cpu() - mask_box_preds.cpu())
        rpn_reg_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5)) 

        rpn_lambda = 10
        N_reg = (gt_cls > 0).float().sum()

        rpn_reg_loss = rpn_reg_loss.sum() / N_reg
        rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_reg_loss)

        return rpn_loss
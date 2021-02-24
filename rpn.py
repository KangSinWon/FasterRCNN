import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        self.anchor_scales = [4, 8, 16]
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

        # self.rpn_proposal = ProposalLayer()
        self.rpn_anchor_target_layer = AnchorTargetLayer()
        
    def forward(self, feature, gt_boxes):
        batch_size = feature.size(0)
        rpn_conv1 = F.relu(self.conv1(feature), inplace=True)
        # [4, 18, 25, 83]
        rpn_cls_score = self.classifier(rpn_conv1)
        # [4, 18, 25, 83]
        rpn_bbox_pred = self.bounding_box_regressor(rpn_conv1)

        ############## if training calculate loss ##############

        # RPN ANCHOR TARGET LAYER TODO
        anchor_target_label, anchor_target_bbox = self.rpn_anchor_target_layer(rpn_cls_score, self.anchor_boxes, gt_boxes)
        # [4, 18675]
        anchor_target_label = torch.from_numpy(anchor_target_label)
        # [4, 18675, 4]
        anchor_target_bbox = torch.from_numpy(anchor_target_bbox)
        # print('anchor_target_bbox shape', anchor_target_bbox.shape)

        # cls loss
        # [4, 18675, 2]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        # [4, 2, 18675]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 1)

        rpn_cls_loss = F.cross_entropy(rpn_cls_score, anchor_target_label.long(), ignore_index=-1)
        # print('rpn_cls_loss:', rpn_cls_loss)

        # bbox loos
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        # print('rpn_bbox_pred shape', rpn_bbox_pred.shape)

        # [4, 18675, 2]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 1)

        # batch_pos = anchor_target_label[:, ] > 0
        # print('###', batch_pos.shape)
        # t = batch_pos[0].unsqueeze(1).expand_as(rpn_bbox_pred[0])
        # print(np.where(t == True)[0].shape)

        batch_mask = []
        batch_mask_bbox_preds = []
        batch_mask_bbox_targets= []

        for i in range(batch_size):
            batch_pos = anchor_target_label[i] > 0

            batch_mask.append(batch_pos.unsqueeze(1).expand_as(rpn_bbox_pred[i]))

            # [4, 128, 4] [tensor in list]
            batch_mask_bbox_preds.append(rpn_bbox_pred[i][batch_mask[i]].view(-1, 4))
            batch_mask_bbox_targets.append(anchor_target_bbox[i][batch_mask[i]].view(-1, 4))

        batch_mask_bbox_preds = torch.stack(batch_mask_bbox_preds)
        batch_mask_bbox_targets = torch.stack(batch_mask_bbox_targets)
        x = torch.abs(batch_mask_bbox_targets.float() - batch_mask_bbox_preds.float())
        rpn_bbox_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
        rpn_bbox_loss = rpn_bbox_loss.sum()
        # print('rpn_bbox_loss:', rpn_bbox_loss.sum())

        rpn_lamda = 10
        N_reg = (anchor_target_label[:, ] > 0).float().sum()

        rpn_bbox_loss = rpn_bbox_loss.sum() / N_reg
        rpn_loss = rpn_cls_loss + (rpn_lamda * rpn_bbox_loss)
        # print('rpn_loss:', rpn_loss)

        return self.anchor_boxes, rpn_cls_score, rpn_bbox_pred, rpn_cls_loss, rpn_bbox_loss, rpn_loss


    # TODO: Multi-task loss
    # def multi_task_loss(rpn_box_reg, rpn_cls, gt_box_reg, gt_cls):
    #     rpn_cls_loss = F.cross_entropy(rpn_cls, gt_cls.long(), ignore_index=-1)

    #     pos = gt_cls > 0
    #     mask = pos.unsqueeze(1).expand_as(rpn_box_reg)

    #     mask_box_preds = rpn_box_reg[mask].view(-1, 4)
    #     mask_box_targets = gt_box_reg[mask].view(-1, 4)

    #     x = torch.abs(mask_box_targets.cpu() - mask_box_preds.cpu())
    #     rpn_reg_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5)) 

    #     rpn_lambda = 10
    #     N_reg = (gt_cls > 0).float().sum()

    #     rpn_reg_loss = rpn_reg_loss.sum() / N_reg
    #     rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_reg_loss)

    #     return rpn_loss
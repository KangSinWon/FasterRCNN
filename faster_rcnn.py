import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2

from rpn import RegionProposalNet
from proposal_layer import ProposalLayer
from anchor import Anchor
from proposal_target_layer import ProposalTargetLayer

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
        
        self.proposal_layer = ProposalLayer()
        self.proposal_target_layer = ProposalTargetLayer()

        # Roi pooling
        self.size = (7, 7)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(self.size)

        self.roi_head = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)])
        self.cls_bbox = nn.Linear(4096, (self.n_classes + 1) * 4)
        self.cls_score = nn.Linear(4096, self.n_classes + 1)


    def forward(self, image, gt_boxes, labels):
        print('foward faster rcnn:', image.shape)
        batch_size = len(image)

        base_feature = self.RCNN_base(image)
        # print('Base feature size:', base_feature.size())

        anchor, rpn_cls_score, rpn_bbox_pred, rpn_cls_loss, rpn_bbox_loss, rpn_loss = self.rpn(base_feature, gt_boxes)

        proposal_layer_roi = self.proposal_layer(anchor, rpn_cls_score, rpn_bbox_pred, rpn_cls_loss, rpn_bbox_loss)
        proposal_layer_roi = torch.as_tensor(proposal_layer_roi)
        print('proposal_layer shape', proposal_layer_roi.shape)

        # TODO: Proposal Target layer | Remove image parameter
        # [4, 128, 4]
        sample_roi, gt_roi_label, gt_roi_bbox = self.proposal_target_layer(proposal_layer_roi, gt_boxes, labels, image)

        rois = torch.as_tensor(sample_roi).float()
        rois.mul_(1 / 16.0)
        rois = rois.long()

        batch_output = []
        for i in range(batch_size):
            output = []
            num_rois = len(rois[i])

            for roi in rois[i]:
                roi_feature = base_feature[i][..., roi[1]:roi[3]+1, roi[0]:roi[2]+1]
                output.append(self.adaptive_max_pool(roi_feature))
            # print(len(output), output[0].shape)
            output = torch.stack(output)
            output_roi_pooling = output.view(output.size(0), -1)
            # [128, 25088]
            # print(output_roi_pooling.shape)
            batch_output.append(output_roi_pooling)
        batch_output = torch.stack(batch_output)
        # [4, 128, 25088]
        # print(batch_output.shape)

        # Fast RCNN
        x = self.roi_head(batch_output)
        roi_cls_bbox = self.cls_bbox(x)
        roi_cls_score = self.cls_score(x)
        # [4, 128, 40], [4, 128, 10]
        # print(roi_cls_bbox.shape, roi_cls_score.shape)

        # Fast RCNN loss
        gt_roi_bbox = torch.as_tensor(gt_roi_bbox)
        gt_roi_label = torch.as_tensor(np.float32(gt_roi_label)).long()
        # [4, 128, 4] , [4, 128]
        # print(gt_roi_bbox.shape, gt_roi_label.shape)

        # Classification loss
        roi_cls_score = roi_cls_score.permute(0, 2, 1)
        roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index = -1)
        # print('roi_cls_loss:', roi_cls_loss)

        # Regression loss
        n_sample = roi_cls_bbox.shape[1]
        roi_bbox = roi_cls_bbox.view(batch_size, n_sample, -1, 4)
        # print(roi_bbox.shape)

        batch_roi_bbox_loss = []
        for i in range(batch_size):
            u = roi_bbox[i][torch.arange(0, n_sample).long(), gt_roi_label[i]]
            
            pos = gt_roi_label[i] > 0
            mask = pos.unsqueeze(1).expand_as(u)

            mask_bbox_preds = u[mask].view(-1, 4)
            mask_bbox_targets = gt_roi_bbox[i][mask].view(-1, 4)

            x = torch.abs(mask_bbox_targets - mask_bbox_preds)
            roi_bbox_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
            batch_roi_bbox_loss.append(roi_bbox_loss)

            # print('roi_bbox_loss:', roi_bbox_loss.sum())
        
        roi_bbox_loss = sum([l.sum() for l in batch_roi_bbox_loss])
        # print('roi_bbox_loss:', roi_bbox_loss)

        roi_lambda = 10.
        roi_loss = roi_cls_loss + (roi_lambda * roi_bbox_loss)
        loss = rpn_loss + roi_loss
        # print('loss', loss)
        
        return rpn_cls_loss, rpn_bbox_loss, roi_cls_loss, roi_bbox_loss, rpn_loss, loss

    def create_architecture(self):
        self._init_modules()
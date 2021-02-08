import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import DataLoader

from anchor import Anchor
from rpn import RPN
from kitti_dataset import KittiDataset

def train():
    img_width = 1224
    img_height = 370
    stride = 16

    anchor = Anchor(img_width, img_height, stride)
    ious = anchor.get_ious()

    # max socre -> each gt objects
    gt_objects_argmax_ious_ = ious.argmax(axis=0)
    gt_max_ious = ious[gt_objects_argmax_ious_, np.arange(ious.shape[1])]
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]

    # max score -> each anchor_boxes
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(anchor.inside_index)), argmax_ious]

    label = np.empty((len(anchor.inside_index),), dtype=np.int32)
    label.fill(-1)

    positive_iou_threshold = 0.7
    negative_iou_threshold = 0.3

    # max score
    label[gt_argmax_ious] = 1
    label[max_ious >= positive_iou_threshold] = 1
    label[max_ious < negative_iou_threshold] = 0
    # Remainder is -1

    # mini-batch size
    n_sample = 256
    positive_ratio = 0.5
    n_positive = positive_ratio * n_sample
    n_negative = (1 - positive_ratio) * n_sample

    positive_index = np.where(label == 1)[0]

    if len(positive_index) > n_positive:
        disable_positive_index = np.random.choice(positive_index,
                                                  size = (len(positive_index) - n_positive),
                                                  replace=False)
        label[disable_positive_index] = -1

    negative_index = np.where(label == 0)[0]
    if len(negative_index) > n_negative:
        disable_negative_index = np.random.choice(negative_index,
                                                  size = (len(negative_index) - n_negative),
                                                  replace=False)
        label[disable_negative_index] = -1

    # TODO: define => gt_boxes
    max_iou_bbox = gt_boxes[argmax_ious]

    height = anchor.inside_anchor_boxes[:, 3] - anchor.inside_anchor_boxes[:, 1]
    width = anchor.inside_anchor_boxes[:, 2] - anchor.inside_anchor_boxes[:, 0]
    center_y = anchor.inside_anchor_boxes[:, 1] + 0.5 * height
    center_x = anchor.inside_anchor_boxes[:, 0] + 0.5 * width

    base_height = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
    base_width = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
    base_center_y = max_iou_bbox[:, 1] + 0.5 * base_height
    base_center_x = max_iou_bbox[:, 0] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    tx = (base_center_x - center_x) / width
    ty = (base_center_y - center_y) / height
    tw = np.log(base_width / width)
    th = np.log(base_height / height)

    # dx, dy, dw, dh ] * anchor_boxes_n
    anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()

    # all anchor_boxes
    anchor_labels = np.empty((len(anchor.anchor_boxes),), dtype=label.dtype)
    anchor_labels.fill(-1)
    anchor_labels[anchor.inside_index] = label

    anchor_locations = np.empty((len(anchor.anchor_boxes),) + anchor.anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[anchor.inside_index, :] = anchor_locs

    # RPN
    rpn = RPN(512, 512, 9)
    pred_box_reg, pred_cls = rpn(feature_map)

    pred_box_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
    pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()
    objectness_score = pred_cls.view(1, anchor.features_map_width, anchor.features_map_height, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
    pred_cls = pred_cls.view(1, -1, 2)

    # prediction
    rpn_box_reg = pred_box_reg[0]
    rpn_cls = pred_cls[0]

    # ground truth
    gt_box_reg = torch.from_numpy(anchor_locations)
    gt_cls = torch.from_numpy(anchor_labels)
    rpn.multi_task_loss(rpn_box_reg, rpn_cls, gt_box_reg, gt_cls)

    # NMS #####################################################################
    # 예측한 로케이션을 앵커를 통해 roi로 변환 ?????
    nms_thresh = 0.7
    n_train_pre_nms = 12000
    n_train_post_nms = 2000

    min_width_size = 76
    min_height_size = 23

    anc_height = anchor.anchor_boxes[:, 3] - anchor.anchor_boxes[:, 1]
    anc_width = anchor.anchor_boxes[:, 2] - anchor.anchor_boxes[:, 0]
    anc_center_y = anchor.anchor_boxes[:, 1] + 0.5 * anc_height
    anc_cneter_x = anchor.anchor_boxes[:, 0] + 0.5 * anc_width

    # forward(pred_regressor(x))
    pred_anchor_locs_numpy = pred_box_reg[0].cpu().data.numpy()
    dy, dx, dh, dw = pred_anchor_locs_numpy.T

    base_height = np.exp(dh) * anc_height
    base_width = np.exp(dw) * anc_width
    base_center_y = dy * anc_height + anc_center_y
    base_center_x = dx * anc_width + anc_center_x

    base_anchors = np.zeros_like(anchor.anchor_boxes)
    base_anchors[:, 0] = base_center_y - base_height * 0.5
    base_anchors[:, 1] = base_center_x - base_width * 0.5
    base_anchors[:, 2] = base_center_y - base_height * 0.5
    base_anchors[:, 3] = base_center_x - base_width * 0.5
    roi = base_anchors

    img_size = (1224, 370)
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

    h = rois[:, 2] - rois[:, 0]
    w = rois[:, 3] - rois[:, 1]
    keep = np.where((hs >= min_height_size) & (ws >= min_width_size))[0]
    roi = roi[keep, :]

    objectness_score_numpy = objectness_score[0].cpu().data.numpy()
    score = objectness_score_numpy[keep]

    #Sort
    order = score.ravel().argsort()[::-1]

    # select 12000
    order = order[:n_train_pre_nms]
    roi = roi[order, :]

    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # select 2000
    order = order.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    keep = keep[:n_train_post_nms]
    roi = roi[keep]
    #############################################################################

    # Fast RCNN #################################################################
    # calculate IOUS
    n_sample = 128
    pos_ratio = 0.25
    pos_iou_thresh = 0.5
    neg_iou_thresh_hi = 0.5
    neg_iou_thresh_lo =0.0

    # define ground_truth bbox
    ious = np.empty((len(roi), bbox.shape[0]), dtype=np.float32)
    ious.fill(0)

    for num1, i in enumerate(roi):
        ya1, xa1, ya2, xa2 = i
        anchor_area = (ya2 - ya1) * (xa2 - xa1)

        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)

            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([xb2, ya2])

            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter_area = (inter_y2 - intery1) * (inter_x2 - inter_x1)
                iou = inter_area / (anchor_area + box_area - inter_area)
            else:
                iou = 0
            ious[num1, num2] = iou

    gt_assignment = ious.argmax(axis=1)
    max_iou = ious.max(axis=1)
    gt_roi_label = labels[gt_assignment]

    ############# SELECT POSITIVE SAMPLES #########################
    # n_sample * pos_ratio
    pos_roi_per_image = 32
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))

    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)


    ############# SELECT NEGATIVE SAMPLES #########################
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    

    ############### GATHER POS/NEG samples #####################
    keep_index = np.append(pos_index, neg_index)
    gt_roi_labels = gt_roi_label[keep_index]
    gt_roi_labels[pos_roi_per_this_image:] = 0
    sample_roi = roi[keep_index]

    # [128, 4]
    bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]

    width = sample_roi[:, 2] - sample_roi[:, 0]
    height = sample_roi[:, 3] - sample_roi[:, 1]
    center_x = sample_roi[:, 0] + 0.5 * width
    center_y = sample_roi[:, 1] + 0.5 * height

    base_width = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
    base_height = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
    base_center_x = bbox_for_sampled_roi[:, 0] + 0.5 * base_width
    base_center_y = bbox_for_sampled_roi[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dx = (base_center_x - center_x) / width
    dy = (base_center_y - center_y) / height
    dw = np.log(base_width/ width)
    dh = np.log(base_height / height)

    gt_roi_locs = np.vstack((dx, dy, dw, dh)).transpose()


    ####################### ROI POOLING #######################
    rois = torch.from_numpy(sample_roi).float()
    roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
    roi_indices = torch.from_numpy(roi_indices).float()

    indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    indices_and_rois = xy_indices_and_rois.contiguous()

    size = (7, 7)
    adaptive_max_pool = nn.AdaptiveMaxPool2d(size[0], size[1])

    output = []
    rois = indices_and_rois.data.float()

    # change ratio
    rois[:, 1:].mul_(1 / 16.0)
    rois = rois.long()
    num_rois = rois.size(0)

    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = output_map.narrow(0, im_idx, 1)[..., roi[1]:(roi[3]+1), roi[2]:(roi[4]+1)]
        tmp = adaptive_max_pool(im)
        output.append(tmp[0])

    output = torch.cat(output, 0)

    k = output.view(output.size(0), -1)
    ##################### Fast R-CNN #############################
    roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)]).to(DEVICE)

    cls_loc = nn.Linear(4096, 2 * 4).to(DEIVCE)
    cls_loc.weight.data.normal_(0, 0.01)
    cls_loc.bias.data.zero_()

    score = nn.Linear(4096, 2).to(DEIVCE)

    # FORWARD
    k = roi_head_classifier(k.to(DEVICE))
    roi_cls_loc = cls_loc(k)
    roi_cls_score = score(k)

    # Classification loss
    gt_roi_loc = torch.from_numpy(gt_roi_locs)
    gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()

    roi_cls_loss = F.cross_entropy(roi_cls_score.cpu(), gt_roi_label.cpu(), ignore_index=-1)

    # Regression loss
    n_sample = roi_cls_loc.shape[0]
    roi_loc = roi_cls_loc.view(n_sample, -1, 4)

    roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]

    pos = gt_roi_label > 0
    mask = pos.unsqueeze(1).expand_as(roi_loc)

    mask_loc_preds = roi_loc[mask].view(-1, 4)
    mask_loc_targets = gt_roi_loc[mask].view(-1, 4)

    x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
    roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x > 1).float() * (x - 0.5))

    # Multi task loss
    roi_lambda = 10.
    roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
    total_loss = rpn_loss + roi_loss


if __name__ == "__main__":
    kitti = KittiDataset('/home/kangsinwon/3D_Object_Detection/KITTI_DATA/training', True, True, True, True)
    kitti[10]
    train_data_loader = DataLoader (
        kitti,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )
    # train()

    # epoch = 100
    # for i in range(epoch):

import torch
import torch.nn as nn
import numpy as np


# anchor box와 gt box를 비교(iou)하여 pos>0.7, neg<0.3
class AnchorTargetLayer(nn.Module):
    def __init__(self):
        super(AnchorTargetLayer, self).__init__()

        self.pos_iou_threshold = 0.7
        self.neg_iou_threshold = 0.3

        self.n_sample = 256
        self.pos_ratio = 0.5
        self.n_pos = self.pos_ratio * self.n_sample

    def forward(self, rpn_cls_score, anchor, gt_boxes):
        batch_size = rpn_cls_score.size(0)

        ious_label = np.empty((batch_size, len(anchor.inside_index)), dtype=np.int32)
        ious_label.fill(-1)

        # TODO: Get max ious
        # print('anchor boxes:', anchor.inside_anchor_boxes)
        # print('gt_boxes:', gt_boxes)

        batch_ious = []

        for b in range(batch_size):
            ious = np.zeros((len(anchor.inside_index), len(gt_boxes[b])), dtype=np.float32)
            ious.fill(0)

            for i, anchor_box in enumerate(anchor.inside_anchor_boxes):
                xa1, ya1, xa2, ya2 = anchor_box
                anchor_area = (xa2 - xa1) * (ya2 - ya1)

                for j, gt_box in enumerate(gt_boxes[b]):
                    xg1, yg1, xg2, yg2 = gt_box
                    box_area = (xg2 - xg1) * (yg2 -yg1)

                    inter_x1 = max([xg1, xa1])
                    inter_y1 = max([yg1, ya1])
                    inter_x2 = min([xg2, xa2])
                    inter_y2 = min([yg2, ya2])

                    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        iou = inter_area / (anchor_area + box_area - inter_area)
                    else:
                        iou = 0

                    ious[i, j] = iou

            batch_ious.append(ious)

        # [batch_size, inside_achor_boxes, n_gt_boxes]
        # print('ious:', batch_ious)

        # pos -> 1, neg -> -1
        # BY Anchor
        batch_achor_max_iou = []
        batch_pos_iou_anchor_label = []
        batch_neg_iou_anchor_label = []

        for i in range(batch_size):
            batch_achor_max_iou.append(np.amax(batch_ious[i], axis=1))

            batch_pos_iou_anchor_label.append(np.where(batch_achor_max_iou[i] >= self.pos_iou_threshold)[0])
            batch_neg_iou_anchor_label.append(np.where(batch_achor_max_iou[i] < self.neg_iou_threshold)[0])

            ious_label[i][batch_pos_iou_anchor_label[i]] = 1
            ious_label[i][batch_neg_iou_anchor_label[i]] = 0
        
        # BY gt box
        batch_gt_max_iou = []
        batch_gt_max_iou_anchor_label = []
        for i in range(batch_size):
            batch_gt_max_iou.append(np.amax(batch_ious[i], axis=0))
            batch_gt_max_iou_anchor_label.append(np.where(batch_ious[i] == batch_gt_max_iou[i])[0])
            ious_label[i][batch_gt_max_iou_anchor_label[i]] = 1

        # Sampling mini_batch size 256
        batch_pos_index = []
        batch_neg_index = []
        for i in range(batch_size):
            batch_pos_index.append(np.where(ious_label[i] == 1)[0])
            batch_neg_index.append(np.where(ious_label[i] == 0)[0])

            if len(batch_pos_index[i]) > self.n_pos:
                disable_index = np.random.choice(batch_pos_index[i], size=(int(len(batch_pos_index[i]) - self.n_pos)), replace=False)
                ious_label[i][disable_index] = -1
            else:
                able_index = np.random.choice(batch_neg_index[i], size=(int(self.n_pos - len(batch_pos_index[i]))), replace=False)
                ious_label[i][able_index] = 1
                # update
                batch_neg_index.pop()
                batch_neg_index.append(np.where(ious_label[i] == 0)[0])

            if len(batch_neg_index[i]) > self.n_pos:
                disable_index = np.random.choice(batch_neg_index[i], size=(int(len(batch_neg_index[i]) - self.n_pos)), replace=False)
                ious_label[i][disable_index] = -1

            if np.where(ious_label[i] == 1)[0].shape[0] != 128 and np.where(ious_label[i] == 0)[0].shape[0] != 128:
                print('mini-batch size is not matching', i)
                print(np.where(ious_label[i] == 1)[0].shape)
                print(np.where(ious_label[i] == 0)[0].shape)

        # Convert the format of inside anchor boxes [x1, y1, x2, y2]
        h = anchor.inside_anchor_boxes[:, 3] - anchor.inside_anchor_boxes[:, 1]
        w = anchor.inside_anchor_boxes[:, 2] - anchor.inside_anchor_boxes[:, 0]
        center_x = anchor.inside_anchor_boxes[:, 0] - 0.5 * w
        center_y = anchor.inside_anchor_boxes[:, 1] + 0.5 * h

        eps = np.finfo(h.dtype).eps
        h = np.maximum(h, eps)
        w = np.maximum(w, eps)

        batch_anchor_locs = []
        for i in range(batch_size):
            max_iou_bbox = anchor.anchor_boxes[np.argmax(batch_ious[i], axis=1)]

            max_iou_bbox_h = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
            max_iou_bbox_w = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
            max_iou_bbox_center_x = max_iou_bbox[:, 0] + 0.5 * max_iou_bbox_w
            max_iou_bbox_center_y = max_iou_bbox[:, 1] + 0.5 * max_iou_bbox_h
            
            dx = (max_iou_bbox_center_x - center_x) / w
            dy = (max_iou_bbox_center_y - center_y) / h
            dh = np.log(max_iou_bbox_h / h)
            dw = np.log(max_iou_bbox_w / w)

            batch_anchor_locs.append(np.vstack((dx, dy, dw, dh)).transpose())
            # anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()
            
        anchor_target_labels = np.empty((batch_size, len(anchor.anchor_boxes)), dtype=np.int32)
        anchor_target_labels.fill(-1)

        anchor_target_bbox = np.empty((batch_size, len(anchor.anchor_boxes)) + anchor.anchor_boxes.shape[1:], dtype=batch_anchor_locs[0].dtype)
        anchor_target_bbox.fill(0)

        for i in range(batch_size):
            anchor_target_labels[i][anchor.inside_index] = ious_label[i]
            anchor_target_bbox[i][anchor.inside_index, :] = batch_anchor_locs[i]

        return anchor_target_labels, anchor_target_bbox
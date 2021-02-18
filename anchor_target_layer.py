import torch
import torch.nn as nn
import numpy as np


# anchor box와 gt box를 비교(iou)하여 pos>0.7, neg<0.3
class AnchorTargetLayer(nn.Module):
    def __init__(self):
        super(AnchorTargetLayer, self).__init__()

        self.pos_iou_threshold = 0.7
        self.neg_iou_threshold = 0.3

    def forward(self, rpn_cls_score, anchor, gt_boxes):
        batch_size = rpn_cls_score.size(0)

        ious_label = np.empty((batch_size, len(anchor.inside_index)), dtype=np.int32)
        ious_label.fill(-1)

        # TODO: Get max ious
        # print('anchor boxes:', anchor.inside_anchor_boxes)
        # print('gt_boxes:', gt_boxes)

        batch_ious = []

        for b in range(batch_size):
            ious = np.zeros((len(anchor.inside_index), len(gt_boxes[b])), dtype=np.int32)
            ious.fill(-1)

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
        for i in range(batch_size):
            batch_achor_max_iou.append(np.amax(batch_ious[i], axis=1))

        print(np.amax(batch_ious[0], axis=1))

        # batch_pos_iou_anchor_label = []
        # for i in range(batch_size):
        #     batch_pos_iou_anchor_label.append(np.where(batch_achor_max_iou[i] >= self.pos_iou_threshold)[0])

        # batch_neg_iou_anchor_label = []
        # for i in range(batch_size):
        #     batch_neg_iou_anchor_label.append(np.where(batch_achor_max_iou[i] < self.neg_iou_threshold)[0])

        # for i in range(batch_size):
        #     for x in batch_pos_iou_anchor_label:
        #         ious_label[i][x] = 1

        #     for y in batch_neg_iou_anchor_label:
        #         ious_label[i][y] = 0

        # # BY gt box
        # batch_gt_max_iou = []
        # for i in range(batch_size):
        #     batch_gt_max_iou.append(np.amax(batch_ious[i], axis=0))

        # print('2', batch_gt_max_iou)

        # batch_gt_max_iou_anchor_label = []
        # for i in range(batch_size):
        #     batch_gt_max_iou_anchor_label.append(np.where(batch_ious[i] == batch_gt_max_iou[i])[0])

        # print(batch_gt_max_iou_anchor_label)
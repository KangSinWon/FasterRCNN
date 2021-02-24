import torch
import torch.nn as nn
import numpy as np
import cv2

class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super(ProposalTargetLayer, self).__init__()

        self.n_sample = 128
        self.pos_ratio = 0.25

        self.pos_iou_thresh = 0.5

        self.neg_iou_thresh_hi = 0.5
        self.neg_iou_thresh_lo = 0.0

    def forward(self, roi, gt_boxes, labels, image):
        batch_size = roi.shape[0]

        # ious = np.empty((batch_size, roi.shape[1], roi.shape[2]), dtype=np.float32)
        # ious.fill(0)

        batch_gt_roi_label = []
        batch_gt_roi_bbox = []
        batch_sample_roi = []
        for i in range(batch_size):
            _ious = np.empty((roi.shape[1], gt_boxes[i].shape[0]))
            _ious.fill(0)
            for idx1, roi_bbox in enumerate(roi[i]):
                xa1, ya1, xa2, ya2 = roi_bbox
                anchor_area = (ya2 - ya1) * (xa2 - xa1)

                for idx2, gt_bbox in enumerate(gt_boxes[i]):
                    xb1, yb1, xb2, yb2 = gt_bbox

                    gt_area_box = (yb2 - yb1) * (xb2 - xb1)
                    inter_x1 = max([xb1, xa1])
                    inter_y1 = max([yb1, ya1])
                    inter_x2 = min([xb2, xa2])
                    inter_y2 = min([yb2, ya2])

                    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                        inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                        iou = inter_area / (anchor_area + gt_area_box - inter_area)
                    else:
                        iou = 0
                    _ious[idx1, idx2] = iou
        
            # print(_ious.shape)
            gt_assignment = _ious.argmax(axis=1)
            max_iou = _ious.max(axis=1)
            gt_roi_label = labels[i][gt_assignment]

            # POS
            pos_roi_per_image = self.n_sample * self.pos_ratio # 32
            pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
            pos_roi_per_image = int(min(pos_roi_per_image, pos_index.size))

            if pos_index.size > 0:
                pos_index = np.random.choice(
                    pos_index, size=pos_roi_per_image, replace=False
                )
            # print('pos_roi_per_image', pos_roi_per_image)

            # NEG
            neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
            neg_roi_per_image = self.n_sample - pos_roi_per_image
            neg_roi_per_image = int(min(neg_roi_per_image, neg_index.size))

            if neg_index.size > 0:
                neg_index = np.random.choice(
                    neg_index, size=neg_roi_per_image, replace=False
                )
            # print('neg_roi_per_image:', neg_roi_per_image)

            img = image[i].permute(1, 2, 0).numpy()
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

            for j in range(pos_roi_per_image):
                x1, y1, x2, y2 = roi[i][pos_index[j]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 2552, 255), thickness=3)

            for j in range(gt_boxes[i].shape[0]):
                x1, y1, x2, y2 = gt_boxes[i][j]
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 2552, 0), thickness=3)

            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            keep_index = np.append(pos_index, neg_index)

            gt_roi_labels = gt_roi_label[keep_index]
            gt_roi_labels[pos_roi_per_image:] = 0
            sample_roi = roi[i][keep_index]
            sample_roi = np.array(sample_roi)

            bbox_for_sample_roi = gt_boxes[i][gt_assignment[keep_index]]

            w = sample_roi[:, 2] - sample_roi[:, 0]
            h = sample_roi[:, 3] - sample_roi[:, 1]
            center_x = sample_roi[:, 0] + 0.5 * w
            center_y = sample_roi[:, 1] + 0.5 * h

            base_w = bbox_for_sample_roi[:, 2] - bbox_for_sample_roi[:, 0]
            base_h = bbox_for_sample_roi[:, 3] - bbox_for_sample_roi[:, 1]
            base_center_x = bbox_for_sample_roi[:, 0] + 0.5 * base_w
            base_center_y = bbox_for_sample_roi[:, 1] + 0.5 * base_h

            eps = np.finfo(np.float32).eps
            h = np.maximum(h, eps)
            w = np.maximum(w, eps)

            dx = (base_center_x - center_x) / w
            dy = (base_center_y - center_y) / h
            dw = np.log(base_w / w)
            dh = np.log(base_h / h)

            gt_roi_bbox = np.vstack((dx, dy, dw, dh)).transpose()

            batch_sample_roi.append(sample_roi)
            batch_gt_roi_label.append(gt_roi_labels)
            batch_gt_roi_bbox.append(gt_roi_bbox)

        return batch_sample_roi, batch_gt_roi_label, batch_gt_roi_bbox
import torch
import torch.nn as nn
import numpy as np
import cv2

from anchor import Anchor


class ProposalLayer(nn.Module):
    def __init__(self):
        super(ProposalLayer, self).__init__()

        self.NMS_thresh = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        
        self.image_size = (1333, 402)
        self.min_size = 16


    def forward(self, anchor, rpn_cls_score, rpn_bbox_pred, rpn_cls_loss, rpn_bbox_loss, image):
        batch_size = rpn_cls_score.shape[0]

        objectness_score = rpn_cls_score.view(batch_size, int(self.image_size[0] / self.min_size), int(self.image_size[1] / self.min_size), 9, 2)[:, :, :, :, 1].contiguous().view(batch_size, -1)

        objectness_score = objectness_score.data.numpy()
        # [4, 18675]
        # print(objectness_score.shape)

        anchor_height = anchor.anchor_boxes[:, 3] - anchor.anchor_boxes[:, 1]
        anchor_width = anchor.anchor_boxes[:, 2] - anchor.anchor_boxes[:, 0]
        anchor_center_y = anchor.anchor_boxes[:, 1] + 0.5 * anchor_height
        anchor_center_x = anchor.anchor_boxes[:, 0] + 0.5 * anchor_width

        pred_anchor_bbox = rpn_bbox_pred.data.numpy()

        dy = pred_anchor_bbox[:, :, 1::4]
        dx = pred_anchor_bbox[:, :, 0::4]
        dh = pred_anchor_bbox[:, :, 3::4]
        dw = pred_anchor_bbox[:, :, 2::4]

        center_y = dy * anchor_height[:, np.newaxis] + anchor_center_y[:, np.newaxis]
        center_x = dx * anchor_width[:, np.newaxis] + anchor_center_x[:, np.newaxis]
        h = np.exp(dh) * anchor_height[:, np.newaxis]
        w = np.exp(dw) * anchor_width[:, np.newaxis]

        # [x1, y1, x2, y2]
        roi = np.zeros(pred_anchor_bbox.shape, dtype=pred_anchor_bbox[0].dtype)
        roi[:, :, 0::4] = center_x - 0.5 * w
        roi[:, :, 1::4] = center_y - 0.5 * h
        roi[:, :, 2::4] = center_x + 0.5 * w
        roi[:, :, 3::4] = center_y + 0.5 * h

        roi[:, :, slice(0, 4, 2)] = np.clip(roi[:, :, slice(0, 4, 2)], 0, self.image_size[0])
        roi[:, :, slice(1, 4, 2)] = np.clip(roi[:, :, slice(1, 4, 2)], 0, self.image_size[1])

        # print(roi.shape, np.max(roi), np.min(roi))

        hs = roi[:, :, 3] - roi[:, :, 1]
        ws = roi[:, :, 2] - roi[:, :, 0]
        
        batch_roi = []
        for i in range(batch_size):
            keep = np.where((hs[i] >= self.min_size) & (ws[i] >= self.min_size))[0]

            _roi = roi[i, keep, :]

            score = objectness_score[i, keep]
            order = score.ravel().argsort()[::-1]

            order = order[:self.n_train_pre_nms]
            _roi = _roi[order, :]
            x1 = _roi[:, 0]
            y1 = _roi[:, 1]
            x2 = _roi[:, 2]
            y2 = _roi[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            # TEST
            keep = []
            for z in range(len(order) - 1):
                xx1 = np.maximum(x1[z], x1[z+1:])
                yy1 = np.maximum(y1[z], y1[z+1:])
                xx2 = np.minimum(x2[z], x2[z+1:])
                yy2 = np.minimum(y2[z], y2[z+1:])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                inter_area = w * h
                ovr = inter_area / (areas[z] + areas[z+1:] - inter_area)
                keep.append(np.where(ovr >= self.NMS_thresh)[0])

            inds = np.concatenate([i for i in keep if i.size > 0])
            inds = np.unique(inds)

            _roi = np.delete(_roi, inds, 0)
            _roi = _roi[:self.n_train_post_nms]
            batch_roi.append(_roi)

                    # print('inds:', inds.shape)

                    # img = image[i].permute(1, 2, 0).numpy()
                    # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                    # _x1, _y1, _x2, _y2 = _roi[z]
                    # cv2.circle(img, (_x1, _y1), 3, (0, 0, 0), 3)
                    # cv2.circle(img, (_x2, _y2), 3, (0, 255, 255), 3)
                    # cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 0, 255), 2)
                    # cv2.putText(img, text=('%.2f' % ovr), org=(_x1, _y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    # _x1, _y1, _x2, _y2 = _roi[y]
                    # cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (255, 0, 0), 2)
                    # cv2.circle(img, (_x1, _y1), 3, (0, 0, 0), 3)
                    # cv2.circle(img, (_x2, _y2), 3, (0, 255, 255), 3)
                    # cv2.rectangle(img, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)
                    # cv2.imshow('img', img)
                    # if cv2.waitKey(100) >= 0:
                        # break

        #     # NMS
        #     # order = order.argsort()[::-1]
        #     keep = []
        #     while order.size > 0:
        #         j = order[0]
        #         keep.append(j)

        #         img = image[i].permute(1, 2, 0).numpy()
        #         img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        #         _x1, _y1, _x2, _y2 = _roi[j]
        #         cv2.rectangle(img, (_x1, _y1), (_x2, _y2), (0, 0, 255), 2)
        #         print(score[j])
        #         cv2.putText(img, text=str(score[j]), org=(_x1, _y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        #         cv2.imshow('img', img)
        #         if cv2.waitKey(1000) >= 0:
        #             break
                
        #         print(np.max(order))
        #         xx1 = np.maximum(x1[j], x1[order[1:]])
        #         yy1 = np.maximum(y1[j], y1[order[1:]])
        #         xx2 = np.minimum(x2[j], x2[order[1:]])
        #         yy2 = np.minimum(y2[j], y2[order[1:]])

        #         w = np.maximum(0.0, xx2 - xx1 + 1)
        #         h = np.maximum(0.0, yy2 - yy1 + 1)

        #         inter_area = w * h
        #         ovr = inter_area / (areas[j] + areas[order[1:]] - inter_area)
        #         inds = np.where(ovr <= self.NMS_thresh)[0]
        #         order = order[inds + 1]
            
        #     keep = keep[:self.n_train_post_nms]
        #     _roi = _roi[keep]
            
        #     batch_roi.append(_roi)
        #     # [2000, 4]
        #     # print(len(keep), _roi.shape)

        # cv2.destroyAllWindows()
        return batch_roi

# img = image[i].permute(1, 2, 0).numpy()
# img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

# for j in range(pos_roi_per_image):
#     x1, y1, x2, y2 = roi[i][pos_index[j]]
#     cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 2552, 255), thickness=3)

# for j in range(gt_boxes[i].shape[0]):
#     x1, y1, x2, y2 = gt_boxes[i][j]
#     cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 2552, 0), thickness=3)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
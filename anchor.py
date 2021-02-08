import numpy as np

class Anchor:
    def __init__(self, width, height, stride):
        # TODO: how to load
        self.gt_boxes = []

        self.ratios = [0.5, 1, 2]
        self.scales = [8, 16, 32]
        
        self.stride = stride
        self.width = width
        self.height = height
        self.features_map_width = self.width // self.stride
        self.features_map_height = self.height // self.stride
        self.anchor_boxes = self.anchor_generator()
        self.inside_index = self.remove_outside_boxes()
        self.inside_anchor_boxes = self.anchor_boxes[self.inside_index]
        
    def anchor_generator(self):
        center_x = np.arange(self.stride, (self.features_map_width + 1) * self.stride, self.stride)
        center_y = np.arange(self.stride, (self.features_map_height + 1) * self.stride, self.stride)
        
        index = 0
        anchor_center = np.zeros((self.features_map_width * self.features_map_height, 2))
        for i in range(len(center_x)):
            for j in range(len(center_y)):
                anchor_center[index, 1] = center_x[i] - 8
                anchor_center[index, 0] = center_y[j] - 8
                index += 1
        
        index = 0
        anchor_boxes = np.zeros(((self.features_map_width * self.features_map_height * 9), 4))
        for c in anchor_center:
            center_x, center_y = c
            for i in range(len(self.ratios)):
                for j in range(len(self.scales)):
                    h = self.stride * self.scales[j] * np.sqrt(self.ratios[i])
                    w = self.stride * self.scales[j] * np.sqrt(1. / self.ratios[i])
                    
                    anchor_boxes[index, 1] = center_y - h / 2.
                    anchor_boxes[index, 0] = center_x - w / 2.
                    anchor_boxes[index, 3] = center_y + h / 2.
                    anchor_boxes[index, 2] = center_x + w / 2.
                    index += 1
        
        return anchor_boxes
    
    def remove_outside_boxes(self, anchor_boxes):
        index = np.where(
                    (anchor_boxes[:, 0] >= 0) &
                    (anchor_boxes[:, 1] >= 0) &
                    (anchor_boxes[:, 2] <= self.width) &
                    (anchor_boxes[:, 3] <= self.height))[0]

        return index

    def get_ious(self):
        # np.empty에 4는 해당 사진의 gt objects 개수
        # TODO: change 4 to gt_object num
        ious = np.empty((len(self.inside_anchor_boxes), 4), dtype=np.float32)
        ious.fill(0)

        for i, anchor_box in enumerate(self.inside_anchor_boxes):
            xa1, ya1, xa2, ya2 = anchor_box
            anchor_area = (xa2 - xa1) * (ya2 - ya1)

            for j, gt_box in enumerate(self.gt_boxes):
                xb1, yb1, xb2, yb2 = gt_box
                gt_area = (xb2 - xb1) * (yb2 - yb1)

                x1 = max([xb1, xa1])
                y1 = max([yb1, ya1])
                x2 = min([xb2, xa2])
                y2 = min([yb2, ya2])
                
                if (x1 < x2) and (y1 < y2):
                    area = (x2 - x1) * (y2 - y1)
                    iou = area / (anchor_area + gt_area - area)
                else:
                    iou = 0
                
                ious[i, j] = iou
        
        return ious
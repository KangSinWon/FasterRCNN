import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionProposalNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_anchor):
        super(RegionProposalNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_anchor = n_anchor
        
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1).to(DEVICE)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        
        self.bounding_box_regressor = nn.Conv2d(self.out_channels, self.n_anchor * 4, 1, 1, 0).to(DEVICE)
        self.bounding_box_regressor.weight.data.normal_(0, 0.01)
        self.bounding_box_regressor.bias.data.zero_()
        
        self.classifier = nn.Conv2d(self.out_channels, self.n_anchor * 2, 1, 1, 0).to(DEVICE)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv1(x.to(DEVICE))
        
        pred_regressor = self.bounding_box_regressor(x)
        pred_classifier = self.classifier(x)
        
        return pred_regressor, pred_classifier
    
    # TODO: Multi-task loss
    def multi_task_loss(rpn_box_reg, rpn_cls, gt_box_reg, gt_cls):
        rpn_cls_loss = F.cross_entropy(rpn_cls, gt_cls.long().to(DEVICE), ignore_index=-1)

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
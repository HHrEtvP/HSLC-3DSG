import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

# from maskrcnn_benchmark.layers import Label_Smoothing_Regression 暂时先不用这个
from my_utils.matcher import Matcher
from my_utils.misc import cat


class RelationLossComputation(object):

    def __init__(
        self,
        use_label_smoothing,
    ):
        self.use_label_smoothing = use_label_smoothing
        """
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
        """
        self.criterion_loss = nn.CrossEntropyLoss()

    def __call__(self, fg_labels, rel_labels, relation_logits, obj_logits, has_obj=True, has_rel=True):
        if has_obj:
            obj_logits = cat(obj_logits, dim=0)
            fg_labels = cat([fg.get_field("labels") for fg in fg_labels], dim=0)
            fg_labels = torch.tensor(fg_labels)
        if has_rel:
            relation_logits = cat(relation_logits, dim=0)
            rel_labels = cat(rel_labels, dim=0)

        loss_refine_obj = torch.tensor(0., requires_grad=True).cuda()
        loss_relation = torch.tensor(0., requires_grad=True).cuda()
        obj_cls_acc = 0
        rel_cls_acc = 0

        if has_obj:
            if obj_logits.shape[0] != 0:
                loss_refine_obj = self.criterion_loss(obj_logits, fg_labels.long().cuda())
                obj_cls_acc = (fg_labels == obj_logits.argmax(-1)).nonzero().shape[0]/fg_labels.shape[0]
        if has_rel:
            if relation_logits.shape[0] != 0:
                loss_relation = self.criterion_loss(relation_logits, rel_labels.long().cuda())
                rel_cls_acc = (rel_labels == relation_logits.argmax(-1)).nonzero().shape[0]/rel_labels.shape[0]

        return loss_relation, loss_refine_obj, rel_cls_acc, obj_cls_acc

    def generate_attributes_target(self, attributes):
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def make_roi_relation_loss_evaluator(cfg):
    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
    )
    return loss_evaluator

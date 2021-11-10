import os
import sys
import torch
import numpy as np


class BatchMetricProcessor:
    def __init__(self, K, P):
        self.K = K
        self.P = P
        self.pred = torch.zeros((1, 32), dtype=torch.int64)
        self.gt = torch.zeros((1, 4), dtype=torch.int64)
        self.batch_cnt = 0

    def reset(self):
        self.pred = torch.zeros((1, 32), dtype=torch.int64)
        self.gt = torch.zeros((1, 4), dtype=torch.int64)
        self.batch_cnt = 0

    def step(self, per_batch_pred, per_batch_gt):
        if per_batch_gt.shape[0] != 1:
            per_batch_gt.unsqueeze(0)
        if per_batch_pred.shape[0] != 1:
            per_batch_gt.unsqueeze(0)
        if self.batch_cnt == 0:
            self.pred = per_batch_pred
            self.gt = per_batch_gt
            self.batch_cnt = self.batch_cnt + 1
        else:
            self.pred = torch.cat([self.pred, per_batch_pred], dim=0)
            self.gt = torch.cat([self.gt, per_batch_gt], dim=0)
            self.batch_cnt = self.batch_cnt + 1

    def compute_obj_cls_accuracy(self, obj_logits, obj_label):
        num_correct = 0
        totoal_num_box = 0
        obj_preds = []
        for obj_logits_this_scan in obj_logits:
            obj_preds.append(obj_logits_this_scan[:, :].max(-1)[1])
        for (obj_pred_this_scan, obj_label_this_scan) in zip(obj_preds, obj_label):
            for o_pred, o_label in zip(obj_pred_this_scan, obj_label_this_scan):
                if int(o_pred) == int(o_label):
                    num_correct = num_correct + 1
                totoal_num_box = totoal_num_box + 1
        return num_correct/totoal_num_box, totoal_num_box

    def compute_obj_cls_recall(self, obj_logits, obj_label):
        obj_preds = []
        for obj_logits_this_scan in obj_logits:
            obj_preds.append(obj_logits_this_scan.max(-1)[1])
        TP = {}
        FN = {}
        recall = {}
        for i in np.arange(77):
            TP[i] = 0
            FN[i] = 0
            recall[i] = 0
        for (obj_pred_this_scan, obj_label_this_scan) in zip(obj_preds, obj_label):
            for o_pred, o_label in zip(obj_pred_this_scan, obj_label_this_scan):
                if int(o_pred) == int(o_label):
                    TP[int(o_label)] = TP[int(o_label)]
                else:
                    FN[int(o_label)] = FN[int(o_label)]
        for tpk in TP.keys():
            if FN[tpk] == 0:
                recall[tpk] = 1
                if TP[tpk] == 0:
                    recall[tpk] = 0
                continue
            recall[tpk] = (TP[tpk])/(TP[tpk]+FN[tpk])
        return recall

    def compute_topK_rel_accuracy_sgcls(self):
        _, idx = torch.sort(self.pred[:, 2:18], dim=-1, descending=True)
        topk_classes = idx[:, :self.K] + 1
        num_correct = 0
        num_fg = 0
        num_cls_error = 0
        for i, (p, g, topk) in enumerate(zip(self.pred, self.gt, topk_classes)):
            if not (int(p[0]) == g[0] and int(p[1]) == g[1]):
                print("mismatch!!!")
                continue
            if g[4] != 0:
                num_fg = num_fg + 1
                if g[4] in topk and int(p[70]) == int(g[5]):
                    head_pred = p[18:44].max(-1)[1]
                    tail_pred = p[44:70].max(-1)[1]
                    if head_pred == g[2] and tail_pred == g[3]:
                        num_correct = num_correct + 1
                    else:
                        num_cls_error = num_cls_error + 1
        return num_correct / num_fg, num_cls_error

    def compute_LowP_rel_accuracy_sgcls(self):
        lowP = torch.full_like(self.pred[0, 2:18], fill_value=self.P)
        _, idx = torch.sort(self.pred[:, 2:18], dim=-1, descending=True)
        num_correct = 0
        num_fg = 0
        for i, (p, g) in enumerate(zip(self.pred, self.gt)):
            if not (int(p[0]) == g[0] and int(p[1]) == g[1]):
                print("mismatch!!!")
                continue
            if g[4] != 0:
                num_fg = num_fg + 1
                lowP_cls = (p[2:18] > lowP).nonzero().squeeze(-1) + 1
                if g[4] in lowP_cls and int(p[70]) == int(g[5]):
                    head_pred = p[18:44].max(-1)[1]
                    tail_pred = p[44:70].max(-1)[1]
                    if head_pred == g[2] and tail_pred == g[3]:
                        num_correct = num_correct + 1
        return num_correct / num_fg


class RecallKProcessor:
    def __init__(self, K, P):
        self.K = K  # top-K
        self.P = P  # lowest-P
        self.pred = torch.zeros((1, 32), dtype=torch.int64)
        self.gt = torch.zeros((1, 4), dtype=torch.int64)
        self.scan_cnt = 0

    def reset(self):
        self.pred = torch.zeros((1, 32), dtype=torch.int64)
        self.gt = torch.zeros((1, 4), dtype=torch.int64)
        self.scan_cnt = 0

    def step(self, per_batch_pred, per_batch_gt, num_gt_per_scan, num_pred_per_scan):
        if per_batch_gt.shape[0] != 1:
            per_batch_gt.unsqueeze(0)
        if per_batch_pred.shape[0] != 1:
            per_batch_gt.unsqueeze(0)
        if self.scan_cnt == 0:
            self.pred = per_batch_pred.split(num_pred_per_scan, dim=0)
            self.gt = per_batch_gt.split(num_gt_per_scan, dim=0)
            self.scan_cnt = self.scan_cnt + 1
        else:
            self.pred = torch.cat([self.pred, per_batch_pred], dim=0)
            self.gt = torch.cat([self.gt, per_batch_gt], dim=0)
            self.scan_cnt = self.scan_cnt + 1

    def computeRecallK(self):
        RK_this_batch = []
        num_scan_tracked = 0
        for pred, gt in zip(self.pred, self.gt):
            if pred.shape[0] == 0 or gt.shape[0] == 0:
                continue
            pred_score, pred_idx = torch.max(pred[:, 2:18], dim=-1)
            triplet_score = pred_score
            _, perm = torch.sort(triplet_score, dim=0, descending=True)
            pred = pred[perm]
            gt = gt[perm]
            pred_idx = pred_idx[perm]
            num_correct = 0
            num_fg_rel = np.where(gt[:, 4] > 0)[0].shape[0]
            i = 0
            for p, idx, g in zip(pred, pred_idx, gt):
                if i >= self.K:
                    break
                assert (int(p[0]) == g[0] and int(p[1]) == g[1])
                if g[4] != 0:
                    if idx+1 == g[4]:
                        head_pred = p[18:44].max(-1)[1]
                        tail_pred = p[44:70].max(-1)[1]
                        if head_pred == g[2] and tail_pred == g[3]:
                            num_correct = num_correct + 1
                i = i + 1
            RK_this_batch.append(num_correct / num_fg_rel if num_fg_rel else 0)
            num_scan_tracked += 1
        RK_sum = 0
        for rk in RK_this_batch:
            RK_sum += rk
        return RK_sum/len(self.pred)


    def computeMeanRecallK(self):
        RK_this_batch = torch.zeros(16)
        num_scan_tracked = 0
        for pred, gt in zip(self.pred, self.gt):
            if pred.shape[0] == 0 or gt.shape[0] == 0:
                continue
            pred_score, pred_idx = torch.max(pred[:, 2:18], dim=-1)
            triplet_score = pred_score
            _, perm = torch.sort(triplet_score, dim=0, descending=True)
            pred = pred[perm]
            gt = gt[perm]
            pred_idx = pred_idx[perm]
            for k in range(16):
                num_correct = 0
                num_fg_rel = np.where(gt[:, 4] == k + 1)[0].shape[0]
                i = 0
                for p, idx, g in zip(pred, pred_idx, gt):
                    if i >= self.K:
                        break
                    assert (int(p[0]) == g[0] and int(p[1]) == g[1])
                    if g[4] != 0:
                        if idx + 1 == g[4] and g[4] == k + 1:
                            head_pred = p[18:44].max(-1)[1]
                            tail_pred = p[44:70].max(-1)[1]
                            if head_pred == g[2] and tail_pred == g[3]:
                                num_correct = num_correct + 1
                    i = i + 1
                RK_this_batch[k] += num_correct / num_fg_rel if num_fg_rel else 0
            num_scan_tracked += 1
        return (RK_this_batch/num_scan_tracked).sum(-1)/RK_this_batch.shape[-1]


def REL_parse_output(relation_logits, obj_logits, pair_idx):
    per_batch_pred = []
    num_obj = 26
    num_rel = 17
    total_dim = 2 + 2 * num_obj + num_rel - 1 + 1
    for i, (rel, obj, pair) in enumerate(zip(relation_logits, obj_logits, pair_idx)):
        per_scan_pred = []
        for r, p in zip(rel, pair):
            one_sample = torch.zeros(total_dim)
            one_sample[:2] = p
            one_sample[2:2+num_rel-1] = r[1:]
            one_sample[2+num_rel-1:2+num_rel-1+num_obj] = obj[p[0]]
            one_sample[2+num_rel-1+num_obj:2+num_rel-1+2 * num_obj] = obj[p[1]]
            one_sample[total_dim-1] = i
            per_scan_pred.append(one_sample)
        if per_scan_pred == []:
            continue
        per_scan_pred = torch.as_tensor([pred.detach().numpy() for pred in per_scan_pred])
        per_batch_pred.append(per_scan_pred)
    return torch.cat(per_batch_pred, dim=0)


def REL_parse_gt(relation_logits, obj_logits, pair_idx):
    per_batch_gt = []
    for i, (rel, obj, pair) in enumerate(zip(relation_logits, obj_logits, pair_idx)):
        per_scan_gt = []
        for r, p in zip(rel, pair):
            one_sample = torch.zeros(6, dtype=torch.int64)
            one_sample[:2] = p
            one_sample[2] = obj[p[0]]
            one_sample[3] = obj[p[1]]
            one_sample[4] = r
            one_sample[5] = i
            per_scan_gt.append(one_sample)
        if per_scan_gt == []:
            continue
        per_scan_gt = torch.as_tensor([gt.detach().numpy() for gt in per_scan_gt])
        per_batch_gt.append(per_scan_gt)
    return torch.cat(per_batch_gt, dim=0)




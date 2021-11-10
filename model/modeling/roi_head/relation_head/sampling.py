# -*- coding: UTF-8 -*-

import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from my_utils.boxlist_ops import boxlist_iou, boxlist_iou_tensor, boxlist_iou_tensor_faster
from my_utils.misc import cat


class RelationSampling(object):
    def __init__(
            self,
            fg_thres,
            require_overlap,
            num_sample_per_gt_rel,
            batch_size_per_image,
            positive_fraction,
            use_gt_box,
            test_overlap,
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.test_overlap = test_overlap

    def prepare_test_pairs(self, device, proposals, gt_rel_pair_idxs, gt_rel_labels):
        rel_labels = []
        rel_pair_idxs = []
        for i, p in enumerate(proposals):
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            labels = torch.zeros((idxs.shape[0]), dtype=torch.int64, device=device)
            for j, pair in enumerate(idxs):
                matched = (torch.sum(gt_rel_pair_idxs[i], dim=-1) == pair[0]+pair[1]).nonzero().squeeze(-1)
                if matched.shape[0] != 0:
                    for m in matched:
                        if gt_rel_pair_idxs[i][m][0] == pair[0]:
                            labels[j] = gt_rel_labels[i][m].item()
                            continue
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
                rel_labels.append(labels)
            else:
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
                rel_labels.append(torch.zeros(1, dtype=torch.int64, device=device))
        return rel_pair_idxs, rel_labels

    def gtbox_relsample(self, proposals, targets):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for scan_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = (target.get_field("rel_obj_id") >= 0).nonzero().shape[0]

            if num_prp == 1:
                rel_idx_pairs.append(torch.zeros((0, 2), dtype=torch.int64, device=device))
                rel_labels.append(torch.zeros(0, dtype=torch.int64, device=device))
                continue

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation")
            tgt_rel_matrix = torch.as_tensor(tgt_rel_matrix).cuda()
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(
                -1)

            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)

            if tgt_bg_idxs.shape[0] == 0:
                img_rel_idxs = tgt_pair_idxs
                img_rel_labels = tgt_rel_labs.contiguous().view(-1)
                rel_idx_pairs.append(img_rel_idxs)
                rel_labels.append(img_rel_labels)
                continue

            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs),
                                     dim=0)

            '''
            re-align pair indexs
            '''
            obj_id = target.get_field("obj_ids")
            obj_ids = target.get_field("rel_obj_id")
            mat_idx = target.get_field("rel_mat_idx")
            obj_map = {}
            for id, idx in zip(obj_ids, mat_idx):
                obj_map[id.item()] = idx.item()
            img_rel_idxs = get_correct_idxs(false_pair_idx=img_rel_idxs, obj_id_this_scan=obj_id,
                                            obj_map_this_scan=obj_map)


            try:
                img_rel_labels = torch.cat(
                    (tgt_rel_labs, torch.zeros((tgt_bg_idxs.shape[0]), device=device, dtype=torch.int64)),
                    dim=0).view(-1)
            except:
                img_rel_labels = torch.zeros(self.batch_size_per_image, device=device, dtype=torch.int64).view(-1)

            problem_idx = (img_rel_idxs.sum(dim=-1) == -2).nonzero().squeeze(-1)
            if img_rel_labels.shape[0] != img_rel_idxs.shape[0] or problem_idx.shape[0] > 0:
                img_rel_labels[problem_idx] = -1
                masked_idx = (img_rel_labels >= 0).nonzero().squeeze(-1)
                img_rel_labels = img_rel_labels[masked_idx]
                img_rel_idxs = img_rel_idxs[masked_idx]
            assert(img_rel_labels.shape[0] == img_rel_idxs.shape[0])

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    # SGDet
    def detect_relsample(self, proposals, targets, ious):

        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]
            prp_lab = proposal.get_field("pred_cls").long()
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation")
            tgt_rel_matrix = tgt_rel_matrix
            # ious2d = ious2d.cuda()
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (
                    ious[img_id] > self.fg_thres)  # [tgt, prp] ，prp_lab[None]=prp_lab[None,:]
            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            # rel_possibility[prp_lab == 0] = 0
            # rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious[img_id], is_match,
                                                                         rel_possibility, target)

            obj_id_this_scan = torch.zeros(proposal.bbox.shape[0])
            for i, prp_state in enumerate(is_match.transpose(1, 0)):
                matched_gt = prp_state.nonzero().squeeze(-1)
                if len(matched_gt) == 0:
                    obj_id_this_scan[i] = -1
                else:
                    obj_id_this_scan[i] = target.get_field("obj_ids")[matched_gt[0]]
            proposal.add_field("obj_ids", obj_id_this_scan)

            rel_idx_pairs.append(img_rel_triplets[:, 0: 2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility, target):

        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)  # (num_tgt_rels,2)
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        obj_id = target.get_field("obj_ids")
        obj_ids = target.get_field("rel_obj_id")
        mat_idx = target.get_field("rel_mat_idx")
        obj_map = {}
        for id, idx in zip(obj_ids, mat_idx):
            obj_map[id.item()] = idx.item()
        tgt_pair_idxs = get_correct_idxs(false_pair_idx=tgt_pair_idxs, obj_id_this_scan=obj_id,
                                         obj_map_this_scan=obj_map)

        problem_idx = (tgt_pair_idxs.sum(dim=-1) == -2).nonzero().squeeze(-1)
        if tgt_pair_idxs.shape[0] != tgt_rel_labs.shape[0] or problem_idx.shape[0] > 0:
            tgt_rel_labs[problem_idx] = -1
            masked_idx = (tgt_rel_labs >= 0).nonzero().squeeze(-1)
            tgt_rel_labs = tgt_rel_labs[masked_idx]
            tgt_pair_idxs = tgt_pair_idxs[masked_idx]

        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]

        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        no_fg = False
        no_bg = False

        fg_rel_triplets = []
        exclude = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)  #
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()

                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])

            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue

            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)

            # remove self-pair
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue

            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0

            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,1)
            fg_rel_i = cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(torch.int64)

            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                for rel in fg_rel_i:
                    if rel[:2].tolist() not in exclude:
                        exclude.append(rel[:2].tolist())
                        fg_rel_triplets.append(rel.unsqueeze(0))
                    else:
                        continue

        if len(fg_rel_triplets) > 0:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            # fg_rel_triplets = torch.as_tensor(fg_rel_triplets).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]
        else:
            fg_rel_triplets = torch.as_tensor(fg_rel_triplets)
            no_fg = True

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        if bg_rel_inds.shape[0] > 0:
            bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
            bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)

            num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0],
                                  bg_rel_triplets.shape[0])
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.as_tensor([])
            no_bg = True

        # if both fg and bg is none
        if no_fg and no_bg:
            placeholder_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
            return placeholder_rel_triplets, binary_rel
        elif no_fg:
            return torch.cat([bg_rel_triplets], dim=0), binary_rel
        elif no_bg:
            return torch.cat([fg_rel_triplets], dim=0), binary_rel
        else:
            return torch.cat([fg_rel_triplets, bg_rel_triplets], dim=0), binary_rel


def make_roi_relation_samp_processor(cfg):
    samp_processor = RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
    )
    return samp_processor


def get_correct_idxs(false_pair_idx, obj_id_this_scan, obj_map_this_scan):
    '''
    helper function for correcting 'un-aligned' idxs
    :param false_pair_idx:fake idx from relation matrix, [this index DOESN'T work on boxlist(proposal or gt)]
    :param obj_id_this_scan:object ID from a scan (a single Box3dList)
    :param obj_map_this_scan:object ID to relation matrix index(pair_idx)
    :return:true_pair_idx
    '''
    true_obj_index_in_mat = []
    for i in obj_id_this_scan:
        try:
            true_obj_index_in_mat.append(obj_map_this_scan[int(i.item())])
        except:
            true_obj_index_in_mat.append(-1)
    true_pair_idx = []
    for j, pair in enumerate(false_pair_idx):
        current_pair = torch.zeros(2, dtype=false_pair_idx.dtype, device=false_pair_idx.device)
        '''
        true_pair_idx[j, 0] = true_obj_index_in_mat.index(pair[0].item())
        true_pair_idx[j, 1] = true_obj_index_in_mat.index(pair[1].item())
        '''
        try:
            current_pair[0] = true_obj_index_in_mat.index(pair[0].item())
            current_pair[1] = true_obj_index_in_mat.index(pair[1].item())
        except:
            current_pair[0] = -1
            current_pair[1] = -1
        true_pair_idx.append(current_pair.unsqueeze(0))
    if true_pair_idx == [] or len(true_pair_idx) == 1 and true_pair_idx[0].sum(-1) == -2:
        # true_pair_idx = torch.zeros(2, dtype=false_pair_idx.dtype, device=false_pair_idx.device).unsqueeze(0)
        true_pair_idx = torch.zeros((0, 2), dtype=torch.int64, device=torch.device('cuda:0'))
    else:
        true_pair_idx = torch.cat(true_pair_idx, dim=0)
    return true_pair_idx

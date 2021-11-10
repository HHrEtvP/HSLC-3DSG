from torch.nn import functional as F
from my_utils.matcher import Matcher
from my_utils.boxlist_ops import boxlist_iou, boxlist_iou_tensor, boxlist_iou_tensor_faster
from my_utils.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
import torch
from model.structure.box3d_list import Box3dList
from model.dataset.rscan_utils import extract_pc_in_box3d
import numpy as np

class MySampling(object):
    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler=None,
            batch_size=8
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.batch_size = batch_size


    def assign_label_to_proposals(self, proposals, targets):
        """
        给proposal中的有匹配的候选框赋予标签
        """
        ious = []
        new_proposals = []
        for scan_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou_tensor(target.bbox, proposal.bbox)[0]
            matched_idxs = self.proposal_matcher(match_quality_matrix)  # matched_idx是每个prediction被分配的GT

            matched_labels = target.get_label_with_negative_val(matched_idxs)  # 根据idx去找对应的GT
            keep_idx = (matched_labels >= 0).nonzero().squeeze(-1)  # keep_idx标记哪些候选框保留，哪些删去，删去的候选框的matched_labels全部小于0，代表欠匹配
            matched_labels = matched_labels[keep_idx]

            pred_cls = proposal.get_field("pred_cls")
            pred_obj_score = proposal.get_field("pred_obj_score")
            pred_obj_centers = proposal.get_field("obj_centers")
            if keep_idx.shape[0] == 0:
                # 没找到对应的
                placeholder_size = torch.zeros((1, 3), device=torch.device('cuda:0'))
                placeHolder_label = torch.zeros(1, dtype=torch.int64, device=torch.device('cuda:0'))
                placeholder_corners = torch.zeros((1, 8, 3), device=torch.device('cuda:0'))
                new_proposal = Box3dList(size=placeholder_size, label=placeHolder_label, corners=placeholder_corners, is_empty=True)
                new_proposal.add_field("pred_cls", torch.tensor([0], dtype=torch.int32, device=torch.device('cuda:0')))
                new_proposal.add_field("pred_obj_score", torch.tensor([1.0], device=torch.device('cuda:0')))
                new_proposal.add_field("obj_centers", torch.zeros((1,3), device=torch.device('cuda:0')))
            else:
                # 直接删去欠匹配的候选框，只保留有对应的候选框
                new_proposal = Box3dList(size=proposal.size[keep_idx], label=matched_labels, corners=proposal.bbox[keep_idx])
                new_proposal.add_field("pred_cls", pred_cls[keep_idx])
                new_proposal.add_field("pred_obj_score", pred_obj_score[keep_idx])
                new_proposal.add_field("obj_centers", pred_obj_centers[keep_idx])
            new_proposals.append(new_proposal)
            ious.append(match_quality_matrix[:, keep_idx])
        return new_proposals, ious

    def prep_bbox_batch(self, proposals):
        real_box = []
        for proposal in proposals:
            real_box.append(proposal.bbox)
        cat_box = torch.cat(real_box, dim=0).double()  # (len(proposals)*每个boxlist中bbox个数, 8, 3)
        idxs = torch.cat([torch.full(size=(len(bbox),), fill_value=i, dtype=torch.int32) for i, bbox in enumerate(real_box)], dim=0).cuda()
        # (len(proposals)*每个boxlist中bbox个数, )，类似(0,0,0,0,1,1,1,2,2,......)
        return cat_box, idxs

    def prep_box_head_batch(self, pc, boxlist):
        box_batch, box_idx = self.prep_bbox_batch(boxlist)
        pc_batch = []
        for j, (b, i) in enumerate(zip(box_batch, box_idx)):
            if i == -1 or b.nonzero().shape[0] == 0:
                pc_per_box = torch.zeros((512, 3), dtype=torch.float64).cuda()
                pc_batch.append(pc_per_box)
                continue
            pc_per_box, _ = extract_pc_in_box3d(pc.xyz[i].detach().cpu().numpy(), b.detach().cpu().numpy())
            pc_per_box = torch.from_numpy(pc_per_box).cuda()
            if pc_per_box.shape[0] == 0:  # possible error(likely on dataset itself)
                keep = np.random.choice(a=np.arange(8), size=512, replace=True)
                pc_batch.append(b[keep].cuda())
                continue
            else:
                keep = np.random.choice(a=pc_per_box.shape[0], size=512, replace=pc_per_box.shape[0] < 512)
            pc_per_box = pc_per_box[keep]
            pc_batch.append(pc_per_box)
        pc_batch = torch.cat([p.double().unsqueeze(0) for p in pc_batch], dim=0)
        pc_batch = pc_batch.view((-1, 512, 3))
        return pc_batch, box_batch, box_idx


def make_roi_box_samp_processor(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    samp_processor = MySampling(
        matcher,
        fg_bg_sampler,
    )

    return samp_processor


if __name__ == "__main__":
    test_box = torch.tensor([[0,0,0],[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1],[0,1,1],[1,0,1]]).double()
    test_pc = torch.zeros((20000, 3))
    test_pc[10000:,:] += 2
    pc_per_box, _ = extract_pc_in_box3d(test_pc.numpy(), test_box.numpy())
    pass

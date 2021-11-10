import torch
from torch import nn

from model.structure.pc_list import PcList
from model.structure.box3d_list import Box3dList

from model.modeling.roi_head.box_head.box_head import build_box_head
from model.modeling.roi_head.relation_head.relation_head import build_pretrain_relation_head


class BoxRoiHead(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg.clone()

        self.mode = mode

        self.relation_head = build_pretrain_relation_head(cfg, self.mode)
        self.box_head = build_box_head(cfg, self.mode)

    def forward(self, pc, rel_mat=None, rel_obj_id=None, rel_mat_idx=None, output=None, GT=None, config=None):
        boxlists = []
        gtlists = []
        pclist = PcList(xyz_tensor=pc)
        if self.mode == "predcls":
            boxlists, pclist, gtlists = self.box_head.parse(pc=pc, rel_mat=rel_mat,
                                                            rel_obj_id=rel_obj_id, rel_mat_idx=rel_mat_idx, GT=GT)
            ious = None
        if self.mode == "sgcls":
            boxlists, pclist, gtlists = self.box_head.parse(pc=pc, rel_mat=rel_mat,
                                                            rel_obj_id=rel_obj_id, rel_mat_idx=rel_mat_idx, GT=GT)
            ious = None
        if self.mode == "sgdet":
            boxlists, pclist, gtlists, ious = self.box_head.parse(pc=pc, rel_mat=rel_mat,
                                                                  rel_obj_id=rel_obj_id, rel_mat_idx=rel_mat_idx,
                                                                  output=output,
                                                                  GT=GT)

        end_dict = self.relation_head(boxlists, pclist, gtlists, ious)

        return end_dict


def build_box_roi_head(cfg, mode):
    return BoxRoiHead(cfg, mode)

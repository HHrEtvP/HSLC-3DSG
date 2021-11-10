import os
import sys
import torch
import numpy as np
from torch import nn
from model.config import cfg

from torch.utils.data import DataLoader
from my_utils.my_collate_fn import my_collate_fn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # "...\Experiment-PredCLS"

from model.modeling.detector.models.votenet import build_votenet
from model.modeling.roi_head.box_roi_head import build_box_roi_head
from model.dataset.rscan_detection_dataset import RScanDetectionVotesDataset

from model.dataset.model_util_rscan import RScanDatasetConfig as DC


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.detector = build_votenet(in_dim=cfg.MODEL.ROI_BOX_HEAD.IN_FEATURE_DIM)
        self.box_roi_head = build_box_roi_head(cfg, self.mode)
        self.parse_config = {'remove_empty_box': False, 'use_3d_nms': False,
                             'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': False,
                             'per_class_proposal': True, 'conf_thresh': 0.05,
                             'dataset_config': DC()}

    def forward(self, inp, rel_mat, rel_obj_id, rel_mat_idx):
        if self.mode == "sgdet":
            end_points = self.detector(inp)
            end_dict = self.box_roi_head(
                pc=inp["point_clouds"],
                rel_mat=rel_mat,
                rel_obj_id=rel_obj_id,
                rel_mat_idx=rel_mat_idx,
                output=end_points,
                GT=inp, config=self.parse_config)
        if self.mode == "predcls":
            end_dict = self.box_roi_head(
                pc=inp["point_clouds"],
                rel_mat=rel_mat,
                rel_obj_id=rel_obj_id,
                rel_mat_idx=rel_mat_idx,
                GT=inp, config=self.parse_config)
        if self.mode == "sgcls":
            end_dict = self.box_roi_head(
                pc=inp["point_clouds"],
                rel_mat=rel_mat,
                rel_obj_id=rel_obj_id,
                rel_mat_idx=rel_mat_idx,
                GT=inp, config=self.parse_config)

        return end_dict


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)




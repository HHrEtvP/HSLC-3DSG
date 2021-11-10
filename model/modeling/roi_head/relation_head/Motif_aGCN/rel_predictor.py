import torch
from torch import nn
from torch.nn import functional as F

from my_utils.misc import cat
from model.modeling.roi_head.relation_head.Motif_aGCN.model_agcn import LSTMContext, FrequencyBias
from model.modeling.roi_head.relation_head.utils_relation import layer_init
from my_utils.misc import get_3rscan_statics


class MotifAGCNPredictor(nn.Module):
    def __init__(self, config, obj_feature_channels):
        super(MotifAGCNPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert obj_feature_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        statistics = get_3rscan_statics(cfg=config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.context_layer = LSTMContext(config, obj_classes, rel_classes, obj_feature_channels)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.rel_compress = nn.Linear(self.hidden_dim, self.num_rel_cls, bias=True)
        # self.obj_compress = nn.Linear(self.hidden_dim, self.num_obj_cls, bias=True)

        layer_init(self.rel_compress, xavier=True)

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, rel_dists, obj_id_this_batch = self.context_layer(roi_features, proposals, rel_pair_idxs)
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # obj_dists = self.obj_compress(obj_dists)
        # rel_dists = self.rel_compress(rel_dists)

        obj_id_this_batch = obj_id_this_batch.split(num_objs, dim=0)
        obj_dists = list(obj_dists.split(num_objs, dim=0))
        rel_dists = list(rel_dists.split(num_rels, dim=0))

        add_losses = {}
        return obj_dists, rel_dists, add_losses, rel_pair_idxs, obj_id_this_batch

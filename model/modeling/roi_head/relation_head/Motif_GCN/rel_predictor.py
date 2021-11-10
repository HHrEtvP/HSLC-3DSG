import torch
from torch import nn
from torch.nn import functional as F

from my_utils.misc import cat
from model.modeling.roi_head.relation_head.Motif_GCN.model_gcn import LSTMContext, FrequencyBias
from model.modeling.roi_head.relation_head.utils_relation import layer_init
from my_utils.misc import get_3rscan_statics


class MotifGCNPredictor(nn.Module):
    def __init__(self, config, obj_feature_channels):
        super(MotifGCNPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
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

        layer_init(self.rel_compress, xavier=True)

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, rel_pair_idxs)
        else:
            obj_dists, obj_preds, edge_ctx, obj_id_this_batch = self.context_layer(roi_features, proposals, rel_pair_idxs)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_id_this_batch = obj_id_this_batch.split(num_objs, dim=0)
        obj_dists = F.softmax(obj_dists, dim=-1)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists = self.rel_compress(edge_ctx)
        rel_dists = F.softmax(rel_dists, dim=-1)

        real_rel_dists = []
        for i, scan_rel_num in enumerate(num_rels):
            scan_rel_dists = rel_dists[i][:scan_rel_num]
            real_rel_dists.append(scan_rel_dists)
        rel_dists = torch.cat(real_rel_dists, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses, rel_pair_idxs, obj_id_this_batch

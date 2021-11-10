import torch
from torch import nn
from torch.nn import functional as F

from my_utils.misc import cat
from model.modeling.roi_head.relation_head.Motif.model_motifs import LSTMContext, FrequencyBias
from model.modeling.roi_head.relation_head.utils_relation import layer_init
from my_utils.misc import get_3rscan_statics


class MotifPredictor(nn.Module):
    def __init__(self, config, obj_feature_channels):
        super(MotifPredictor, self).__init__()
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
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        self.up_dim = nn.Linear(2048, self.pooling_dim)
        layer_init(self.up_dim, xavier=True)

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        obj_dists, obj_preds, edge_ctx, obj_id_this_batch = self.context_layer(roi_features, proposals, logger)

        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_id_this_batch = obj_id_this_batch.split(num_objs, dim=0)

        '''
        rel_obj_map = []
        for proposal in proposals:
            rel_obj_map.append(proposal.get_field("rel_obj_idx_map"))
        '''

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                            obj_preds,
                                                                            ):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0].long()], tail_rep[pair_idx[:, 1].long()]),
                                       dim=-1))
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0].long()], obj_pred[pair_idx[:, 1].long()]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        prod_rep = self.post_cat(prod_rep)

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        add_losses = {}

        return obj_dists, rel_dists, add_losses, rel_pair_idxs, obj_id_this_batch

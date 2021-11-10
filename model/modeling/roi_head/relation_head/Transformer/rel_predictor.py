import torch
from torch import nn
from torch.nn import functional as F

from my_utils.misc import cat
from model.modeling.roi_head.relation_head.Transformer.feature_refinement import ObjContext, FrequencyBias
from model.modeling.roi_head.relation_head.Transformer.feature_refinement_lstm import LSTMContext
from model.modeling.roi_head.relation_head.Transformer.support_detector import support_sampler, impaired_mlp_detector
from model.modeling.roi_head.relation_head.Transformer.proxi_detector import proxi_sampler, impaired_gcn_detector
from model.modeling.roi_head.relation_head.utils_relation import layer_init
from model.modeling.roi_head.relation_head.Motif.utils_motifs import to_onehot
from my_utils.misc import get_3rscan_statics


class PretrainPredictor(nn.Module):
    def __init__(self, config, obj_feature_channels):
        super(PretrainPredictor, self).__init__()
        self.cfg = config

        # DATASET
        statistics = get_3rscan_statics(cfg=config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # Other Configs
        assert obj_feature_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # Mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # Object Feature Refinement for Support Relationship
        self.context_layer1 = LSTMContext(config, obj_classes, rel_classes, obj_feature_channels)

        # Support Relation Detector
        self.support_rel_idx_in_dataset = config.DATASETS.RSCAN_SUPPORT_REL_CLASSES_IDX
        self.support_detector = support_sampler(obj_dim=512, rel_dim=512,
                                                num_rel_cls=len(self.support_rel_idx_in_dataset) + 1, num_rel_sample=32,
                                                support_rel=self.support_rel_idx_in_dataset)

        # Proximity Relation Detector
        self.proxi_rel_idx_in_dataset = config.DATASETS.RSCAN_PROXI_REL_CLASSES_IDX
        self.proximity_detector = proxi_sampler(obj_dim=128, rel_dim=128,
                                                num_rel_cls=len(self.proxi_rel_idx_in_dataset) + 1, num_rel_sample=128,
                                                proxi_rel=self.proxi_rel_idx_in_dataset,
                                                mode=self.mode, use_rel_node=True, pos_fuse=True)

        # MISC
        self.obj_proj1 = nn.Linear(1024, 256)
        self.obj_proj2 = nn.Linear(512, 128)
        self.obj_compress = nn.Linear(456, self.num_obj_cls, bias=True)

        # Initialization
        layer_init(self.obj_compress, xavier=True)
        layer_init(self.obj_proj1, xavier=True)
        layer_init(self.obj_proj2, xavier=True)

        # Frequency Bias
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        # Ablation Study:
        self.ablation_study = self.cfg.MODEL.ROI_RELATION_HEAD.ABLATION_STUDY
        self.no_lstm = self.cfg.MODEL.ROI_RELATION_HEAD.NO_LSTM
        self.no_supp = self.cfg.MODEL.ROI_RELATION_HEAD.NO_SUPP
        self.no_prox = self.cfg.MODEL.ROI_RELATION_HEAD.NO_PROX
        if self.ablation_study:
            if self.no_lstm:
                self.obj_proj_no_lstm = nn.Linear(256, 512)
            if self.no_supp:
                self.impaired_no_supp_detector = impaired_gcn_detector(obj_dim=512, rel_dim=512,
                                                num_rel_cls=16+1, num_rel_sample=32)
            if self.no_prox:
                self.impaired_no_prox_detector = impaired_mlp_detector(obj_dim=512, rel_dim=512,
                                                num_rel_cls=16+1, num_rel_sample=32)

    def forward(self, proposals, obj_features, point_features, point_pos, rel_pair_idxs, rel_labels, union_features,
                logger=None):
        num_real_objs = [p.bbox.shape[0] for p in proposals]
        obj_id_this_batch = []
        for proposal in proposals:
            obj_id_this_batch.append(proposal.get_field("obj_ids"))
        obj_id_this_batch = cat(obj_id_this_batch, dim=0)

        # feature downsampling 1
        obj_features = self.obj_proj1(obj_features)

        # feature refinement using LSTM
        if not self.no_lstm:
            obj_feats, obj_dists = self.context_layer1(obj_features, proposals)
            obj_feats = list(obj_feats.split(num_real_objs, dim=0)) # 512 dim
            obj_dists = list(obj_dists.split(num_real_objs, dim=0))
        else:
            obj_feats = obj_features
            obj_feats = self.obj_proj_no_lstm(obj_feats)
            obj_feats = list(obj_feats.split(num_real_objs, dim=0))
            if self.mode == "predcls":
                obj_dists = []
                for p in proposals:
                    obj_dists.append(to_onehot(p.get_field("labels"), self.num_obj_cls))
            else:
                obj_dists = [p.get_field("pred_cls") for p in proposals]


        # Get floor
        label_dists = [p.get_field('labels') for p in proposals]
        floor_idx = self.get_floor_idx(label_dists, use_gt=(self.mode == "predcls"))

        if not self.ablation_study:
            # Support
            support_rel_dist, support_rel_labels, supportor_batch, supported_batch, support_map = self.support_detector(
                obj_feats,
                rel_pair_idxs,
                rel_labels,
                floor_idx)

            # Proximity
            proxi_obj_feats = torch.cat(obj_feats, dim=0)
            proxi_obj_feats = self.obj_proj2(proxi_obj_feats)  # feature downsampling 2
            proxi_obj_feats = list(proxi_obj_feats.split(num_real_objs, dim=0))
            proxi_rel_dist, proxi_rel_labels, proxi_pairs, proxi_map = self.proximity_detector(proposals, supportor_batch,
                                                                                               supported_batch,
                                                                                               proxi_obj_feats, rel_labels,
                                                                                               rel_pair_idxs)

            # Some post-processing
            proxi_labels = []
            support_pairs = self.assemble_support_pairs(supportor_batch, supported_batch)
            if not self.training:
                # 在inference时，Support和Proximity都不会立刻给出pairs对应的labels，所以要在这里组装
                support_labels = self.assemble_eval_labels(support_pairs, rel_pair_idxs, rel_labels, support_map)
                proxi_labels = self.assemble_eval_labels(proxi_pairs, rel_pair_idxs, rel_labels, proxi_map)

            if self.use_bias:
                # rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
                pass

            if self.training:
                return obj_dists, support_pairs, support_rel_dist, \
                       support_rel_labels, proxi_pairs, proxi_rel_dist, \
                       proxi_rel_labels, obj_id_this_batch
            else:
                return obj_dists, support_rel_dist, proxi_rel_dist, \
                       support_pairs, proxi_pairs, support_labels, \
                       proxi_labels, obj_id_this_batch
        elif self.no_supp:
            rel_dists = self.impaired_no_supp_detector(obj_feats, rel_pair_idxs)

            proxi_rel_dist, support_rel_dist = [], []
            proxi_labels, support_labels = [], []
            proxi_pairs, support_pairs = [], []
            proxi_idxs, support_idxs = [], []
            for i, labels in enumerate(rel_labels):
                proxi_idx, support_idx = [], []
                for j, label in enumerate(labels):
                    if label in self.proxi_rel_idx_in_dataset:
                        proxi_idx.append(j)
                    elif label in self.support_rel_idx_in_dataset:
                        support_idx.append(j)
                proxi_pairs.append(rel_pair_idxs[i][proxi_idx])
                support_pairs.append(rel_pair_idxs[i][support_idx])
                proxi_rel_dist.append(rel_dists[i][proxi_idx])
                support_rel_dist.append(rel_dists[i][support_idx])
                proxi_labels.append(rel_labels[i][proxi_idx])
                support_labels.append(rel_labels[i][support_idx])
                proxi_idxs.append(proxi_idx)
                support_idxs.append(support_idx)

            return obj_dists, support_pairs, support_rel_dist, \
                   support_labels, proxi_pairs, proxi_rel_dist, \
                   proxi_labels, obj_id_this_batch
        elif self.no_prox:
            rel_dists = self.impaired_no_prox_detector(obj_feats, rel_pair_idxs)

            # separate support and proximity relation
            proxi_rel_dist, support_rel_dist = [], []
            proxi_labels, support_labels = [], []
            proxi_pairs, support_pairs = [], []
            proxi_idxs, support_idxs = [], []
            for i, labels in enumerate(rel_labels):
                proxi_idx, support_idx = [], []
                for j, label in enumerate(labels):
                    if label in self.proxi_rel_idx_in_dataset:
                        proxi_idx.append(j)
                    elif label in self.support_rel_idx_in_dataset:
                        support_idx.append(j)
                proxi_pairs.append(rel_pair_idxs[i][proxi_idx])
                support_pairs.append(rel_pair_idxs[i][support_idx])
                proxi_rel_dist.append(rel_dists[i][proxi_idx])
                support_rel_dist.append(rel_dists[i][support_idx])
                proxi_labels.append(rel_labels[i][proxi_idx])
                support_labels.append(rel_labels[i][support_idx])
                proxi_idxs.append(proxi_idx)
                support_idxs.append(support_idx)

            return obj_dists, support_pairs, support_rel_dist, \
                   support_labels, proxi_pairs, proxi_rel_dist, \
                   proxi_labels, obj_id_this_batch

    def assemble_support_pairs(self, supportor, supported):
        batched_pairs = []
        for i, (supportor_i, supported_i) in enumerate(zip(supportor, supported)):
            pair = []
            for sub, obj in zip(supported_i, supportor_i):
                for k in sub:
                    pair.append(torch.tensor((k, obj), dtype=torch.int64, device=torch.device('cuda:0')).unsqueeze(0))
            if not pair:
                pair = torch.as_tensor(pair, dtype=torch.int64).cuda()
            else:
                pair = torch.cat(pair, dim=0)
            batched_pairs.append(pair)
        return batched_pairs

    def assemble_eval_labels(self, batched_pairs, rel_pair_idxs, rel_labels, map):
        def find_matched_pair(p, p0):
            idx = -1
            inter = p0.sum(-1).cuda()
            cand_idx = (p.sum(-1) == inter).nonzero().squeeze(-1)
            for i in cand_idx:
                if p0[i][0].item() == p[0].item() and p0[i][1].item() == p[1].item():
                    if labels[i].item() in map.keys():
                        idx = i
                        break
            return idx

        batched_labels = []
        for pair2sample, true_pair, labels in zip(batched_pairs, rel_pair_idxs, rel_labels):
            label2sample = []
            for p in pair2sample:
                idx = find_matched_pair(p, true_pair)
                if idx == -1:
                    label2sample.append(torch.zeros(1, dtype=torch.int64, device=torch.device('cuda:0')))
                else:
                    label2sample.append(labels[idx].long().unsqueeze(0))
            if label2sample == []:
                label2sample = torch.as_tensor(label2sample, dtype=torch.int64).cuda()
            else:
                label2sample = torch.cat(label2sample)
            for i, l in enumerate(label2sample):
                label2sample[i] = map[l.item()]
            batched_labels.append(label2sample)
        return batched_labels

    def get_floor_idx(self, obj_dists, use_gt=False):
        floor_idxs = []
        for dist in obj_dists:
            if not use_gt:
                obj_preds = dist
            else:
                obj_preds = dist
            try:
                idx = (obj_preds == 1).nonzero()[0].item()  # 只需要一个floor
            except:
                idx = 0  # 如果找不到floor，就用0代替
                print("==================NO FLOOR===================")
            floor_idxs.append(idx)
        return floor_idxs

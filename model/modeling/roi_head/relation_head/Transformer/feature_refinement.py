import copy

import numpy as np
import torch
import os
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from my_utils.misc import cat, line2space
from.transformer_with_cross_attn import TransformerEncoder
from.transformer import TransformerStack
from model.modeling.roi_head.relation_head.Motif.utils_motifs import obj_edge_vectors


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel, num_pos_feats, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(num_pos_feats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1)
        pass

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        xyz = self.conv1(xyz)
        xyz = self.bn1(xyz)
        xyz = self.relu(xyz)
        position_embedding = self.conv2(xyz)
        return position_embedding


class ObjContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, obj_feauture_channels):
        super(ObjContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        # obj_embed_vecs:(num_obj_classes,embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.pos_embed_learned1 = PositionEmbeddingLearned(input_channel=3, num_pos_feats=456)
        self.pos_embed_learned2 = PositionEmbeddingLearned(input_channel=3, num_pos_feats=456)

        self.obj_proj1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.obj_proj2 = nn.Conv1d(512, 256, kernel_size=1)

        d_model = 456
        nhead = 8
        dim_feedforward = 1024
        dropout = 0.1
        activation = "relu"
        selfposembed = clones(self.pos_embed_learned1, 6)
        crossposembed = clones(self.pos_embed_learned2, 6)

        if self.mode == "sgdet":
            self.num_box = 256
            self.encoderstack = TransformerEncoder(N=4, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation, self_posembed=selfposembed,
                                                   cross_posembed=crossposembed)
        else:
            self.num_box = 128
            self.encoderstack = TransformerStack(N=6, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                 dropout=dropout, activation=activation, self_posembed=selfposembed)


    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, obj_x, point_x, point_pos, proposals, num_real_objs, logger=None, all_average=False, ctx_average=False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        '''Visual Feature'''
        obj_x = torch.cat(obj_x, dim=0).view(len(proposals), self.num_box, -1).permute(0, 2, 1)
        obj_x = self.obj_proj1(obj_x)
        obj_x = self.obj_proj2(obj_x)
        obj_x = obj_x.permute(0, 2, 1)

        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = torch.cat([proposal.get_field("padded_labels") for proposal in proposals]).view(len(proposals), self.num_box).long().cuda()
        else:
            obj_labels = None

        obj_embed = []
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            for i, (obj_label, N) in enumerate(zip(obj_labels, num_real_objs)):
                obj2embed = self.obj_embed(obj_label[: N])
                obj2embed = torch.cat([obj2embed, torch.zeros((self.num_box-N, 200)).cuda()], dim=0)
                assert(obj2embed.shape[0] == self.num_box)
                obj_embed.append(obj2embed)
            obj_embed = torch.cat(obj_embed, dim=0).view(len(proposals), self.num_box, -1)
        else:
            obj_logits = cat([proposal.get_field("pred_cls") for proposal in proposals], dim=0).detach()
            if self.mode == 'sgcls':
                obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
            elif self.mode == 'sgdet':
                obj_embed = self.obj_embed(obj_logits.long().cuda())

        '''Positional Information'''
        obj_pos = torch.cat([p.get_field("padded_centers") for p in proposals], dim=0).float()\
            .cuda().view(len(proposals), self.num_box, 3)
        obj_pos = torch.autograd.Variable(obj_pos, requires_grad=True)

        '''Padding Mask'''
        key_padding_mask = []
        for proposal in proposals:
            mask = proposal.get_field("padding_mask")
            key_padding_mask.append(mask)
        key_padding_mask = torch.cat(key_padding_mask, dim=0).view(len(proposals), self.num_box).byte().cuda()

        '''Transformer Stack'''
        obj_x = torch.cat([obj_embed, obj_x], dim=-1)
        # obj_x = self.obj_proj(obj_x)
        if self.mode == "sgdet":
            point_pos = torch.cat(point_pos, dim=0).view(len(proposals), -1, 3)
            point_x = torch.cat(point_x, dim=0).permute(0, 2, 1).contiguous().view(len(proposals), -1, 256)
            obj_feats = self.encoderstack(obj_x, point_x, obj_pos, point_pos, key_padding_mask)
        else:
            obj_feats = self.encoderstack(obj_x, obj_pos, key_padding_mask)

        obj_id_this_batch = []
        for proposal in proposals:
            obj_id_this_batch.append(proposal.get_field("obj_ids"))
        obj_id_this_batch = cat(obj_id_this_batch, dim=0)

        return obj_feats, None, None, obj_id_this_batch


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
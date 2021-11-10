import torch
import os
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from my_utils.misc import cat, line2space
from .utils_gcn import obj_edge_vectors, volume, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, \
    pseudo_encode_box_info
from .simpleGCN import simpleGCN

class FrequencyBias(nn.Module):
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
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        return joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

    def forward(self, labels):
        return self.index_with_labels(labels)


class DecoderRNN(nn.Module):
    def __init__(self, config, obj_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop):
        super(DecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim

        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.rnn_drop = rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            torch.nn.init.constant_(self.state_linearity.bias, 0.0)
            torch.nn.init.constant_(self.input_linearity.bias, 0.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:,
                                   0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])  # f
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])  # g
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])  # o
        memory = input_gate * memory_init + forget_gate * previous_memory  # c = i*g+f*c
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                     projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self, inputs, initial_state=None, labels=None, boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            out_pred = pred_dist[:, :].max(1)[1]
            out_commitments.append(out_pred)
            previous_obj_embed = self.obj_embed(out_pred + 1)

        out_commitments = torch.cat(out_commitments, 0)
        return torch.cat(out_dists, 0), out_commitments


class LSTMContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, obj_feauture_channels):
        super(LSTMContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
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
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.pos_embed = nn.Sequential(*[
            nn.Linear(24, 64), nn.BatchNorm1d(64, momentum=0.001),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
        ])

        self.obj_feature_dim = obj_feauture_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.obj_feature_dim + self.embed_dim + 128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                                      inputs_dim=self.hidden_dim + self.obj_feature_dim + self.embed_dim + 128,
                                      hidden_dim=self.hidden_dim,
                                      rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim + self.hidden_dim + self.obj_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True).flatten_parameters()

        self.protogcn = simpleGCN(in_dim=self.embed_dim + self.hidden_dim + self.obj_feature_dim, out_dim=self.hidden_dim)

        self.rel_proj = torch.nn.Linear((self.embed_dim + self.hidden_dim + self.obj_feature_dim)*2,
                                        self.embed_dim + self.hidden_dim + self.obj_feature_dim)

        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat",
                                 torch.zeros(self.hidden_dim + self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.obj_dim))

    def sort_rois(self, proposals):
        scores = volume(proposals)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        self.obj_ctx_rnn.flatten_parameters()
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]

        encoder_rep = self.lin_obj_h(encoder_rep)

        batch_size = encoder_rep.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)

        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,
                labels=obj_labels[perm] if obj_labels is not None else None,
            )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = torch.as_tensor(obj_labels)
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]
        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(torch.as_tensor(obj_labels).long().cuda())
        else:
            obj_logits = cat([proposal.get_field("pred_cls") for proposal in proposals], dim=0).detach()
            if self.mode == 'sgcls':
                obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            elif self.mode == 'sgdet':
                obj_embed = self.obj_embed1(obj_logits.long().cuda())

        encoded_pos = pseudo_encode_box_info(proposals)
        encoded_pos = encoded_pos.float()
        encoded_pos = encoded_pos.cuda()
        pos_embed = self.pos_embed(encoded_pos)

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels,
                                                                                    boxes_per_cls=None,
                                                                                    ctx_average=ctx_average)
        obj_embed2 = self.obj_embed2(obj_preds.long().cuda())

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = cat((obj_embed2, x, obj_ctx), -1)

        obj_rel_rep = list(obj_rel_rep.split([len(proposal) for proposal in proposals]))
        batch_X, batch_A, batch_rel_node = self.getGraph(obj_rel_rep, rel_pair_idxs, len(proposals),  obj_rel_rep[0].shape[-1], self.cfg)
        batch_node_feat = self.protogcn(X=batch_X, normalized_A=batch_A)
        edge_ctx = torch.zeros([len(proposals), 128, 512]).cuda()
        for i, (node_feat, rel_node) in enumerate(zip(batch_node_feat, batch_rel_node)):
            rel_node_idx = (rel_node >= 0).nonzero().squeeze(-1)
            rel_node_idx = rel_node[rel_node_idx]
            rel_feat = node_feat[rel_node_idx]
            edge_ctx[i][:rel_feat.shape[0]] = rel_feat

        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((obj_embed2, x), -1))

        obj_id_this_batch = []
        for proposal in proposals:
            obj_id_this_batch.append(proposal.get_field("obj_ids"))
        obj_id_this_batch = cat(obj_id_this_batch, dim=0)

        return obj_dists, obj_preds, edge_ctx, obj_id_this_batch

    def getGraph(self, obj_node_feat, rel_pair_idxs, batch_size, obj_feat_dim, cfg):
        num_box = cfg.MODEL.ROI_HEADS.MAX_PRED_BOX_PER_SCAN
        X = torch.zeros((batch_size, num_box*2, obj_feat_dim), dtype=torch.float).cuda()
        A = torch.zeros((batch_size, num_box*2, num_box*2), dtype=torch.float).cuda()
        batch_rel_node = torch.zeros((batch_size, 128), dtype=torch.int64).cuda() - 1
        rel_feat_dim = obj_feat_dim
        for i, (scan_pair, scan_nodes) in enumerate(zip(rel_pair_idxs, obj_node_feat)):
            scan_rel_node_idx = torch.zeros(128, dtype=torch.int64).cuda() - 1
            for j, pair in enumerate(scan_pair):
                '''Object Node Feature'''
                X[i][pair[0]] = scan_nodes[pair[0]]
                X[i][pair[1]] = scan_nodes[pair[1]]
                '''Relation Node Feature'''
                rel_node = self.rel_proj(torch.cat([scan_nodes[pair[0]], scan_nodes[pair[1]]], dim=0))
                rel_ind = pair[0] + pair[1]
                X[i][rel_ind] = rel_node
                scan_rel_node_idx[j] = rel_ind
                '''Adj Matrix'''
                A[i][pair[0]][rel_ind] = 1
                A[i][rel_ind][pair[1]] = 1
            A[i] = normalize(A[i])
            batch_rel_node[i] = scan_rel_node_idx
        return X, A, batch_rel_node


def normalize(A, symmetric=True):
    A = A.float() + torch.eye(A.shape[0]).float().cuda()
    D = A.sum(dim=1)
    if symmetric:
        D = torch.diag(torch.pow(D, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(D, -1))
        return D.mm(A)


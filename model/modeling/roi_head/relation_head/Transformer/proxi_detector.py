import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modeling.roi_head.relation_head.Motif_GCN.simpleGCN import simpleGCN, normalize, aGCN
from model.modeling.roi_head.relation_head.Motif.utils_motifs import pseudo_encode_box_info


def inverse(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


class proto_proxi_detector(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.num_rel_cls = num_rel_cls
        '''subject=主体, object=客体'''
        self.post_cat = nn.Linear(self.obj_dim * 2, self.rel_dim)
        self.rel_compress = nn.Linear(self.rel_dim, self.num_rel_cls)

    def forward(self, sub_feat, obj_feat):
        cat_feat = torch.cat([sub_feat, obj_feat], dim=-1)
        rel_feat = F.relu(self.post_cat(cat_feat))
        rel_dist = self.rel_compress(rel_feat)
        return rel_dist


class proxi_detector(nn.Module):
    def __init__(self, rel_dim, num_rel_cls):
        super().__init__()
        self.rel_dim = rel_dim
        self.num_rel_cls = num_rel_cls
        self.fc1 = nn.Linear(self.rel_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.num_rel_cls)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.rel_compress = nn.Linear(self.rel_dim, self.num_rel_cls)

    def forward(self, rel_feat):
        x_ = F.relu(self.fc1(rel_feat))
        x_ = F.relu(self.dropout(self.fc2(x_)))
        rel_dist = self.fc3(x_)
        return rel_dist


class proxi_sampler(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls, num_rel_sample, proxi_rel, mode, use_rel_node=False,
                 pos_fuse=False):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.num_obj = 128 if mode != "sgdet" else 256
        self.num_rel_cls = num_rel_cls
        self.num_rel_sample = 32
        self.positive_ratio = 0.5
        self.num_rel_sample_eval = 256
        self.proxi_rel = proxi_rel

        self.map = {}
        for i in range(len(self.proxi_rel)):
            self.map[self.proxi_rel[i]] = i + 1
        self.map[0] = 0

        self.detector = proxi_detector(rel_dim=rel_dim, num_rel_cls=num_rel_cls)
        self.obj_fuse = nn.Linear(self.obj_dim * 2, self.obj_dim)
        self.pos_embed = nn.Sequential(*[
            nn.Linear(3, 64),
            nn.Linear(64, self.rel_dim),
            nn.ReLU(inplace=True),
        ])
        nn.init.xavier_uniform_(self.obj_fuse.weight)
        nn.init.constant_(self.obj_fuse.bias, 0.0)

        self.gcn = simpleGCN(in_dim=self.obj_dim, out_dim=self.rel_dim)

        self.mode = mode
        self.use_rel_node = use_rel_node
        self.pos_fuse = pos_fuse
        assert (len(proxi_rel) + 1 == num_rel_cls)

    def forward(self, proposals, supportor, supported, obj_feats, rel_labels, rel_idx_pairs):
        final_rel_dist = []
        num_box_per_scan = [f.shape[0] for f in obj_feats]

        if self.training:
            N = self.num_obj + self.num_rel_sample if self.mode != 'sgdet' else self.num_obj + self.num_rel_sample
        else:
            N = self.num_obj + self.num_rel_sample_eval if self.mode != 'sgdet' else self.num_obj + self.num_rel_sample_eval

        A = torch.zeros((len(proposals), N, N), device=torch.device('cuda:0'))
        X = torch.zeros((len(proposals), N, self.obj_dim), device=torch.device('cuda:0'))
        obj_ctrs = [p.get_field("obj_centers") for p in proposals]

        '''Sample:FG&BG'''
        fgbg_pairs = []
        fgbg_labels = []
        if self.training:
            for pairs_j, labels_j in zip(rel_idx_pairs, rel_labels):
                '''FG'''
                no_fg = False
                fg_limit = int(self.num_rel_sample * self.positive_ratio)
                non_proximity_relation = torch.ones(labels_j.shape[0], device=torch.device('cuda:0'))
                proxi_rel_idx = []

                '''Look for FG'''
                for p in self.proxi_rel:
                    proxi_rel_idx.append((labels_j == p).nonzero().squeeze(-1))
                if proxi_rel_idx == []:
                    proxi_rel_idx = torch.as_tensor(proxi_rel_idx).cuda()
                else:
                    proxi_rel_idx = torch.cat(proxi_rel_idx, dim=0)

                '''Set FG proximity relation's flag to False'''
                non_proximity_relation[proxi_rel_idx] = 0

                '''Subsample, if necessary'''
                if proxi_rel_idx.shape[0] > fg_limit:
                    perm = torch.randperm(proxi_rel_idx.shape[0], device=torch.device('cuda:0'))[:fg_limit]
                    proxi_rel_idx = proxi_rel_idx[perm]
                num_fg = min(proxi_rel_idx.shape[0], fg_limit)
                '''Prepare Labels and Pairs'''
                if proxi_rel_idx.shape[0] == 0:
                    no_fg = True
                else:
                    proxi_pairs = pairs_j[proxi_rel_idx]
                    proxi_labels = labels_j[proxi_rel_idx]
                    for l in range(proxi_labels.shape[0]):  # label needs mapping to prevent index overflow
                        proxi_labels[l] = self.map[proxi_labels[l].item()]

                '''BG'''
                num_bg = self.num_rel_sample - num_fg
                '''Randomly Select BG'''
                bg_rel_idx = (non_proximity_relation == 1).nonzero().squeeze(-1)
                if bg_rel_idx.shape[0] > num_bg:
                    perm = torch.randperm(bg_rel_idx.shape[0], device=torch.device('cuda:0'))[:num_bg]
                    bg_rel_idx = bg_rel_idx[perm]
                '''Generate Labels and Pairs'''
                bg_pairs = pairs_j[bg_rel_idx]
                bg_labels = torch.zeros(bg_rel_idx.shape[0], device=torch.device('cuda:0'), dtype=torch.int64)

                '''Concatenate FG and BG'''
                if no_fg:
                    scan_rel_pairs = bg_pairs
                    scan_rel_labels = bg_labels
                    # scan_rel_pairs = torch.as_tensor([], dtype=torch.int64).cuda()
                    # scan_rel_labels = torch.as_tensor([], dtype=torch.int64).cuda()
                else:
                    scan_rel_pairs = torch.cat([proxi_pairs, bg_pairs], dim=0)
                    scan_rel_labels = torch.cat([proxi_labels, bg_labels], dim=0)
                    # scan_rel_pairs = proxi_pairs
                    # scan_rel_labels = proxi_labels
                fgbg_pairs.append(scan_rel_pairs)
                fgbg_labels.append(scan_rel_labels)
        else:
            for i, (supportor_i, supported_i) in enumerate(zip(supportor, supported)):
                if len(supported_i) <= 1 or len(supported_i) <= 1:
                    pairs = torch.as_tensor([], dtype=torch.int64, device=torch.device('cuda:0'))
                else:
                    pairs = self.generate_eval_pairs(supportor_this_scan=supportor_i,
                                                     supported_this_scan=supported_i)
                fgbg_pairs.append(pairs)
                # fgbg_labels.append(labels)
        '''Construct Graph'''
        rel_node_maps = []
        init_rel_feats = []
        for i, (supportor_i, supported_i, obj_feats_i, num_box, pairs_i) in enumerate(
                zip(supportor, supported, obj_feats, num_box_per_scan, fgbg_pairs)):
            if not self.use_rel_node:
                pass
            else:
                A[i], node_maps = self.get_graph_gcn_with_rel(N, pairs_i)
                X[i][:obj_feats_i.shape[0]] = obj_feats_i
                rel_feats_i = self.init_rel_node(node_maps, obj_feats_i, obj_ctrs[i], pairs_i)
                X[i][list(node_maps.values())] = rel_feats_i
                rel_node_maps.append(node_maps)
                init_rel_feats.append(rel_feats_i)
        '''GCN'''
        for j, (X_j, A_j) in enumerate(zip(X, A)):
            X_ = self.gcn(X_j, A_j)
            rel_dists = []
            for k, pairs in enumerate(fgbg_pairs[j]):
                # k = (pairs[0].item(), pairs[1].item())
                idx = rel_node_maps[j][k]
                rel_feat = X_[idx]
                '''Residual Connection'''
                rel_feat = rel_feat + init_rel_feats[j][idx - self.num_obj]

                rel_dists.append(self.detector(rel_feat).unsqueeze(0))
            if rel_dists == []:
                rel_dists = torch.as_tensor(rel_dists).cuda()
            else:
                rel_dists = torch.cat(rel_dists, dim=0)
            final_rel_dist.append(torch.softmax(rel_dists, -1))
        return final_rel_dist, fgbg_labels, fgbg_pairs, self.map

    def get_graph_gcn(self, supportor_this_scan, supported_this_scan, num_obj_this_scan):
        A = torch.zeros((num_obj_this_scan, num_obj_this_scan), device=torch.device('cuda:0'))
        for supportor_i, supported_i in zip(supportor_this_scan, supported_this_scan):
            # supported<->supported
            for a in supported_i:
                for b in supported_i:
                    if a != b:
                        A[a][b] = 1
        return A

    def get_graph_gcn_with_rel(self, N, pairs):
        A = torch.zeros((N, N), device=torch.device('cuda:0'))
        rel_node_maps = {}
        j = 0
        for p in pairs:
            rel_idx = j + self.num_obj
            A[p[0]][rel_idx] = 1
            A[rel_idx][p[1]] = 1
            A[p[1]][rel_idx] = 1
            A[rel_idx][p[0]] = 1
            rel_node_maps[j] = rel_idx
            j += 1
        return A, rel_node_maps

    def init_rel_node(self, rel_node_maps, obj_feats_i, obj_corners, pairs):

        rel_feats = torch.zeros((len(rel_node_maps), self.rel_dim), device=torch.device('cuda:0'))
        for i, (pair_idx, node_idx) in enumerate(rel_node_maps.items()):
            if not self.pos_fuse:
                sub = obj_feats_i[pairs[pair_idx][0]]
                obj = obj_feats_i[pairs[pair_idx][1]]
                feat = torch.cat([sub, obj], dim=-1)
                feat = self.obj_fuse(feat)
            else:
                sub = obj_corners[pairs[pair_idx][0]]
                obj = obj_corners[pairs[pair_idx][1]]
                feat = obj - sub
                feat = self.pos_embed(feat.view(-1))
            rel_feats[i] = feat
        return rel_feats

    def generate_eval_pairs(self, supportor_this_scan, supported_this_scan):
        """
        为eval生成测试用关系对
        单纯的从supportor和supported中抽取关系，达到指定数量或遍历完成后结束
        """
        pairs = []
        num_rel = 0
        reached = False
        for supportor_i, supported_i in zip(supportor_this_scan, supported_this_scan):
            if reached is True:
                break
            for a in supported_i:
                for b in supported_i:
                    if a != b:
                        if num_rel < self.num_rel_sample_eval:
                            pairs.append(torch.tensor((a, b), dtype=torch.int64, device=torch.device('cuda:0')).unsqueeze(0))
                            num_rel += 1
                        else:
                            reached = True
                            break
        if pairs == []:
            return torch.as_tensor([], dtype=torch.int64, device=torch.device('cuda:0'))
        else:
            return torch.cat(pairs, dim=0)


class impaired_gcn_detector(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls, num_rel_sample):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.num_rel_cls = num_rel_cls
        self.num_rel_sample = num_rel_sample
        self.detector = packed_GCN_detector(self.obj_dim, num_max_nodes=128)

    def forward(self, obj_feats, rel_pair_idxs):
        rel_dists = self.detector(obj_feats, rel_pair_idxs)
        return rel_dists


class packed_GCN_detector(nn.Module):
    def __init__(self, node_dim, num_max_nodes):
        super().__init__()
        self.node_dim = node_dim
        self.N = num_max_nodes
        self.detector = proxi_detector(rel_dim=node_dim, num_rel_cls=16+1)
        self.gcn = simpleGCN(in_dim=self.node_dim, out_dim=self.node_dim)

    def get_graph_gcn_with_rel(self, N, pairs):
        A = torch.zeros((N, N), device=torch.device('cuda:0'))
        rel_node_maps = {}
        j = 0
        for p in pairs:
            rel_idx = j + self.num_obj
            A[p[0]][rel_idx] = 1
            A[rel_idx][p[1]] = 1
            A[p[1]][rel_idx] = 1
            A[rel_idx][p[0]] = 1
            rel_node_maps[j] = rel_idx
            j += 1
        return A, rel_node_maps

    def init_rel_node(self, rel_node_maps, obj_feats_i, pairs):
        rel_feats = torch.zeros((len(rel_node_maps), self.node_dim), device=torch.device('cuda:0'))
        for i, (pair_idx, node_idx) in enumerate(rel_node_maps.items()):
            sub = obj_feats_i[pairs[pair_idx][0]]
            obj = obj_feats_i[pairs[pair_idx][1]]
            feat = torch.cat([sub, obj], dim=-1)
            feat = self.obj_fuse(feat)
            rel_feats[i] = feat
        return rel_feats

    def forward(self, obj_feats, rel_pair_idxs):
        final_rel_dist = []

        A = torch.zeros((len(obj_feats), self.N, self.N), device=torch.device('cuda:0'))
        X = torch.zeros((len(obj_feats), self.N, self.node_dim), device=torch.device('cuda:0'))

        rel_node_maps = []
        init_rel_feats = []
        for i, (obj_feats_i, pairs_i) in enumerate(
                zip(obj_feats, rel_pair_idxs)):
            A[i], node_maps = self.get_graph_gcn_with_rel(self.N, pairs_i)
            X[i][:obj_feats_i.shape[0]] = obj_feats_i
            rel_feats_i = self.init_rel_node(node_maps, obj_feats_i, pairs_i)
            X[i][list(node_maps.values())] = rel_feats_i
            rel_node_maps.append(node_maps)
            init_rel_feats.append(rel_feats_i)
        '''GCN'''
        for j, (X_j, A_j) in enumerate(zip(X, A)):
            X_ = self.gcn(X_j, A_j)
            rel_dists = []
            for k, pairs in enumerate(rel_pair_idxs[j]):
                idx = rel_node_maps[j][k]
                rel_feat = X_[idx]
                '''Residual Connection'''
                rel_feat = rel_feat + init_rel_feats[j][idx - self.num_obj]
                rel_dists.append(self.detector(rel_feat).unsqueeze(0))
            if rel_dists == []:
                rel_dists = torch.as_tensor(rel_dists).cuda()
            else:
                rel_dists = torch.cat(rel_dists, dim=0)
            final_rel_dist.append(torch.softmax(rel_dists, -1))
        return final_rel_dist








if __name__ == "__main__":
    '''
    supportor = [0, 2, 3]
    supported = [[2, 3], [6, 8], [9]]
    A = get_graph_gcn(supportor_this_scan=supportor, supported_this_scan=supported, N=10)
    '''
    a = torch.tensor((0, 1, 2, 3, 4))
    perm = [0, 4, 3, 1, 2]
    inv_perm = inverse(perm)
    b = a[perm]
    c = b[inv_perm]
    pass

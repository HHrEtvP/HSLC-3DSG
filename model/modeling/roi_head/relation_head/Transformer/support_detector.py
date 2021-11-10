import torch
import torch.nn as nn
from torch.nn import functional as F


class proto_support_detector(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        # self.conv_sub = nn.Conv1d(obj_dim, 512, kernel_size=1)
        # self.conv_obj = nn.Conv1d(obj_dim, 512, kernel_size=1)
        self.conv_post_cat = nn.Linear(self.obj_dim*2, self.obj_dim)
        self.rel_compress = nn.Linear(self.obj_dim, num_rel_cls)

    def forward(self, sub_feat, obj_feat):
        # sub_feat = self.conv_sub(sub_feat)
        # obj_feat = self.conv_obj(obj_feat)
        rel_feat = torch.cat([sub_feat, obj_feat], dim=-1)
        rel_feat = self.conv_post_cat(rel_feat)
        rel_dist = self.rel_compress(rel_feat)
        return rel_dist


class support_sampler(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls, num_rel_sample, support_rel):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.num_rel_cls = num_rel_cls
        self.num_rel_sample = num_rel_sample
        self.support_rel = support_rel

        self.map = {}
        for i in range(len(self.support_rel)):
            self.map[self.support_rel[i]] = i + 1
        self.map[0] = 0

        self.detector = proto_support_detector(obj_dim=self.obj_dim, rel_dim=self.rel_dim, num_rel_cls=self.num_rel_cls)
        assert(len(support_rel) + 1 == num_rel_cls)

    def forward(self, obj_feats, rel_idx_pairs, rel_labels, floor_idx):
        rel_dist = []
        new_rel_labels = []
        supportor_batch = []
        supported_batch = []
        assert(len(obj_feats) == len(rel_labels) == len(rel_idx_pairs))
        if self.training:
            for i, (feats, labels, pairs, floor) in enumerate(zip(obj_feats, rel_labels, rel_idx_pairs, floor_idx)):
                no_fg = False
                assert(isinstance(pairs, torch.Tensor))
                if self.training:
                    fg_dist = []
                    not_support_pairs = torch.ones(pairs.shape[0], device=torch.device('cuda:0'))
                    supportor, supported, support_rel_idx = self.get_set_train(labels=labels, pairs=pairs, floor_idx=floor)
                    supportor_batch.append(supportor)
                    supported_batch.append(supported)

                    if support_rel_idx.shape[0] > 0:
                        not_support_pairs[support_rel_idx] = 0
                        fg_labels = labels[support_rel_idx]
                        for j, label in enumerate(fg_labels):
                            fg_labels[j] = self.map[label.item()]
                        assert(len(supportor) == len(supported))

                        for k, A in enumerate(supportor):
                            for B in supported[k]:
                                dist = self.detector(sub_feat=feats[B], obj_feat=feats[A])
                                fg_dist.append(dist.unsqueeze(0))
                    else:
                        no_fg = True
                    bg_dist = []
                    num_bg_support_rel = self.num_rel_sample - len(fg_dist)
                    non_support_rels = pairs[not_support_pairs == 1]
                    if non_support_rels.shape[0] > num_bg_support_rel:
                        perm = torch.randperm(non_support_rels.shape[0], device=torch.device('cuda:0'))[:num_bg_support_rel]
                        non_support_rels = non_support_rels[perm]

                    for pair in non_support_rels:
                        sub = feats[pair[0]]
                        obj = feats[pair[1]]
                        dist = self.detector(sub_feat=sub, obj_feat=obj)
                        bg_dist.append(dist.unsqueeze(0))

                    if not no_fg:
                        fg_dist = torch.cat(fg_dist, dim=0)

                    if bg_dist == []:
                        bg_dist = torch.as_tensor(bg_dist).cuda()
                    else:
                        bg_dist = torch.cat(bg_dist, dim=0)
                    bg_labels = torch.zeros(bg_dist.shape[0], device=torch.device('cuda:0'), dtype=torch.int64)

                    if no_fg:
                        rel_dist_this_scan = torch.cat([bg_dist], dim=0)
                        rel_dist.append(rel_dist_this_scan)
                        rel_label_this_scan = torch.cat([bg_labels], dim=0)
                        new_rel_labels.append(rel_label_this_scan)
                        continue
                    else:
                        rel_dist_this_scan = torch.cat([fg_dist, bg_dist], dim=0)
                        rel_dist.append(rel_dist_this_scan)
                        rel_label_this_scan = torch.cat([fg_labels, bg_labels], dim=0)
                        new_rel_labels.append(rel_label_this_scan)

        else:
            for i, (feats, labels, floor) in enumerate(zip(obj_feats, rel_labels, floor_idx)):
                supportor = []
                supported = []
                supported_exclude = [floor]
                has_support = True
                supportor_queue = [floor]
                rel_dist_this_scan = []
                while has_support:
                    supportor_now = supportor_queue[0]
                    supportor_queue.pop(0)
                    dists = []
                    supported_candidate = []

                    for j in range(feats.shape[0]):
                        if j not in supported_exclude:
                            dists.append(self.detector(sub_feat=feats[j], obj_feat=feats[supportor_now]))
                            supported_candidate.append(j)
                    supported_candidate = torch.as_tensor(supported_candidate, dtype=torch.int64, device=torch.device('cuda:0'))

                    if len(dists) == 0:
                        if len(supportor_queue) == 0:
                            has_support = False
                        continue

                    dists = torch.cat(dists, dim=0).view((-1, self.num_rel_cls))
                    preds = torch.argmax(dists, dim=-1)

                    if (preds != 0).nonzero().shape[0] != 0:
                        confirmed = supported_candidate[(preds != 0).nonzero().squeeze(-1)]
                        rel_dist_this_scan.append(dists[(preds != 0).nonzero().squeeze(-1)])
                        supported.append(confirmed.tolist())
                        supportor.append(supportor_now)

                        for e in confirmed.tolist():
                            supportor_queue.append(e)

                        supported_exclude.append(supportor_now)

                    elif len(supportor_queue) == 0:
                        has_support = False

                supportor_batch.append(supportor)
                supported_batch.append(supported)
                if rel_dist_this_scan == []:
                    rel_dist.append(torch.as_tensor(rel_dist_this_scan).cuda())
                else:
                    rel_dist.append(torch.cat(rel_dist_this_scan, dim=0))

        return rel_dist, new_rel_labels, supportor_batch, supported_batch, self.map

    def get_set_train(self, labels, pairs, floor_idx):
        assert(labels.shape[0] == pairs.shape[0])
        supportor = []
        supported = []
        supportor_now = floor_idx
        no_support = False
        support_rel_idx = []
        supportor_candidate = []
        supportor_exclude = []
        while not no_support:
            supported_now = []
            idx_i = (pairs[:, 1] == supportor_now).nonzero().squeeze(-1)
            for idx in idx_i:
                if labels[idx] in self.support_rel:
                    supported_now.append(pairs[idx][0].item())
                    support_rel_idx.append(idx.item())
            supportor_exclude.append(supportor_now)
            if len(supported_now) > 0:
                supportor.append(supportor_now)
                supported.append(supported_now)
                for s in supported_now:
                    if s not in supportor_exclude and s not in supportor_candidate:
                        supportor_candidate.append(s)
                if len(supportor_candidate) > 0:
                    supportor_now = supportor_candidate[0]
                    supportor_candidate.pop(0)
                else:
                    no_support = True
            else:
                if len(supportor_candidate) == 0:
                    no_support = True
                else:
                    supportor_now = supportor_candidate[0]
                    supportor_candidate.pop(0)
        return supportor, supported, torch.as_tensor(support_rel_idx, device=torch.device('cuda:0')).long()


class impaired_mlp_detector(nn.Module):
    def __init__(self, obj_dim, rel_dim, num_rel_cls, num_rel_sample):
        super().__init__()
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim
        self.num_rel_cls = num_rel_cls
        self.num_rel_sample = num_rel_sample
        self.detector = proto_support_detector(obj_dim=self.obj_dim, rel_dim=self.rel_dim, num_rel_cls=self.num_rel_cls)

    def forward(self, obj_feats, rel_idx_pairs):
        rel_dists = []
        for i, (feats, pairs) in enumerate(zip(obj_feats, rel_idx_pairs)):
            dists = []
            for j, pair in enumerate(pairs):
                dists.append(self.detector(sub_feat=feats[pair[0]], obj_feat=feats[pair[1]]))
            if dists == []:
                dists = torch.as_tensor([], dtype=torch.float32, device=torch.device('cuda:0'))
            else:
                dists = torch.cat(dists).view(-1, 17)
            rel_dists.append(dists)
        return rel_dists







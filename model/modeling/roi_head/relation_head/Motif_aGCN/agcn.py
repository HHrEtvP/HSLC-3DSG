import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # [Nobj x Nrel]*[Nrel x dim], aka A * XW, standard GCN equation
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)  # attention here
        return collect_avg


class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()

    def forward(self, target, source):
        """propagate node features"""
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, obj_dim, rel_dim):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(rel_dim, obj_dim)) # rel -> subject
        self.collect_units.append(_Collection_Unit(rel_dim, obj_dim)) # rel -> object
        self.collect_units.append(_Collection_Unit(obj_dim, rel_dim)) # subject -> rel
        self.collect_units.append(_Collection_Unit(obj_dim, rel_dim)) # object -> rel
        self.collect_units.append(_Collection_Unit(obj_dim, obj_dim)) # subject <-> object

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, obj_dim, rel_dim):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(obj_dim)) # obj from others
        self.update_units.append(_Update_Unit(rel_dim)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


class aGCN(nn.Module):
    def __init__(self, obj_dim, rel_dim, step):
        super().__init__()
        self.step = step
        self.gcn_collect_f = _GraphConvolutionLayer_Collect(obj_dim, rel_dim)
        self.gcn_update_f = _GraphConvolutionLayer_Update(obj_dim, rel_dim)
        self.gcn_collect_l = _GraphConvolutionLayer_Collect(26, 17)
        self.gcn_update_l = _GraphConvolutionLayer_Update(26, 17)
        self.obj_predictor = nn.Linear(obj_dim, 26)
        self.rel_predictor = nn.Linear(rel_dim, 17)

    def forward(self, obj_feats_input, rel_feats_input, obj_obj, sub_rel, obj_rel):
        rel_feats = [rel_feats_input]
        obj_feats = [obj_feats_input]

        for t in range(self.step):
            '''update object feature'''
            '''message from other object'''
            source_obj = self.gcn_collect_f(obj_feats[t], obj_feats[t], obj_obj, 4)
            '''message from relation'''
            source_rel_sub = self.gcn_collect_f(obj_feats[t], rel_feats[t], sub_rel, 0)
            source_rel_obj = self.gcn_collect_f(obj_feats[t], rel_feats[t], obj_rel, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_feats.append(self.gcn_update_f(obj_feats[t], source2obj_all, 0))

            '''update predicate feature'''
            '''message from object'''
            source_obj_sub = self.gcn_collect_f(rel_feats[t], obj_feats[t], sub_rel.t(), 2)
            source_obj_obj = self.gcn_collect_f(rel_feats[t], obj_feats[t], obj_rel.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            rel_feats.append(self.gcn_update_f(rel_feats[t], source2rel_all, 1))

        obj_dist = self.obj_predictor(obj_feats[-1])
        rel_dist = self.rel_predictor(rel_feats[-1])

        """
        
        obj_logits = [obj_dist]
        rel_logits = [rel_dist]

        for t in range(self.step):
            '''update object feature'''
            '''message from other object'''
            source_obj = self.gcn_collect_l(obj_logits[t], obj_logits[t], obj_obj, 4)
            '''message from relation'''
            source_rel_sub = self.gcn_collect_l(obj_logits[t], rel_logits[t], sub_rel, 0)
            source_rel_obj = self.gcn_collect_l(obj_logits[t], rel_logits[t], obj_rel, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_logits.append(self.gcn_update_l(obj_logits[t], source2obj_all, 0))

            '''update predicate feature'''
            '''message from object'''
            source_obj_sub = self.gcn_collect_l(rel_logits[t], obj_logits[t], sub_rel.t(), 2)
            source_obj_obj = self.gcn_collect_l(rel_logits[t], obj_logits[t], obj_rel.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            rel_logits.append(self.gcn_update_l(rel_logits[t], source2rel_all, 1))
        """
        return obj_dist, rel_dist



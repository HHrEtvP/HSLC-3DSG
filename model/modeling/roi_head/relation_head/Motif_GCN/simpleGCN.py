import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def normalize(A, symmetric=True):
    A = A + torch.eye(A.shape[0], device=torch.device('cuda:0'))
    D = A.sum(dim=1)
    if symmetric:
        D = torch.diag(torch.pow(D, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(D, -1))
        return D.mm(A)


class simpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim*2, bias=True)
        self.fc3 = nn.Linear(out_dim*2, out_dim, bias=True)
        normal_init(self.fc1, 0, 0.01)
        normal_init(self.fc2, 0, 0.01)
        normal_init(self.fc3, 0, 0.01)

    def forward(self, X, normalized_A):
        """
        :param X: [B, N, in_dim], Node Feature
        :param normalized_A: [B, N, N] Normalized
        :return: AXW
        """
        '''
        node_feat = torch.zeros((X.shape[0], X.shape[1], self.out_dim)).cuda()
        for i in range(X.shape[0]):
            feat = F.relu(self.fc1(normalized_A[i].mm(X[i])))
            feat = F.relu(self.fc2(normalized_A[i].mm(feat)))
            feat = self.fc3(normalized_A[i].mm(feat))
            node_feat[i] = feat
        '''
        # node_feat = torch.zeros((X.shape[0], self.out_dim)).cuda()
        X = F.relu(self.fc1(X))
        X = torch.mm(normalized_A, X)
        X = F.relu(self.fc2(X))
        X = torch.mm(normalized_A, X)
        X = self.fc3(X)
        X = torch.mm(normalized_A, X)
        return X


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
        self.collect_units.append(_Collection_Unit(obj_dim, obj_dim))  # subject <-> object

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, obj_dim, rel_dim):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(obj_dim))  # obj from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


class aGCN(nn.Module):
    def __init__(self, obj_dim, rel_dim, step):
        super().__init__()
        self.step = step
        self.gcn_collect_f = _GraphConvolutionLayer_Collect(obj_dim, rel_dim)
        self.gcn_update_f = _GraphConvolutionLayer_Update(obj_dim, rel_dim)


    def forward(self, obj_feats_input, obj_obj):
        obj_feats = [obj_feats_input]

        for t in range(self.step):
            '''update object feature'''
            '''message from other object'''
            source_obj = self.gcn_collect_f(obj_feats[t], obj_feats[t], obj_obj, 0)
            obj_feats.append(self.gcn_update_f(obj_feats[t], source_obj, 0))

        # obj_dist = self.obj_predictor(obj_feats[-1])
        # rel_dist = self.rel_predictor(rel_feats[-1])

        return obj_feats[-1]


if __name__ == "__main__":
    GCN = simpleGCN(5, 1)
    X = torch.ones((5, 5))
    A = torch.ones((5, 5))
    A = normalize(A)
    output = GCN(X, A)
    print(output)


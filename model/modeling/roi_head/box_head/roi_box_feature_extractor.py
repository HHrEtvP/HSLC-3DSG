"""
提取一个bbox的特征
这里应当只需要简单的提取，而不像fasterRCNN需要roi_pooling或者roi_align等操作
只接收一个proposals参数(一个batch上的box3d_list)
这些proposal都经过sampling处理过，
每一个proposal都有一个label域
其实就是ret_dict["sem_cls_label"]，这个数组存储了一个box3d_list中每个bbox的label
0代表这个box没有对应的GT，可以理解为BG
特征的提取可以用pointnet2搞一下
pointnet2具体的细节就不在这里写了，输入一个点云，输出这个点云的特征
pointnet2每次都会挑选一些seed point，然后用ball query在这些seed point邻域内进行特征提取，这样每个seed point都会提取出一个特征
然后这些seed point作为新的点云输入下一层，如此继续
直到最后一层，剩下的所有点统一提取特征，就会得到一个单一的特征向量
"""
from __future__ import print_function

from torch import nn
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from model.modeling.detector.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from model.structure.pc_list import PcList
from model.structure.box3d_list import Box3dList
from model.dataset.rscan_utils import extract_pc_in_box3d
from model.modeling.detector.models.backbone_module import Pointnet2Backbone


class box_feature_extractor(nn.Module):
    """
    Modified PointNet++ extractor, currently deprecated
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_dim = in_dim  # in_dim仅指feature
        self.output_dim = out_dim

        self.sa1 = PointnetSAModuleVotes(  # 修改一下超参数
            npoint=256,
            radius=0.2,
            nsample=32,
            mlp=[self.input_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=64,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(  # 只要npoint给的是None，就算作是group_all了，radius和nsample也全部置None
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 128, 128, self.output_dim],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=32,
            radius=0.6,
            nsample=64,
            mlp=[256, 128, 128, self.output_dim],
            use_xyz=True,
            normalize_xyz=True
        )

        '''
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        '''


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()  # (B,C,N)
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pclist, end_points=None):
        """
        :param pclist: 一个batch上的pc_list
        :param end_points: 要返回的结果
        :return:
        """
        '''
        assert(isinstance(pclist, PcList))
        in_xyz, features = self._break_up_pc(
            pc=pclist.xyz
        )
        '''
        in_xyz = pclist
        features = None
        if features is not None:
            sa1_xyz, sa1_features, _ = self.sa1(in_xyz.float(),
                                        features.transpose(1, 2).float())
        else:
            sa1_xyz, sa1_features, _ = self.sa1(in_xyz.float())  # 因为没有feature，所以留空
        sa2_xyz, sa2_features, _ = self.sa2(sa1_xyz, sa1_features)
        sa3_xyz, sa3_features, _ = self.sa3(sa2_xyz, sa2_features)
        sa4_xyz, sa4_features, _ = self.sa4(sa2_xyz, sa2_features)
        # features = self.fp1(sa2_xyz, sa4_xyz, sa2_features,
                            # sa4_features)
        # features = self.fp2(sa1_xyz, sa2_xyz, sa1_features, features)
        return sa3_features, sa4_features
    """
    obj features:(256, )
    point features:(128, 256)
    """


class PointNet(nn.Module):
    """Part of the PointNetCls"""
    def __init__(self, point_num):
        super(PointNet, self).__init__()

        self.inputTransform = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num, 1)),
        )
        self.inputFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.featureTransform = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num, 1)),
        )
        self.featureFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 64),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.7,inplace=True),对于ShapeNet数据集来说,用dropout反而准确率会降低
            nn.Linear(256, 16),
            nn.Softmax(dim=1),
        )
        self.inputFC[4].weight.data = torch.zeros(3 * 3, 256)
        self.inputFC[4].bias.data = torch.eye(3).view(-1)

    def forward(self, x):  # [B, N, XYZ]
        '''
            B:batch_size
            N:point_num
            K:k_classes
            XYZ:input_features
        '''
        batch_size = x.size(0)  # batchsize大小
        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        t_net = self.inputTransform(x)  # [B, 1024, 1,1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.inputFC(t_net)  # [B, 3*3]
        t_net = t_net.view(batch_size, 3, 3)  # [B, 3, 3]

        x = x.squeeze(1)  # [B, N, XYZ]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, XYZ]# 因为mm只能二维矩阵之间，故逐个乘再拼起来

        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        x = self.mlp1(x)  # [B, 64, N, 1]

        t_net = self.featureTransform(x)  # [B, 1024, 1, 1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.featureFC(t_net)  # [B, 64*64]
        t_net = t_net.view(batch_size, 64, 64)  # [B, 64, 64]

        x = x.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, 64]

        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, N, 1]

        p_x = self.mlp2(x)  # [B, N, 64]

        x, _ = torch.max(p_x, 2)  # [B, 1024, 1]

        x = self.fc(x.squeeze(2))  # [B, K]
        return x, p_x


class STN3d(nn.Module):
    """Part of the PointNetCls"""
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """Part of the PointNetCls"""
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    """Part of the PointNetCls"""
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        pf = self.bn3(self.conv3(x))
        x = torch.max(pf, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, pf
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat, pf


class PointNetCls(nn.Module):
    """Modified PointNet extractor, currently in use"""
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, pf = self.feat(x)
        x_ = F.relu(self.bn1(self.fc1(x)))
        x_ = F.relu(self.bn2(self.dropout(self.fc2(x_))))
        x_ = self.fc3(x_)
        return F.log_softmax(x_, dim=-1), x, trans_feat, pf


if __name__ == "__main__":
    extractor = box_feature_extractor(in_dim=3, out_dim=256)
    model_dict = extractor.state_dict()
    for k in model_dict.keys():
        print(k)



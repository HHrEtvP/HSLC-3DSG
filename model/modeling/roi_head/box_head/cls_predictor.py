'''
Perform Object Classification (SGCLS only)
Input is the pointcloud of the BOX, its location and heading is known
Output the predicted class of the BOX
'''

import torch
from torch import nn
from torch.nn import functional as F

from model.modeling.detector.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from model.structure.pc_list import PcList
from model.structure.box3d_list import Box3dList
from model.dataset.rscan_utils import extract_pc_in_box3d
from model.modeling.detector.models.backbone_module import Pointnet2Backbone

class cls_predictor(nn.Module):
    def __init__(self, in_dim, num_class):
        '''
        :param in_dim: input feature dim, if only XYZ is provided, then in_dim == 0
        :param num_class: number of foreground classes
        '''
        super().__init__()
        self.input_dim = in_dim
        self.num_class = num_class

        self.sa1 = PointnetSAModuleVotes(  # 修改一下超参数
            npoint=128,
            radius=0.2,
            nsample=32,
            mlp=[self.input_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        ).cuda()

        self.sa2 = PointnetSAModuleVotes(
            npoint=32,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        ).cuda()

        self.sa3 = PointnetSAModuleVotes(  # 只要npoint给的是None，就算作是group_all了，radius和nsample也全部置None
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        ).cuda()

        self.post_SA = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_class),
        ).cuda()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()  # (B,C,N)
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pclist):
        assert (isinstance(pclist, PcList))
        if self.input_dim>0:
            in_xyz, features = self._break_up_pc(
                pc=pclist.xyz
            )
        else:
            in_xyz = pclist.xyz
            features = None

        if features is None:
            xyz, features, _ = self.sa1(in_xyz.float().cuda())
        else:
            xyz, features, _ = self.sa1(in_xyz.float().cuda(), features.transpose(1, 2).float().cuda())
        xyz, features, _ = self.sa2(xyz.cuda(), features.cuda())
        xyz, features, _ = self.sa3(xyz.cuda(), features.cuda())  # (batch_pc_size, 128)
        output = self.post_SA(features.cuda())  # (batch_pc_size , num_class)

        return output

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import FTHE_DATA_DIR
from lib.models.spin import Regressor, hmr

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            seqlen=16
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.gru2 = nn.GRU(
            input_size=1024,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.gru3 = nn.GRU(
            input_size=1024,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        self.linear = nn.Linear(hidden_size, 2048)
        self.SA = SpatialAttention(7)

    def forward(self, x):

        n, t, f = x.shape
        identity = x

        # extract coarse-grained temporal feature
        x_coarse = identity.permute(1, 0, 2)
        y_coarse, _ = self.gru(x_coarse)

        # extract fine-grained temporal feature
        x_fine = torch.empty([n, int(t*2), int(f/2)]).to('cuda')
        for i in range(int(t*2)):
             if i % 2 == 0:
                 x_fine[:, i, :] = identity[:, int(i/2), 0:1024]
                 x_fine[:, i+1, :] = identity[:, int(i/2), 1024:2048]
             else:
                 continue
        x_fine = x_fine.permute(1, 0, 2)
        y_fine, _ = self.gru2(x_fine)

        # Integrate multi-granularity features
        y = torch.cat((y_fine, y_coarse), dim=0)
        y, _ = self.gru3(y)

        y = y[-t:, ...]

        y = F.relu(y)
        y = self.linear(y.view(-1, y.size(-1)))
        y = y.view(t, n, f)

        x = self.SA(x_coarse) * x_coarse
        y = y + x

        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y


class FTHE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(FTHE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(FTHE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


class FTHE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(FTHE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(FTHE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        feature = feature.reshape(batch_size, seqlen, -1)
        feature = self.encoder(feature)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output

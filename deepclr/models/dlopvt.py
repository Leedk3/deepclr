import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_cluster import knn
import torchgeometry as tgm
from pointnet2 import PointnetSAModuleMSG
from functools import partial
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch, sample_points_with_roi
from pcdet.utils import common_utils
from pcdet.utils.spconv_utils import replace_feature, spconv
from pcdet.models.backbones_3d.spconv_backbone import post_act_block 
from spconv.pytorch.utils import PointToVoxel as VoxelGenerator
from torch.profiler import profile, record_function, ProfilerActivity
from .transformer import SelfAttention, Transformer3D, Transformer

from detr3d.utils.pc_util import scale_points, shift_scale_points

from detr3d.models.helpers import GenericMLP
from detr3d.models.position_embedding import PositionEmbeddingCoordsSine
from detr3d.models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)


from ..config.config import Config
from ..data.labels import LabelType
from ..utils.factory import factory
from ..utils.metrics import trans_loss, rot_loss

from .base import BaseModel
from .helper import Conv1dMultiLayer, LinearMultiLayer
# from pcdet.models.backbones_3d.pfe import 

# my note :
# change point feature extractor -> voxel feature extractor (same as PV_RCNN network)

class DLOPVTModule(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for DLOPVT modules."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


def split_features(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split complete point cloud into xyz coordinates and features."""
    xyz = x[:, :3, :].transpose(1, 2).contiguous()
    features = (
        x[:, 3:, :].contiguous()
        if x.size(1) > 3 else None
    )
    return xyz, features


def merge_features(xyz: torch.Tensor, features: Optional[torch.Tensor]) -> torch.Tensor:
    """Merge xyz coordinates and features to point cloud."""
    if features is None:
        return xyz.transpose(1, 2)
    else:
        return torch.cat((xyz.transpose(1, 2), features), dim=1)

class BackBoneModule(abc.ABC, nn.Module):
    """Abstract base class for backbone."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, clouds) -> torch.Tensor:
        raise NotImplementedError


class MeanVFE(BackBoneModule):
    def __init__(self, num_point_features, voxel_size, point_cloud_range, max_points_per_voxel, max_num_voxels, **kwargs):
        super().__init__()
        self.num_point_features = num_point_features
        self.voxel_generator = None
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels

        # print("self.num_point_features", self.num_point_features)
        # print("self.voxel_generator", self.voxel_generator)
        # print("self.voxel_size", self.voxel_size)
        # print("self.point_cloud_range", self.point_cloud_range)
        # print("self.max_points_per_voxel", self.max_points_per_voxel)
        # print("self.max_num_voxels", self.max_num_voxels)
        # self.num_point_features 128
        # self.voxel_generator None
        # self.voxel_size [0.05, 0.05, 0.1]
        # self.point_cloud_range [-40.0, -40.0, -4, 40.0, 40.0, 2]
        # self.max_points_per_voxel 5
        # self.max_num_voxels 150000

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, clouds, **kwargs): #changed batch_dict input to cloud
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_dict={}
        pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])
        batch_dict['points'] = pts
        
        if self.voxel_generator is None:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_voxels=self.max_num_voxels,
                max_num_points_per_voxel=self.max_points_per_voxel,
                device= torch.device("cuda")
            )
        # voxel generator in spconv generate indices in ZYX order, the params format are XYZ.
        # generated indices don't include batch axis, you need to add it by yourself.
        # see examples/voxel_gen.py for examples.
        # https://github.com/traveller59/spconv/blob/bd5bc8db882a0ad12a13e87a76b668c90ca68c4e/docs/USAGE.md
        points = batch_dict['points'].contiguous()

        voxel_output = self.voxel_generator(points, empty_mean=False)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        num_voxels_in_batch = voxels.shape[0] / clouds.shape[0] 
        num_voxels_in_batch = int(num_voxels_in_batch)

        # if not batch_dict['use_lead_xyz']:
        #     voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        batch_dict['voxels'] = voxels
        batch_dict['voxel_coords'] = coordinates
        batch_dict['voxel_num_points'] = num_points

        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        # print("VFE batch_dict['voxel_features'] : ", batch_dict['voxel_features'].shape)
        # print("VFE batch_dict['voxel_coords'] : ", batch_dict['voxel_coords'].shape)
        # print("VFE batch_dict['voxel_coords'] : ", batch_dict['voxel_coords'])
        # print("VFE batch_dict['voxel_num_points'] : ", batch_dict['voxel_num_points'].shape)
        # print(voxel_features.shape, voxel_num_points.shape)

        return batch_dict

class VoxelBackBone8xBase(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }


    def output_dim(self) -> int:
        return self.num_point_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # Generate the batch indices
        # voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx] -->spconv 2.x output :  z_idx, y_idx, x_idx
        batch_indices = torch.arange(batch_size, device=voxel_coords.device).view(-1, 1).repeat(1, voxel_coords.shape[0]).view(-1, batch_size).long()
        batch_indices = batch_indices[:,0]
        batch_indices = batch_indices.unsqueeze(1)
        voxel_coords = torch.cat([batch_indices, voxel_coords], dim=1)
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelSetAbstraction(DLOPVTModule):
    voxel_backbone_8x : BackBoneModule
    def __init__(self, input_dim: int, point_dim: int, 
                 PFE: Dict, BACKBONE_3D : Dict, 
                 batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        assert point_dim == 3
        # assert len(mlps) == len(npoint) == len(radii) == len(nsamples)
        # assert 0 < len(mlps) <= 2

        self._point_dim = point_dim
        self.input_feat_dim = input_dim - self._point_dim
        # self._output_feat_dim = int(np.sum([x[-1] for x in mlps[-1]]))
        self.PFE = PFE
        self.voxel_size = PFE.VOXEL_SIZE
        self.point_cloud_range = PFE.POINT_CLOUD_RANGE

        SA_cfg = PFE.SA_LAYER
        self.data_dict = Dict

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in PFE.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        # if 'bev' in PFE.FEATURES_SOURCE:
        #     c_bev = num_bev_features
        #     c_in += c_bev

        if 'raw_points' in PFE.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [PFE.NUM_POINT_FEATURES - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, PFE.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(PFE.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = PFE.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        grid_size = (np.array(PFE.POINT_CLOUD_RANGE[3:6]) - np.array(PFE.POINT_CLOUD_RANGE[0:3])) / np.array(PFE.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.meanVEF = MeanVFE(input_dim, 
                               PFE.VOXEL_SIZE, 
                               PFE.POINT_CLOUD_RANGE,
                               PFE.MAX_POINTS_PER_VOXEL,
                               PFE.MAX_NUMBER_OF_VOXELS)

        self.voxel_backbone_8x = VoxelBackBone8xBase(BACKBONE_3D.NAME,
                                                     input_channels=PFE.NUM_POINT_FEATURES,
                                                     grid_size= self.grid_size,
                                                     voxel_size=PFE.VOXEL_SIZE,
                                                     point_cloud_range=PFE.POINT_CLOUD_RANGE)

    def output_dim(self) -> int:
        return 3 + self.PFE.NUM_OUTPUT_FEATURES

    def get_sampled_points(self, clouds):

        batch_size = clouds.shape[0] #batch_dict['batch_size']
        pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])

        if self.PFE.POINT_SOURCE == 'raw_points':
            src_points = pts[:, :3]
            batch_indices = torch.arange(batch_size, device=clouds.device).view(-1, 1).repeat(1, clouds.shape[2]).view(-1).long()
            # print("src_points:", src_points.shape)
            # print("batch_indices:", batch_indices, batch_indices.shape)
        # elif self.PFE.POINT_SOURCE == 'voxel_centers':
        #     src_points = common_utils.get_voxel_centers(
        #         batch_dict['voxel_coords'][:, 1:4],
        #         downsample_times=1,
        #         voxel_size=self.voxel_size,
        #         point_cloud_range=self.point_cloud_range
        #     )
        #     batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.PFE.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.PFE.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.PFE.NUM_KEYPOINTS:
                    empty_num = self.PFE.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.PFE.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, clouds: torch.Tensor, *_args: Any) -> torch.Tensor:
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """

        keypoints = self.get_sampled_points(clouds) #keypoints: (B, num_keypoints, 3)

        pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])
        batch_indices = torch.arange(clouds.shape[0], device=clouds.device).view(-1, 1).repeat(1, clouds.shape[2]).view(-1).long()
        batch_indices = batch_indices.view(-1, 1)
        raw_points = torch.cat([batch_indices, pts], dim=1)
        # print("keypoints : ", keypoints.shape)
        # print("clouds : ", clouds.shape)
        # print("pts : ", pts.shape)
        # print("batch_indices : ", batch_indices.shape)
        # print("raw_points", raw_points, raw_points.shape)
        # keypoints :  torch.Size([10, 4096, 3])
        # clouds :  torch.Size([10, 4, 55765])
        # pts :  torch.Size([557650, 4])
        # batch_indices :  torch.Size([603430])

        point_features_list = []
        # if 'bev' in self.PFE.FEATURES_SOURCE:
        #     point_bev_features = self.interpolate_from_bev_features(
        #         keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
        #         bev_stride=batch_dict['spatial_features_stride']
        #     )
        #     point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        # print("batch_size", batch_size)
        # print("new_xyz", new_xyz.shape)
        # print("new_xyz_batch_cnt", new_xyz_batch_cnt, new_xyz_batch_cnt.shape)
        # batch_size 10
        # new_xyz torch.Size([40960, 3])
        # new_xyz_batch_cnt torch.Size([10])

        batch_dict = self.meanVEF(clouds)
        batch_dict['batch_size'] = batch_size
        batch_dict = self.voxel_backbone_8x(batch_dict)

        if 'raw_points' in self.PFE.FEATURES_SOURCE:
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            # print("raw_points : ", xyz.shape)
            # print("xyz : ", xyz.shape)
            # print("xyz_batch_cnt : ", xyz_batch_cnt.shape)
            # raw_points :  torch.Size([603430, 3])
            # xyz :  torch.Size([603430, 3])
            # xyz_batch_cnt :  torch.Size([10])

            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:,0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            # print("raw pooled_features : ", pooled_features, pooled_features.shape)
            # print("raw xyz : ", xyz, xyz.shape)
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            # print(k, src_name)
            # print("pooled_features : ", pooled_features, pooled_features.shape)
            # print("voxel xyz : ", xyz, xyz.shape)

            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2)
        # print("point_features 0:  ", point_features.shape)  #: torch.Size([10, 4096, 32])

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        # batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        # batch_dict['point_features'] = point_features  # (BxN, C)
        # batch_dict['point_coords'] = point_coords[:, :3] # (BxN, 4)

        point_features = point_features.view(batch_size, self.PFE.NUM_KEYPOINTS, -1)
        point_xyz = point_coords[:, 1:4].view(batch_size, -1, self._point_dim)

        # print("point_features : ", point_features.shape)
        # print("point_xyz : ", point_xyz, point_xyz.shape)
        # point_features :  torch.Size([10, 1024, 128])
        # point_xyz :  torch.Size([10, 1024, 3])


        clouds = torch.cat((point_xyz.transpose(1, 2), point_features.transpose(1, 2)), dim=1)
        # print("clouds : ", clouds.shape)
        

        return clouds
        # return batch_dict
    #         # print("xyz : ", xyz.shape)
    #         # print("features : ", features.shape)
    #         # print("clouds : ", clouds.shape)
    #         # xyz :  torch.Size([10, 1024, 3])
    #         # features :  torch.Size([10, 64, 1024])
    #         # clouds :  torch.Size([10, 67, 1024])
    #         # xyz :  torch.Size([10, 1024, 3])
    #         # features :  torch.Size([10, 64, 1024])
    #         # clouds :  torch.Size([10, 67, 1024])

    #         return clouds


class GroupingModule(abc.ABC, nn.Module):
    """Abstract base class for point cloud grouping."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GlobalGrouping(GroupingModule):
    """Group points over the whole point cloud."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def _prepare_batch(cloud: torch.Tensor) -> torch.Tensor:
        pts = cloud.transpose(1, 2).contiguous().view(-1, cloud.shape[1])
        return pts

    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # prepare data
        pts0 = self._prepare_batch(cloud0)
        pts1 = self._prepare_batch(cloud1)

        # select all points from pts2 for each point of pts1
        idx0 = pts0.new_empty((pts0.shape[0], 1), dtype=torch.long)
        torch.arange(pts0.shape[0], out=idx0)
        idx0 = idx0.repeat(1, cloud1.shape[2])

        idx1 = pts1.new_empty((1, pts1.shape[0]), dtype=torch.long)
        torch.arange(pts1.shape[0], out=idx1)
        idx1 = idx1.view(cloud1.shape[0], -1).repeat(1, cloud0.shape[2]).view(idx0.shape)

        group_index = torch.stack((idx0, idx1))

        # get group data [group, point_dim, group points] and subtract sample (center) pos
        group_pts0 = pts0[group_index[0, ...]]
        group_pts1 = pts1[group_index[1, ...]]

        return pts0, pts1, group_pts0, group_pts1


class KnnGrouping(GroupingModule):
    """Group points with k nearest neighbor."""
    def __init__(self, point_dim: int, k: int):
        super().__init__()
        self._point_dim = point_dim
        self._k = k

    @staticmethod
    def _prepare_batch(clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # print("========Glocal grouping LAYER TEST ==========\n")
        # print("clouds : ", clouds.shape)
        pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])
        # print("pts : ", pts.shape)
        batch = pts.new_empty(clouds.shape[0], dtype=torch.long)
        # print("batch : ", batch.shape)
        torch.arange(clouds.shape[0], out=batch)
        batch = batch.view(-1, 1).repeat(1, clouds.shape[2]).view(-1)
        # print("batch 2: ", batch.shape)

        # print("========Glocal grouping LAYER TEST ==========\n")

        # clouds :  torch.Size([5, 67, 512])
        # pts :  torch.Size([2560, 67])
        # batch :  torch.Size([5])
        # batch 2:  torch.Size([2560]) --> final batch

        return pts, batch

    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # prepare data
        pts0, batch0 = self._prepare_batch(cloud0)
        pts1, batch1 = self._prepare_batch(cloud1)

        # select k nearest points from pts1 for each point of pts0
        group_index = knn(pts1[:, :self._point_dim].contiguous().detach(), #xyz from pts1
                          pts0[:, :self._point_dim].contiguous().detach(), #xyz from pts0
                          k=self._k, batch_x=batch1, batch_y=batch0)
        # print("group_index : ", group_index.shape)
        group_index = group_index.view(2, pts0.shape[0], self._k)
        # print("group_index2 : ", group_index.shape)

        # get group data [group, point_dim, group points] and subtract sample (center) pos
        group_pts0 = pts0[group_index[0, ...]]
        group_pts1 = pts1[group_index[1, ...]]
        # print("group_pts0 : ", group_pts0.shape)

        # print("========Glocal grouping LAYER TEST ==========\n")

        # group_index :  torch.Size([2, 51200])
        # group_index2 :  torch.Size([2, 2560, 20])
        # pts :  torch.Size([2560, 67])
        # group_pts0 :  torch.Size([2560, 20, 67])

        return pts0, pts1, group_pts0, group_pts1


class MotionEmbeddingBase(nn.Module):
    """Base implementation for motion embedding to merge point clouds."""
    _grouping: GroupingModule

    def __init__(self, input_dim: int, point_dim: int, k: int, radius: float, mlp: List[int],
                append_features: bool = True, batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        # print("========Embedding LAYER TEST ==========\n")

        self._point_dim = point_dim
        self._append_features = append_features
        self._k = k
        self._input_dim = input_dim
        # self.transformer_dict = transformer
        print("point_dim + 2 * (input_dim - point_dim) : " , point_dim + 2 * (input_dim - point_dim))

        # self._transformer = Transformer(transformer.NUM_KEYPOINTS, point_dim + 2 * (input_dim - point_dim), transformer.NUM_LAYER, transformer.NUM_HEAD)
        # self._transformer = ModelTransformer(transformer, encoder_dim=transformer.enc_dim, decoder_dim=transformer.dec_dim)

        if k == 0:
            self._grouping = GlobalGrouping()
        else:
            self._grouping = KnnGrouping(point_dim, k)

        if self._append_features:
            mlp_layers = [point_dim + 2 * (input_dim - point_dim), *mlp]
        else:
            mlp_layers = [input_dim, *mlp]
        # print("mlp_layers : ", mlp_layers) # [3 + 64 x 2, 128, 128, 256]
        self._conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)

        pts_diff_mlp_layers = [point_dim , *mlp]
        self._pts_diff_conv = Conv1dMultiLayer(pts_diff_mlp_layers, batch_norm=batch_norm)

        self._radius = radius
        print("self._point_dim*2 + self._conv.output_dim() : " , self._point_dim*2 + self._conv.output_dim())

        # print("========Embedding LAYER TEST ==========\n")

    def output_dim(self) -> int:
        return self._point_dim + self._conv.output_dim() #2 * (self._input_dim - self._point_dim)

    def forward(self, clouds0: torch.Tensor, clouds1: torch.Tensor) -> torch.Tensor:
        # print("========Embedding LAYER TEST ==========\n")
        # group
        pts0, pts1, group_pts0, group_pts1 = self._grouping(clouds0, clouds1)

        # merge
        pos_diff = group_pts1[:, :, :self._point_dim] - group_pts0[:, :, :self._point_dim]
        origin_pos_diff = pos_diff.reshape(pts0.shape[0], -1)
        # print("pts0 : " , pts0, pts0.shape)
        # print("pts1 : " , pts1, pts1.shape)
        # print("pos_diff : " , pos_diff.shape)
        # new_pos_diff = pos_diff.squeeze(1)
        # print("new_pos_diff : " , new_pos_diff.shape)
        # print("group_pts0[:, :, self._point_dim:] : " , group_pts0[:, :, self._point_dim:].shape)
        # print("group_pts1[:, :, self._point_dim:] : " , group_pts1[:, :, self._point_dim:].shape)
        # print("group_pts0 : " , group_pts0.shape)
        # print("group_pts1 : " , group_pts1.shape)

        if self._append_features:
            merged = torch.cat((pos_diff, group_pts0[:, :, self._point_dim:], group_pts1[:, :, self._point_dim:]),
                               dim=2)
        else:
            merged = torch.cat((pos_diff, group_pts1[:, :, self._point_dim:] - group_pts0[:, :, self._point_dim:]),
                               dim=2)

        # run pointnet
        # print("merged : " , merged.shape)
        merged = merged.transpose(1, 2)
        # print("merged 2: " , merged.shape)
        merged_feat = self._conv(merged)
        # print("merged_feat : " , merged_feat.shape)

        # radius
        if self._radius > 0.0:
            pos_diff_norm = torch.norm(pos_diff, dim=2)
            mask = pos_diff_norm >= self._radius
            merged_feat.masked_scatter_(mask.unsqueeze(1), merged_feat.new_zeros(merged_feat.shape))

        # print("merged_feat 2: " , merged_feat.shape)
        feat, _ = torch.max(merged_feat, dim=2)
        # print("feat : " , feat.shape)

        # append features to pts1 pos and separate batches
        out = torch.cat((pts0[:, :self._point_dim], feat), dim=1) #origin out
        # print("out : " , out.shape)
        # out = torch.cat((new_pos_diff, feat), dim=1) #pos diff positional encoder

        # pts diff 
        # use_pts_diff = False
        # if use_pts_diff:
        #     pos_diff = pos_diff.transpose(1, 2)
        #     print("pos_diff 2: " , pos_diff.shape)
        #     new_pos_diff, _ = torch.max(pos_diff, dim=2)
        #     print("pos_diff : ", pos_diff.shape)
        #     print("new_pos_diff : ", new_pos_diff.shape)
        #     out = torch.cat((new_pos_diff, feat), dim=1) #test with diff

        out = out.view(clouds0.shape[0], -1, out.shape[1]).transpose(1, 2).contiguous()
        # print("(self._input_dim - self._point_dim) : ", (self._input_dim - self._point_dim))
        # print("self._conv.output_dim()", self._conv.output_dim())
        # print("out2 : " , out.shape)

        # return transformer_output

        # pos_diff :  torch.Size([2560, 20, 3])
        # group_pts0[:, :, self._point_dim:] :  torch.Size([2560, 20, 64])
        # group_pts1[:, :, self._point_dim:] :  torch.Size([2560, 20, 64])
        # merged :  torch.Size([2560, 20, 131])
        # merged 2:  torch.Size([2560, 131, 20])
        # merged_feat :  torch.Size([2560, 256, 20])
        # feat :  torch.Size([2560, 256])
        # out :  torch.Size([2560, 259])
        # out2 :  torch.Size([5, 259, 512]) batch, 3+2c , keypoints


        # out2 :  torch.Size([2, 259, 1024])
        # xyz torch.Size([2, 1024, 3])
        # feature torch.Size([2, 256, 1024])
        # enc_pos :  torch.Size([1024, 2, 256])
        # query_embed :  torch.Size([256, 2, 256])
        # dec_features :  torch.Size([4, 256, 2, 256])
        # dec_feature :  torch.Size([2, 256, 1024])
        # xyz torch.Size([2, 3, 1024])
        # trans_out torch.Size([2, 259, 1024])

        return out


class MotionEmbedding(DLOPVTModule):
    """Motion embedding for point cloud batch with sorting [template1, template2, ..., source1, source2, ...]."""
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._embedding = MotionEmbeddingBase(**kwargs)

    def output_dim(self):
        return self._embedding.output_dim()

    def forward(self, clouds: torch.Tensor) -> torch.Tensor:
        batch_dim = int(clouds.shape[0] / 2)

        return self._embedding(clouds[:batch_dim, ...],
                               clouds[batch_dim:, ...])

class TransformerModule(abc.ABC, nn.Module):
    """Abstract base class for 3d points transformer."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, cloud0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ModelTransformer(TransformerModule):
    """
    Main ModelTransformer model. Consists of the following learnable sub-models
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        transformer_dict,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        num_queries=256,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dict.enc_dim,
            nhead=transformer_dict.enc_nhead,
            dim_feedforward=transformer_dict.enc_ffn_dim,
            dropout=transformer_dict.enc_dropout,
            activation=transformer_dict.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_dict.enc_nlayers
        )
        self.encoder = encoder


        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_dict.dec_dim,
            nhead=transformer_dict.dec_nhead,
            dim_feedforward=transformer_dict.dec_ffn_dim,
            dropout=transformer_dict.dec_dropout,
        )
        decoder = TransformerDecoder(
            decoder_layer, num_layers=transformer_dict.dec_nlayers, return_intermediate=True
        )

        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.num_queries = num_queries
        self.transformer_dict = transformer_dict

        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=transformer_dict.mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        self.pose_head = mlp_func(output_dim=256)
        self.trans_head = mlp_func(output_dim=3)
        self.rot_head = mlp_func(output_dim=3)


    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = pointnet2_stack_utils.furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, pre_enc_xyz, pre_enc_features):
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        return enc_xyz, enc_features

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)


        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        pose_logits = self.pose_head(box_features).transpose(1, 2)
        trans_normalized = self.trans_head(box_features).transpose(1, 2) 
        rot_logits = self.rot_head(box_features).transpose(1, 2) 


        # reshape outputs to num_layers x batch x nqueries x noutput
        pose_logits = pose_logits.reshape(num_layers, batch, num_queries, -1)
        trans_normalized = trans_normalized.reshape(num_layers, batch, num_queries, -1)
        rot_logits = rot_logits.reshape(num_layers, batch, num_queries, -1)

        outputs = []
        for l in range(num_layers):

            
            trans_normalized_, trans_unnormalized_ = self.compute_predicted_center(
                trans_normalized[l], query_xyz, point_cloud_dims
            )            

            box_prediction = {
                "pose_logits": pose_logits[l],
                "trans_normalized": trans_normalized_.contiguous(),
                "trans_unnormalized": trans_unnormalized_,
                "rot_logits": rot_logits[l],
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]
        return outputs
        # return {
        #     "outputs": outputs,  # output from last layer of decoder
        #     "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        # }

    def forward(self, pre_enc_xyz, pre_enc_features):
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        enc_xyz, enc_features= self.run_encoder(pre_enc_xyz, pre_enc_features)
        # print("pre_enc_xyz : ",pre_enc_xyz.shape)
        # print("pre_enc_features : ",pre_enc_features.shape) 
        # print("enc_xyz : ",enc_xyz.shape)
        # print("enc_features : ", enc_features, enc_features.shape) 

        # test here. we dont need encoder to decoder projection. length is same.
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3
        # print("enc_features 2 : ", enc_features, enc_features.shape) 

        # min_max_xyz = pre_enc_xyz.reshape(-1, pre_enc_xyz.shape[2]).contiguous()
        # print(pre_enc_xyz.shape)
        # Find the minimum and maximum value at each position in the tensor
        min_xyz = pre_enc_xyz.min(dim=1)[0]
        max_xyz = pre_enc_xyz.max(dim=1)[0]
        # print(min_xyz)
        # print(max_xyz)

        point_cloud_dims = [
            min_xyz,
            max_xyz,
        ]

        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)

        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        # print("enc_pos : ", enc_pos.shape)
        # print("query_embed : ", query_embed.shape)

        tgt = torch.zeros_like(query_embed)
        dec_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        # print("dec_features : ", dec_features.shape)

        # #  dec_features: num_layers x num_queries x batch x channel

        # # box_features change to batch x (num_layers x channel) x num_queries
        # dec_features = dec_features.permute(2, 0, 3, 1)
        # batch, num_layers, channel, num_queries = (
        #     dec_features.shape[0],
        #     dec_features.shape[1],
        #     dec_features.shape[2],
        #     dec_features.shape[3],
        # )
        # dec_features = dec_features.reshape( batch, num_layers * channel, num_queries)
        # print("dec_features : ", dec_features)

        prediction = self.get_box_predictions(
            query_xyz, point_cloud_dims, dec_features
        )
        return prediction

class TransformerBase(nn.Module):
    _transformer: ModelTransformer
    def __init__(self, input_dim: int, point_dim: int,
                transformer_dict: Dict, append_features: bool = True, batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        self._point_dim = point_dim
        self._append_features = append_features
        self._input_dim = input_dim
        self.transformer_dict = transformer_dict
        self._transformer = ModelTransformer(transformer_dict, encoder_dim=transformer_dict.enc_dim, decoder_dim=transformer_dict.dec_dim)

    def output_dim(self) -> int:
        return self._input_dim - self._point_dim

    def forward(self, embedded_flow: torch.Tensor) -> torch.Tensor:
        # # TODO : self-attention layer here
        trans_xyz = embedded_flow[:, :self._point_dim,:].transpose(1,2).contiguous()
        trans_feature = embedded_flow[:, self._point_dim:,:].contiguous()

        # print("trans_xyz : ", trans_xyz)
        # print("trans_feature : ", trans_feature)
        # print("target_xyz : ", target_xyz)

        # print("xyz", trans_xyz.shape)
        # print("feature", trans_feature.shape)
        # xyz torch.Size([2, 4096, 3])
        # feature torch.Size([2, 256, 4096])

        # # xyz: batch x npoints x 3
        # # features: batch x channel x npoints
        # # print(trans_xyz)
        prediction = self._transformer(trans_xyz, trans_feature)
        # print("pose_logits : ", prediction["pose_logits"].shape)
        # print("trans_normalized : ", prediction["trans_normalized"].shape)
        # print("trans_unnormalized : ", prediction["trans_unnormalized"].shape)
        # print("rot_logits : ", prediction["rot_logits"].shape)
        # print(torch.cat((prediction["trans_normalized"], prediction["rot_logits"]), dim=2).shape)
        # dec_feature = dec_feature.transpose(1, 2).contiguous()
        # trans_xyz = trans_xyz.transpose(1,2).contiguous()
        # print("xyz", trans_xyz.shape)

        # trans_out = torch.cat((trans_xyz, dec_feature), dim=1)
        # print("trans_out", trans_out.shape)

        return prediction["pose_logits"]
        # return torch.cat((prediction["pose_logits"], prediction["trans_normalized"], prediction["rot_logits"]), dim=2)



class Transformer(DLOPVTModule):
    """Transformer encoder-decoder block for attention mechanism [embedded_flow]."""
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._transformer = TransformerBase(**kwargs)
    def output_dim(self):
        return self._transformer.output_dim()

    def forward(self, clouds: torch.Tensor) -> torch.Tensor:
        return self._transformer(clouds)


class OutputSimple(DLOPVTModule):
    """Simple output module with mini-PointNet and fully connected layers."""
    def __init__(self, input_dim: int, label_type: LabelType, mlp: List[int], linear: List[int],
                 batch_norm: bool = False, dropout: bool = False, **_kwargs: Any):
        super().__init__()
        self._label_type = label_type
        # print("========Output LAYER TEST ==========\n")
        print(label_type)
        # layers
        mlp_layers = [input_dim, *mlp]
        # print("mlp_layers : ", mlp_layers)
        self.conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)
        # print("output_dim : ", self.conv.output_dim())
        self.linear = LinearMultiLayer(linear, batch_norm=batch_norm,
                                       dropout_keep=dropout, dropout_last=True)
        self.output = nn.Linear(linear[-1], label_type.dim, bias=True)

        # init weights
        nn.init.xavier_uniform_(self.output.weight)

        # bias
        if label_type.bias is not None:
            for i, v in enumerate(label_type.bias):
                self.output.bias.data[i] = v

    def output_dim(self) -> int:
        return self._label_type.dim

    def _output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self._label_type == LabelType.POSE3D_QUAT:
            x[:, 3] = torch.sigmoid(x[:, 3])
            x[:, 4:] = torch.tanh(x[:, 4:])
        elif self._label_type == LabelType.POSE3D_DUAL_QUAT:
            x[:, 0] = torch.sigmoid(x[:, 0])
            x[:, 1:4] = torch.tanh(x[:, 1:4])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply pointnet to get final feature vector
        # print("========Output LAYER TEST ==========\n")
        # print("0 : ", x.shape)
        x = self.conv(x)
        # print("1 : ", x.shape)

        x, _ = torch.max(x, dim=2)
        # print("2 : ", x.shape)

        # output shape
        x = self.linear(x)
        # print("3 : ", x.shape)

        x = self.output(x)

        # print("4 : ", x.shape)
        x = self._output_activation(x)


        # print("5 : ", x.shape)
        # 0 :  torch.Size([5, 259, 512])
        # 1 :  torch.Size([5, 1024, 512])
        # 2 :  torch.Size([5, 1024])
        # 3 :  torch.Size([5, 256])
        # 4 :  torch.Size([5, 8])
        # 5 :  torch.Size([5, 8])

        return x


class TransformLossCalculation(nn.Module):
    """Transform loss in network."""
    def __init__(self, label_type: LabelType, p: int, reduction: Optional[str] = 'mean'):
        super().__init__()
        self._label_type = label_type
        self._p = p
        self._reduction = reduction

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor)\
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # translation and rotation loss
        t_loss = trans_loss(y_pred, y, self._label_type, p=self._p, reduction='none')
        r_loss = rot_loss(y_pred, y, self._label_type, p=self._p, reduction='none')
        tr_loss = torch.cat((t_loss, r_loss), dim=1)
        if self._reduction == 'mean':
            tr_loss = torch.mean(tr_loss, dim=0)

            # check nan
            if torch.isnan(tr_loss[0]) or torch.isinf(tr_loss[0]):
                raise RuntimeError("TransformLoss: translation loss is nan or inf:\ny_pred = \n{}\ny = \n{}"
                                   .format(y_pred, y))
            if torch.isnan(tr_loss[1]) or torch.isinf(tr_loss[1]):
                raise RuntimeError("TransformLoss: rotation loss is nan or inf:\ny_pred = \n{}\ny = \n{}"
                                   .format(y_pred, y))

            return tr_loss[0], tr_loss[1]

        else:
            # check nan
            if torch.any(torch.isnan(tr_loss[:, 0])) or torch.any(torch.isinf(tr_loss[:, 0])):
                raise RuntimeError("TransformLoss: translation loss is nan or inf:\ny_pred = \n{}\ny = \n{}"
                                   .format(y_pred, y))
            if torch.any(torch.isnan(tr_loss[:, 1])) or torch.any(torch.isinf(tr_loss[:, 1])):
                raise RuntimeError("TransformLoss: rotation loss is nan or inf:\ny_pred = \n{}\ny = \n{}"
                                   .format(y_pred, y))

            return tr_loss


class DLOPVTLoss(DLOPVTModule, metaclass=abc.ABCMeta):
    """Abstract base class for loss calculation modules."""
    def __init__(self):
        super().__init__()

    def output_dim(self) -> int:
        return 1

    @abc.abstractmethod
    def get_weights(self) -> Dict:
        raise NotImplementedError


class TransformLoss(DLOPVTLoss):
    """Weighted transform loss with fixed weights."""
    def __init__(self, label_type: LabelType, p: int, sx: float, sq: float, **_kwargs: Any):
        super().__init__()
        self._transform_loss = TransformLossCalculation(label_type, p, reduction='mean')
        self._sx = sx
        self._sq = sq

    def get_weights(self) -> Dict:
        return {}

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # position and quaternion loss
        p_loss, q_loss = self._transform_loss(y_pred, y)

        # weighted loss
        loss = p_loss * self._sx + q_loss * self._sq

        return loss


class TransformUncertaintyLoss(DLOPVTLoss):
    """Weighted transform loss with epistemic uncertainty."""
    def __init__(self, label_type: LabelType, p: int, sx: float, sq: float, **_kwargs: Any):
        super().__init__()
        self._transform_loss = TransformLossCalculation(label_type, p, reduction='mean')
        self._sx = torch.nn.Parameter(torch.Tensor([sx]))
        self._sq = torch.nn.Parameter(torch.Tensor([sq]))

    def get_weights(self) -> Dict:
        return {'sx': self._sx.item(), 'sq': self._sq.item()}

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        p_loss, q_loss = self._transform_loss(y_pred, y)

        # weighted loss
        loss = p_loss * torch.exp(-self._sx) + self._sx + \
            q_loss * torch.exp(-self._sq) + self._sq

        return loss


class AccumulatedLoss(DLOPVTLoss):
    """Accumulated loss of multiple loss types."""
    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.loss_list = torch.nn.ModuleList(modules)

    def get_weights(self) -> Dict:
        weights = {}
        for loss in self.loss_list:
            for key, value in loss.get_weights().items():
                if key in weights:
                    raise RuntimeError("Duplicate loss keys")
                weights[key] = value
        return weights

    def forward(self, *args: Any) -> torch.Tensor:
        loss_values = [loss(*args) for loss in self.loss_list]
        return torch.stack(loss_values, dim=0).sum()


def init_module(cfg: Config, *args: Any, **kwargs: Any) -> DLOPVTModule:
    """Initialize DLOPVTModule from config."""
    return factory(DLOPVTModule, cfg.name, *args, **cfg.params, **kwargs)


def init_loss_module(cfg: Config, label_type: LabelType, *args: Any, **kwargs: Any) -> DLOPVTLoss:
    """Initialize DLOPVTLoss from config."""
    return factory(DLOPVTLoss, cfg.name, *args, label_type=label_type, **cfg.params, **kwargs)


def init_optional_module(cfg: Optional[Config], *args: Any, **kwargs: Any) -> Optional[DLOPVTModule]:
    """Initialize optional DLOPVTModule from config."""
    if cfg is None:
        return None
    else:
        return factory(DLOPVTModule, cfg.name, *args, **cfg.params, **kwargs)


def split_output(output: Any) -> Tuple[Any, Any]:
    """Split network output into main and auxiliary data."""
    if isinstance(output, (list, tuple)):
        assert len(output) == 2
        data = output[0]
        aux = output[1]
    else:
        data = output
        aux = dict()
    return data, aux


class DLOPVT(BaseModel):
    """Main DLOPVT network."""
    _loss_layer: Optional[DLOPVTLoss]

    def __init__(self, input_dim: int, label_type: LabelType, cloud_features: Config,
                 merge: Config, transformer: Config, output: Config, transform: Optional[Config] = None,
                 loss: Optional[Config] = None, **kwargs: Any):
        super().__init__()

        self._input_dim = input_dim

        transform_layer = init_optional_module(transform, input_dim=input_dim, **kwargs)
        transform_layer_output_dim = input_dim if transform_layer is None else transform_layer.output_dim()
        cloud_feat_layer = init_module(cloud_features, input_dim=transform_layer_output_dim, **kwargs)
        merge_layer = init_module(merge, input_dim=cloud_feat_layer.output_dim(), **kwargs)
        transformer_layer = init_module(transformer, input_dim=merge_layer.output_dim(), **kwargs)
        output_layer = init_module(output, input_dim=transformer_layer.output_dim(), label_type=label_type, **kwargs)
        # print("transform_layer_output_dim : ", transform_layer_output_dim)
        # print("cloud_feat_layer_output_dim : ", cloud_feat_layer.output_dim())
        # print("merge_layer_output_dim : ", merge_layer.output_dim())
        # print("transformer_layer : ", transformer_layer.output_dim())
        # print("output_layer_output_dim : ", output_layer.output_dim())
        # transform_layer_output_dim :  4
        # cloud_feat_layer_output_dim :  67
        # merge_layer_output_dim :  259
        # output_layer_output_dim :  8

        if transform_layer is None:
            self._cloud_layers = nn.Sequential(cloud_feat_layer)
            self._merge_layers = nn.Sequential(merge_layer, transformer_layer, output_layer)
        else:
            self._cloud_layers = nn.Sequential(transform_layer, cloud_feat_layer)
            self._merge_layers = nn.Sequential(merge_layer, transformer_layer, output_layer)

        if loss is not None:
            if isinstance(loss, list):
                loss_modules = [init_loss_module(loss_cfg, label_type, **kwargs) for loss_cfg in loss]
                self._loss_layer = AccumulatedLoss(loss_modules)
                print("LOSS MODULE : 1")
            else:
                self._loss_layer = init_loss_module(loss, label_type, **kwargs)
                print("LOSS MODULE : 2")
        else:
            self._loss_layer = None
            print("LOSS MODULE : 3")

    def get_input_dim(self) -> int:
        return self._input_dim

    def has_loss(self) -> bool:
        return self._loss_layer is not None

    def get_loss_weights(self) -> Dict:
        if self._loss_layer is not None:
            return self._loss_layer.get_weights()
        else:
            return {}

    def forward(self, x: torch.Tensor, is_feat: bool = False, m: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None, debug: bool = False)\
            -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        # cloud features
        if not is_feat:
            x = self.cloud_features(x, m=m)

        # merge
        model_output = self._merge_layers(x)
        y_pred, model_aux = split_output(model_output)

        # loss
        if self._loss_layer is not None and y is not None:
            loss_output = self._loss_layer(y_pred, y, **model_aux)
            loss, loss_aux = split_output(loss_output)
            debug_output = {**model_aux, **loss_aux, 'x_aug': x} if debug else None
        else:
            loss = None
            debug_output = None

        # print("y_pred: ", y_pred ) 
        # print("loss: ", loss ) 
        # print("debug_output: ", debug_output ) 
        

        return y_pred, loss, debug_output

    def cloud_features(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
        # apply transforms
        if m is not None:
            dim = m.shape[-1] - 1
            x[:, :, :dim] = tgm.transform_points(m, x[:, :, :dim])

        # format clouds for pointnet2
        x = x.transpose(1, 2)

        # forward pass
        x = self._cloud_layers(x)
        return x

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_cluster import knn
import torchgeometry as tgm
from pointnet2 import PointnetSAModuleMSG
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch, sample_points_with_roi
from pcdet.utils import common_utils

from ..config.config import Config
from ..data.labels import LabelType
from ..utils.factory import factory
from ..utils.metrics import trans_loss, rot_loss

from .base import BaseModel
from .helper import Conv1dMultiLayer, LinearMultiLayer
# from pcdet.models.backbones_3d.pfe import 

# my note :
# change point feature extractor -> voxel feature extractor (same as PV_RCNN network)

class PVNAVIModule(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for PVNAVI modules."""
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

# class VoxelSetAbstraction(PVNAVIModule):
#     def __init__(self, input_dim: int, point_dim: int, mlps: List[List[List[int]]],
#                  npoint: List[int], radii: List[List[float]], nsamples: List[List[int]], PFE: Dict, 
#                  batch_norm: bool = False, **_kwargs: Any):
#         super().__init__()
#         assert point_dim == 3
#         assert len(mlps) == len(npoint) == len(radii) == len(nsamples)
#         assert 0 < len(mlps) <= 2

#         self._point_dim = point_dim
#         input_feat_dim = input_dim - self._point_dim
#         self._output_feat_dim = int(np.sum([x[-1] for x in mlps[-1]]))
#         self.PFE = PFE
#         self.voxel_size = PFE.VOXEL_SIZE
#         self.point_cloud_range = PFE.POINT_CLOUD_RANGE

#         SA_cfg = PFE.SA_LAYER

#         self.SA_layers = nn.ModuleList()
#         self.SA_layer_names = []
#         self.downsample_times_map = {}
#         c_in = 0
#         for src_name in PFE.FEATURES_SOURCE:
#             if src_name in ['bev', 'raw_points']:
#                 continue
#             self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

#             if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
#                 input_channels = SA_cfg[src_name].MLPS[0][0] \
#                     if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
#             else:
#                 input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

#             cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
#                 input_channels=input_channels, config=SA_cfg[src_name]
#             )
#             self.SA_layers.append(cur_layer)
#             self.SA_layer_names.append(src_name)

#             c_in += cur_num_c_out

#         # if 'bev' in PFE.FEATURES_SOURCE:
#         #     c_bev = num_bev_features
#         #     c_in += c_bev

#         if 'raw_points' in PFE.FEATURES_SOURCE:
#             self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
#                 input_channels=PFE.NUM_POINT_FEATURES - 3, config=SA_cfg['raw_points']
#             )

#             c_in += cur_num_c_out

#         self.vsa_point_feature_fusion = nn.Sequential(
#             nn.Linear(c_in, PFE.NUM_OUTPUT_FEATURES, bias=False),
#             nn.BatchNorm1d(PFE.NUM_OUTPUT_FEATURES),
#             nn.ReLU(),
#         )
#         self.num_point_features = PFE.NUM_OUTPUT_FEATURES
#         self.num_point_features_before_fusion = c_in

#     def output_dim(self) -> int:
#         return 3 + self._output_feat_dim

#     def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
#         """
#         Args:
#             keypoints: (N1 + N2 + ..., 4)
#             bev_features: (B, C, H, W)
#             batch_size:
#             bev_stride:

#         Returns:
#             point_bev_features: (N1 + N2 + ..., C)
#         """
#         x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
#         y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

#         x_idxs = x_idxs / bev_stride
#         y_idxs = y_idxs / bev_stride

#         point_bev_features_list = []
#         for k in range(batch_size):
#             bs_mask = (keypoints[:, 0] == k)

#             cur_x_idxs = x_idxs[bs_mask]
#             cur_y_idxs = y_idxs[bs_mask]
#             cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
#             point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
#             point_bev_features_list.append(point_bev_features)

#         point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
#         return point_bev_features

#     # def sectorized_proposal_centric_sampling(self, roi_boxes, points):
#     #     """
#     #     Args:
#     #         roi_boxes: (M, 7 + C)
#     #         points: (N, 3)

#     #     Returns:
#     #         sampled_points: (N_out, 3)
#     #     """

#     #     sampled_points, _ = sample_points_with_roi(
#     #         rois=roi_boxes, points=points,
#     #         sample_radius_with_roi=PFE.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
#     #         num_max_points_of_part=PFE.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
#     #     )
#     #     sampled_points = sector_fps(
#     #         points=sampled_points, num_sampled_points=PFE.NUM_KEYPOINTS,
#     #         num_sectors=PFE.SPC_SAMPLING.NUM_SECTORS
#     #     )
#     #     return sampled_points

#     def get_sampled_points(self, batch_dict):
#         """
#         Args:
#             batch_dict:

#         Returns:
#             keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
#         """
#         batch_size = batch_dict['batch_size']
#         if self.PFE.POINT_SOURCE == 'raw_points':
#             src_points = batch_dict['points'][:, 1:4]
#             batch_indices = batch_dict['points'][:, 0].long()
#         elif self.PFE.POINT_SOURCE == 'voxel_centers':
#             src_points = common_utils.get_voxel_centers(
#                 batch_dict['voxel_coords'][:, 1:4],
#                 downsample_times=1,
#                 voxel_size=self.voxel_size,
#                 point_cloud_range=self.point_cloud_range
#             )
#             batch_indices = batch_dict['voxel_coords'][:, 0].long()
#         else:
#             raise NotImplementedError
#         keypoints_list = []
#         for bs_idx in range(batch_size):
#             bs_mask = (batch_indices == bs_idx)
#             sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
#             if self.PFE.SAMPLE_METHOD == 'FPS':
#                 cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
#                     sampled_points[:, :, 0:3].contiguous(), self.PFE.NUM_KEYPOINTS
#                 ).long()

#                 if sampled_points.shape[1] < self.PFE.NUM_KEYPOINTS:
#                     times = int(self.PFE.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
#                     non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
#                     cur_pt_idxs[0] = non_empty.repeat(times)[:self.PFE.NUM_KEYPOINTS]

#                 keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

#             # elif self.PFE.SAMPLE_METHOD == 'SPC':
#             #     cur_keypoints = self.sectorized_proposal_centric_sampling(
#             #         roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0]
#             #     )
#             #     bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
#             #     keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
#             else:
#                 raise NotImplementedError

#             keypoints_list.append(keypoints)

#         keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
#         if len(keypoints.shape) == 3:
#             batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
#             keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)

#         return keypoints

#     @staticmethod
#     def aggregate_keypoint_features_from_one_source(
#             batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
#             filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
#     ):
#         """

#         Args:
#             aggregate_func:
#             xyz: (N, 3)
#             xyz_features: (N, C)
#             xyz_bs_idxs: (N)
#             new_xyz: (M, 3)
#             new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

#             filter_neighbors_with_roi: True/False
#             radius_of_neighbor: float
#             num_max_points_of_part: int
#             rois: (batch_size, num_rois, 7 + C)
#         Returns:

#         """
#         xyz_batch_cnt = xyz.new_zeros(batch_size).int()
#         if filter_neighbors_with_roi:
#             point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
#             point_features_list = []
#             for bs_idx in range(batch_size):
#                 bs_mask = (xyz_bs_idxs == bs_idx)
#                 _, valid_mask = sample_points_with_roi(
#                     rois=rois[bs_idx], points=xyz[bs_mask],
#                     sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
#                 )
#                 point_features_list.append(point_features[bs_mask][valid_mask])
#                 xyz_batch_cnt[bs_idx] = valid_mask.sum()

#             valid_point_features = torch.cat(point_features_list, dim=0)
#             xyz = valid_point_features[:, 0:3]
#             xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
#         else:
#             for bs_idx in range(batch_size):
#                 xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

#         pooled_points, pooled_features = aggregate_func(
#             xyz=xyz.contiguous(),
#             xyz_batch_cnt=xyz_batch_cnt,
#             new_xyz=new_xyz,
#             new_xyz_batch_cnt=new_xyz_batch_cnt,
#             features=xyz_features.contiguous(),
#         )
#         return pooled_features

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size:
#                 keypoints: (B, num_keypoints, 3)
#                 multi_scale_3d_features: {
#                         'x_conv4': ...
#                     }
#                 points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
#                 spatial_features: optional
#                 spatial_features_stride: optional

#         Returns:
#             point_features: (N, C)
#             point_coords: (N, 4)

#         """
#         keypoints = self.get_sampled_points(batch_dict)

#         point_features_list = []
#         if 'bev' in self.PFE.FEATURES_SOURCE:
#             point_bev_features = self.interpolate_from_bev_features(
#                 keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
#                 bev_stride=batch_dict['spatial_features_stride']
#             )
#             point_features_list.append(point_bev_features)

#         batch_size = batch_dict['batch_size']

#         new_xyz = keypoints[:, 1:4].contiguous()
#         new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
#         for k in range(batch_size):
#             new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

#         if 'raw_points' in self.PFE.FEATURES_SOURCE:
#             raw_points = batch_dict['points']

#             pooled_features = self.aggregate_keypoint_features_from_one_source(
#                 batch_size=batch_size, aggregate_func=self.SA_rawpoints,
#                 xyz=raw_points[:, 1:4],
#                 xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
#                 xyz_bs_idxs=raw_points[:, 0],
#                 new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
#                 filter_neighbors_with_roi=self.PFE.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
#                 radius_of_neighbor=self.PFE.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
#                 rois=batch_dict.get('rois', None)
#             )
#             point_features_list.append(pooled_features)

#         for k, src_name in enumerate(self.SA_layer_names):
#             cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
#             cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()

#             xyz = common_utils.get_voxel_centers(
#                 cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
#                 voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
#             )

#             pooled_features = self.aggregate_keypoint_features_from_one_source(
#                 batch_size=batch_size, aggregate_func=self.SA_layers[k],
#                 xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
#                 new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
#                 filter_neighbors_with_roi=self.PFE.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
#                 radius_of_neighbor=self.PFE.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
#                 rois=batch_dict.get('rois', None)
#             )

#             point_features_list.append(pooled_features)

#         point_features = torch.cat(point_features_list, dim=-1)

#         batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
#         point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

#         batch_dict['point_features'] = point_features  # (BxN, C)
#         batch_dict['point_coords'] = keypoints  # (BxN, 4)

#         # clouds = merge_features(xyz, point_features)
#         return keypoints
#         # return batch_dict


class VoxelSetAbstraction(PVNAVIModule):
    """Set abstraction layer for preprocessing the individual point cloud."""
    def __init__(self, input_dim: int, point_dim: int, mlps: List[List[List[int]]],
                 npoint: List[int], radii: List[List[float]], nsamples: List[List[int]], PFE: Dict, 
                 batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        assert point_dim == 3
        assert len(mlps) == len(npoint) == len(radii) == len(nsamples)
        assert 0 < len(mlps) <= 2

        print("========LAYER TEST ==========\n")
        # print(PFE)
        print("POINT_SOURCE : ", PFE.POINT_SOURCE)
        print("NUM_KEYPOINTS : ", PFE.NUM_KEYPOINTS)
        print("NUM_OUTPUT_FEATURES : ", PFE.NUM_OUTPUT_FEATURES)
        print("SAMPLE_METHOD : ", PFE.SAMPLE_METHOD)
        print("FEATURES_SOURCE : ", PFE.FEATURES_SOURCE)
        print("SA_LAYER : ", PFE.SA_LAYER)
        print("========LAYER TEST ==========\n")

        self._point_dim = point_dim
        input_feat_dim = input_dim - self._point_dim
        self._output_feat_dim = int(np.sum([x[-1] for x in mlps[-1]]))

        sa0_mlps = [[input_feat_dim, *x] for x in mlps[0]]
        self._sa0 = PointnetSAModuleMSG(
            npoint=npoint[0],
            radii=radii[0],
            nsamples=nsamples[0],
            mlps=sa0_mlps,
            use_xyz=True,
            bn=batch_norm
        )

        if len(npoint) == 2:
            sa1_mlps = [[*x] for x in mlps[1]]
            self._sa1 = PointnetSAModuleMSG(
                npoint=npoint[1],
                radii=radii[1],
                nsamples=nsamples[1],
                mlps=sa1_mlps,
                use_xyz=True,
                bn=batch_norm
            )
        else:
            self._sa1 = None

    def output_dim(self) -> int:
        return 3 + self._output_feat_dim

    def forward(self, clouds: torch.Tensor, *_args: Any) -> torch.Tensor:
        # print("input clouds : ", clouds.shape)
        xyz, features = split_features(clouds)
        xyz, features = self._sa0(xyz, features)
        if self._sa1 is not None:
            xyz, features = self._sa1(xyz, features)

        clouds = merge_features(xyz, features)

        # print("xyz : ", xyz.shape)
        # print("features : ", features.shape)
        # print("clouds : ", clouds.shape)
        # xyz :  torch.Size([10, 1024, 3])
        # features :  torch.Size([10, 64, 1024])
        # clouds :  torch.Size([10, 67, 1024])
        # xyz :  torch.Size([10, 1024, 3])
        # features :  torch.Size([10, 64, 1024])
        # clouds :  torch.Size([10, 67, 1024])

        return clouds


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
        pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])
        batch = pts.new_empty(clouds.shape[0], dtype=torch.long)
        torch.arange(clouds.shape[0], out=batch)
        batch = batch.view(-1, 1).repeat(1, clouds.shape[2]).view(-1)
        return pts, batch

    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # prepare data
        pts0, batch0 = self._prepare_batch(cloud0)
        pts1, batch1 = self._prepare_batch(cloud1)

        # select k nearest points from pts1 for each point of pts0
        group_index = knn(pts1[:, :self._point_dim].contiguous().detach(),
                          pts0[:, :self._point_dim].contiguous().detach(),
                          k=self._k, batch_x=batch1, batch_y=batch0)
        group_index = group_index.view(2, pts0.shape[0], self._k)

        # get group data [group, point_dim, group points] and subtract sample (center) pos
        group_pts0 = pts0[group_index[0, ...]]
        group_pts1 = pts1[group_index[1, ...]]

        return pts0, pts1, group_pts0, group_pts1


class MotionEmbeddingBase(nn.Module):
    """Base implementation for motion embedding to merge point clouds."""
    _grouping: GroupingModule

    def __init__(self, input_dim: int, point_dim: int, k: int, radius: float, mlp: List[int],
                 append_features: bool = True, batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        self._point_dim = point_dim
        self._append_features = append_features

        if k == 0:
            self._grouping = GlobalGrouping()
        else:
            self._grouping = KnnGrouping(point_dim, k)

        if self._append_features:
            mlp_layers = [point_dim + 2 * (input_dim - point_dim), *mlp]
        else:
            mlp_layers = [input_dim, *mlp]
        self._conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)
        self._radius = radius

    def output_dim(self) -> int:
        return self._point_dim + self._conv.output_dim()

    def forward(self, clouds0: torch.Tensor, clouds1: torch.Tensor) -> torch.Tensor:
        # group
        pts0, pts1, group_pts0, group_pts1 = self._grouping(clouds0, clouds1)

        # merge
        pos_diff = group_pts1[:, :, :self._point_dim] - group_pts0[:, :, :self._point_dim]

        if self._append_features:
            merged = torch.cat((pos_diff, group_pts0[:, :, self._point_dim:], group_pts1[:, :, self._point_dim:]),
                               dim=2)
        else:
            merged = torch.cat((pos_diff, group_pts1[:, :, self._point_dim:] - group_pts0[:, :, self._point_dim:]),
                               dim=2)

        # run pointnet
        merged = merged.transpose(1, 2)
        merged_feat = self._conv(merged)

        # radius
        if self._radius > 0.0:
            pos_diff_norm = torch.norm(pos_diff, dim=2)
            mask = pos_diff_norm >= self._radius
            merged_feat.masked_scatter_(mask.unsqueeze(1), merged_feat.new_zeros(merged_feat.shape))

        feat, _ = torch.max(merged_feat, dim=2)

        # append features to pts1 pos and separate batches
        out = torch.cat((pts0[:, :self._point_dim], feat), dim=1)
        out = out.view(clouds0.shape[0], -1, out.shape[1]).transpose(1, 2).contiguous()

        return out


class MotionEmbedding(PVNAVIModule):
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


class OutputSimple(PVNAVIModule):
    """Simple output module with mini-PointNet and fully connected layers."""
    def __init__(self, input_dim: int, label_type: LabelType, mlp: List[int], linear: List[int],
                 batch_norm: bool = False, dropout: bool = False, **_kwargs: Any):
        super().__init__()
        self._label_type = label_type

        # layers
        mlp_layers = [input_dim, *mlp]
        self.conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)

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
        x = self.conv(x)
        x, _ = torch.max(x, dim=2)

        # output shape
        x = self.linear(x)
        x = self.output(x)
        x = self._output_activation(x)

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


class PVNAVILoss(PVNAVIModule, metaclass=abc.ABCMeta):
    """Abstract base class for loss calculation modules."""
    def __init__(self):
        super().__init__()

    def output_dim(self) -> int:
        return 1

    @abc.abstractmethod
    def get_weights(self) -> Dict:
        raise NotImplementedError


class TransformLoss(PVNAVILoss):
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


class TransformUncertaintyLoss(PVNAVILoss):
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


class AccumulatedLoss(PVNAVILoss):
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


def init_module(cfg: Config, *args: Any, **kwargs: Any) -> PVNAVIModule:
    """Initialize PVNAVIModule from config."""
    return factory(PVNAVIModule, cfg.name, *args, **cfg.params, **kwargs)


def init_loss_module(cfg: Config, label_type: LabelType, *args: Any, **kwargs: Any) -> PVNAVILoss:
    """Initialize PVNAVILoss from config."""
    return factory(PVNAVILoss, cfg.name, *args, label_type=label_type, **cfg.params, **kwargs)


def init_optional_module(cfg: Optional[Config], *args: Any, **kwargs: Any) -> Optional[PVNAVIModule]:
    """Initialize optional PVNAVIModule from config."""
    if cfg is None:
        return None
    else:
        return factory(PVNAVIModule, cfg.name, *args, **cfg.params, **kwargs)


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


class PVNAVI(BaseModel):
    """Main PVNAVI network."""
    _loss_layer: Optional[PVNAVILoss]

    def __init__(self, input_dim: int, label_type: LabelType, cloud_features: Config,
                 merge: Config, output: Config, transform: Optional[Config] = None,
                 loss: Optional[Config] = None, **kwargs: Any):
        super().__init__()

        self._input_dim = input_dim

        transform_layer = init_optional_module(transform, input_dim=input_dim, **kwargs)
        transform_layer_output_dim = input_dim if transform_layer is None else transform_layer.output_dim()
        cloud_feat_layer = init_module(cloud_features, input_dim=transform_layer_output_dim, **kwargs)
        merge_layer = init_module(merge, input_dim=cloud_feat_layer.output_dim(), **kwargs)
        output_layer = init_module(output, input_dim=merge_layer.output_dim(), label_type=label_type, **kwargs)
        # print("transform_layer_output_dim : ", transform_layer_output_dim)
        # print("cloud_feat_layer_output_dim : ", cloud_feat_layer.output_dim())
        # print("merge_layer_output_dim : ", merge_layer.output_dim())
        # print("output_layer_output_dim : ", output_layer.output_dim())
        # transform_layer_output_dim :  4
        # cloud_feat_layer_output_dim :  67
        # merge_layer_output_dim :  259
        # output_layer_output_dim :  8

        if transform_layer is None:
            self._cloud_layers = nn.Sequential(cloud_feat_layer)
            self._merge_layers = nn.Sequential(merge_layer, output_layer)
        else:
            self._cloud_layers = nn.Sequential(transform_layer, cloud_feat_layer)
            self._merge_layers = nn.Sequential(merge_layer, output_layer)

        if loss is not None:
            if isinstance(loss, list):
                loss_modules = [init_loss_module(loss_cfg, label_type, **kwargs) for loss_cfg in loss]
                self._loss_layer = AccumulatedLoss(loss_modules)
            else:
                self._loss_layer = init_loss_module(loss, label_type, **kwargs)
        else:
            self._loss_layer = None

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

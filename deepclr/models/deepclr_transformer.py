import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_cluster import knn
import torchgeometry as tgm
from pointnet2 import PointnetSAModuleMSG
from functools import partial
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
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
import math
# my note :
# change point feature extractor -> voxel feature extractor (same as PV_RCNN network)

class DEEPCLRTFModule(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for DEEPCLRTF modules."""
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

class SetAbstraction(DEEPCLRTFModule):
    """Set abstraction layer for preprocessing the individual point cloud."""
    def __init__(self, input_dim: int, point_dim: int, mlps: List[List[List[int]]],
                 npoint: List[int], radii: List[List[float]], nsamples: List[List[int]],
                 batch_norm: bool = False, **_kwargs: Any):
        super().__init__()
        assert point_dim == 3
        assert len(mlps) == len(npoint) == len(radii) == len(nsamples)
        assert 0 < len(mlps) <= 2

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
        xyz, features = split_features(clouds)
        # print("xyz : ", xyz.shape)
        # print("features : ", features.shape)

        xyz, features = self._sa0(xyz, features)
        # print("xyz 2: ", xyz.shape)
        # print("features 2: ", features.shape)
        
        if self._sa1 is not None:
            xyz, features = self._sa1(xyz, features)
        set_abstract_cloud = merge_features(xyz, features)
        # print("clouds : ", clouds.shape)
        
        batch_dim = int(set_abstract_cloud.shape[0] / 2)
        output_dict = {
            "src_cloud": clouds[:batch_dim, ...],
            "tgt_cloud": clouds[batch_dim:, ...],
            "set_abstract_cloud": set_abstract_cloud,
        } 
        # return clouds
        return output_dict


            # xyz :  torch.Size([10, 1024, 3])
            # features :  torch.Size([10, 64, 1024])
            # clouds :  torch.Size([10, 67, 1024])

            # xyz :  torch.Size([10, 44477, 3])
            # features :  torch.Size([10, 1, 44477])
            # xyz 2:  torch.Size([10, 1024, 3])
            # features 2:  torch.Size([10, 64, 1024])
            # clouds :  torch.Size([10, 67, 1024])

class GroupingModule(abc.ABC, nn.Module):
    """Abstract base class for point cloud grouping."""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class PositionalEncodingGrouping(GroupingModule):
    """Group points with positional encoding."""
    def __init__(self, point_dim: int, encoding_dim: int):
        super().__init__()
        self._point_dim = point_dim
        self._encoding_dim = encoding_dim

        self._positional_encoding = PositionEmbeddingCoordsSine(
            d_pos=encoding_dim - point_dim, pos_type="fourier", normalize=True
        )
        # coords = torch.rand((32, 1024, 3))  # Example coordinates
        # encoded_coords = self._positional_encoding(coords, num_channels=256)

    def _prepare_batch(self, clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("clouds : ",clouds.shape) #b x D x n
        pts = clouds.permute(0, 2, 1)
        # print("pts : ", pts.shape) #b 
        # pts = clouds.transpose(1, 2).contiguous().view(-1, clouds.shape[1])
        # batch = pts.new_empty(clouds.shape[0], dtype=torch.long)
        # torch.arange(clouds.shape[0], out=batch)
        # batch = batch.view(-1, 1).repeat(1, clouds.shape[2]).view(-1)

        return pts

    def forward(self, cloud0: torch.Tensor, cloud1: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # prepare data
        # print(cloud0.size())
        pts0 = self._prepare_batch(cloud0)
        pts1 = self._prepare_batch(cloud1)

        # pts0 :  torch.Size([2048, 67])
        # pts1 :  torch.Size([2048, 67])
        # batch0 :  torch.Size([2048])
        # batch1 :  torch.Size([2048])
        # return batch x d_pos x npoints embedding

        # print(cloud0[:, :self._point_dim, :].shape)
        # print(cloud1[:, :self._point_dim, :].shape)

        # Find the minimum and maximum value at each position in the tensor
        min_xyz = (cloud0[:, :self._point_dim, :]).min(dim=2)[0]
        max_xyz = (cloud1[:, :self._point_dim, :]).max(dim=2)[0]

        # print(min_xyz)
        # print(max_xyz)

        point_cloud_dims = [
            min_xyz,
            max_xyz,
        ]
        
        # apply positional encoding
        encoded_pts0 = self._positional_encoding(pts0[:, :, :self._point_dim].contiguous(), num_channels=64, input_range=point_cloud_dims)
        encoded_pts1 = self._positional_encoding(pts1[:, :, :self._point_dim].contiguous(), num_channels=64, input_range=point_cloud_dims)

        # encoded_pts0.permute(2, 0, 1)
        # encoded_pts1.permute(2, 0, 1)
        # print("encoded_pts0 : ",encoded_pts0.shape)
        # print("encoded_pts1 : ",encoded_pts1.shape)

        return pts0, pts1, encoded_pts0, encoded_pts1

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

        # print("pts0 : ", pts0.size())
        # print("pts1 : ", pts1.size())
        # print("batch0 : ", batch0.size())
        # print("batch1 : ", batch1.size())
        # pts0 :  torch.Size([2048, 67])
        # pts1 :  torch.Size([2048, 67])
        # batch0 :  torch.Size([2048])
        # batch1 :  torch.Size([2048])

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
        # self.transformer_params = transformer
        print("point_dim + 2 * (input_dim - point_dim) : " , point_dim + 2 * (input_dim - point_dim))
        # self._transformer = Transformer(transformer.NUM_KEYPOINTS, point_dim + 2 * (input_dim - point_dim), transformer.NUM_LAYER, transformer.NUM_HEAD)
        # self._transformer = ModelTransformer(transformer, encoder_dim=transformer.enc_dim, decoder_dim=transformer.dec_dim)
        self._pe = PositionalEncodingGrouping(point_dim, input_dim)

        if k == 0:
            self._grouping = GlobalGrouping()
        else:
            self._grouping = KnnGrouping(point_dim, k)

        pe_mlp_layers = [2 * (input_dim - point_dim), *mlp]

        if self._append_features:
            mlp_layers = [point_dim + 2 * (input_dim - point_dim), *mlp]
        else:
            mlp_layers = [input_dim, *mlp]
        # print("mlp_layers : ", mlp_layers) # [3 + 64 x 2, 128, 128, 256]
        self._conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)
        self._pe_conv = Conv1dMultiLayer(pe_mlp_layers, batch_norm=batch_norm)

        pts_diff_mlp_layers = [point_dim , *mlp]
        self._pts_diff_conv = Conv1dMultiLayer(pts_diff_mlp_layers, batch_norm=batch_norm)

        self._radius = radius

        # print("========Embedding LAYER TEST ==========\n")

    def output_dim(self) -> int:
        print("self._point_dim + self._conv.output_dim() : " , self._point_dim + self._conv.output_dim())
        return self._point_dim + self._conv.output_dim() + self._pe_conv.output_dim()


    def forward(self, clouds0: torch.Tensor, clouds1: torch.Tensor) -> torch.Tensor:
        # print("========Embedding LAYER TEST ==========\n")
        # group
        pts0, pts1, group_pts0, group_pts1 = self._grouping(clouds0, clouds1)
        # print("pts0 : " , pts0.shape)
        # print("pts1 : " , pts1.shape)
        # print("encoded_pts0 : " , encoded_pts0.shape)
        # print("encoded_pts1 : " , encoded_pts1.shape)
        # pts0 :  torch.Size([2, 1024, 67])                                                              
        # pts1 :  torch.Size([2, 1024, 67])                                                              
        # encoded_pts0 :  torch.Size([2, 64, 1024])                                                        
        # encoded_pts1 :  torch.Size([2, 64, 1024]) 

        # print(group_pts0)
        # merge
        pos_diff = group_pts1[:, :, :self._point_dim] - group_pts0[:, :, :self._point_dim]
        # print("pts0 : " , pts0)
        # print("pts1 : " , pts1)
        # print("pos_diff : " , pos_diff.shape)
        # print("group_pts0[:, :, self._point_dim:] : " , group_pts0[:, :, self._point_dim:].shape)
        # print("group_pts1[:, :, self._point_dim:] : " , group_pts1[:, :, self._point_dim:].shape)
        # print("group_pts0 : " , group_pts0.shape)
        # print("group_pts1 : " , group_pts1.shape)
        # pos_diff :  torch.Size([2048, 10, 3])
        # group_pts0[:, :, self._point_dim:] :  torch.Size([2048, 10, 64])
        # group_pts1[:, :, self._point_dim:] :  torch.Size([2048, 10, 64])
        # group_pts0 :  torch.Size([2048, 10, 67])
        # group_pts1 :  torch.Size([2048, 10, 67])
        # out :  torch.Size([2048, 259])

        if self._append_features:
            merged = torch.cat((pos_diff, group_pts0[:, :, self._point_dim:], group_pts1[:, :, self._point_dim:]),
                               dim=2)
        else:
            merged = torch.cat((pos_diff, group_pts1[:, :, self._point_dim:] - group_pts0[:, :, self._point_dim:]),
                               dim=2)

        # merged = torch.cat((group_pts1, group_pts0), dim=1)
        # print("merged 1 : ", merged.shape)
        # merged = merged.permute(2, 1, 0)
        # print("merged 2 : ", merged.shape)

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

        # feat = merged_feat.transpose(1, 2).contiguous().view(-1, merged_feat.shape[1])
        # print("feat : " , feat.shape)
        
        # pts0_new = clouds0.transpose(1, 2).contiguous().view(-1, clouds0.shape[1])
        # print("pts0_new : " , pts0_new.shape)

        # out = torch.cat((pts0_new[:, :self._point_dim], feat), dim=1) #origin out
        # out = out.view(clouds0.shape[0], -1, out.shape[1]).transpose(1, 2).contiguous()

        # PE
        _, _, encoded_pts0, encoded_pts1 = self._pe(clouds0, clouds1)
        pe_merged = torch.cat((encoded_pts0, encoded_pts1), dim=1)
        # print("pe_merged 1 : ", pe_merged.shape)
        pe_merged = pe_merged.permute(2, 1, 0)
        # print("pe_merged 2 : ", pe_merged.shape)
        pe_merged_feat = self._pe_conv(pe_merged)
        # print("pe_merged_feat : ", pe_merged_feat.shape)
        pe_feat = pe_merged_feat.transpose(1, 2).contiguous().view(-1, pe_merged_feat.shape[1])
        # print("pe_feat : " , pe_feat.shape)

        # append features to pts1 pos and separate batches
        out = torch.cat((pts0[:, :self._point_dim], feat, pe_feat), dim=1) #origin out
        # print("out : " , out.shape)
        # merged :  torch.Size([2048, 10, 131])
        # merged 2:  torch.Size([2048, 131, 10])
        # merged_feat :  torch.Size([2048, 256, 10])
        # merged_feat 2:  torch.Size([2048, 256, 10])
        # feat :  torch.Size([2048, 256])
        # out :  torch.Size([2048, 259])

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


class MotionEmbedding(DEEPCLRTFModule):
    """Motion embedding for point cloud batch with sorting [template1, template2, ..., source1, source2, ...]."""
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._embedding = MotionEmbeddingBase(**kwargs)

    def output_dim(self):
        return self._embedding.output_dim()

    # def forward(self, clouds: torch.Tensor) -> torch.Tensor:
    def forward(self, clouds_dict: Dict) -> torch.Tensor:
        clouds_dict_buf = clouds_dict
        merged_clouds = clouds_dict['set_abstract_cloud']
        batch_dim = int(merged_clouds.shape[0] / 2)

        motion_embed_clouds = self._embedding(merged_clouds[:batch_dim, ...],
                               merged_clouds[batch_dim:, ...])
        clouds_dict_buf['motion_embed_clouds'] = motion_embed_clouds
        return clouds_dict_buf

        # self._embedding(merged_clouds[:batch_dim, ...],
        #                        merged_clouds[batch_dim:, ...])


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
        transformer_params,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        num_queries=256,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_params.enc_dim,
            nhead=transformer_params.enc_nhead,
            dim_feedforward=transformer_params.enc_ffn_dim,
            dropout=transformer_params.enc_dropout,
            activation=transformer_params.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_params.enc_nlayers
        )
        self.encoder = encoder

        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_params.dec_dim,
            nhead=transformer_params.dec_nhead,
            dim_feedforward=transformer_params.dec_ffn_dim,
            dropout=transformer_params.dec_dropout,
        )
        decoder = TransformerDecoder(
            decoder_layer, num_layers=transformer_params.dec_nlayers, return_intermediate=True
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
        self.transformer_params = transformer_params

        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=transformer_params.mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        pose_head = mlp_func(output_dim=8)
        trans_head = mlp_func(output_dim=4)
        rot_head = mlp_func(output_dim=4)
        mlp_heads = [
            ("pose_head", pose_head),
            ("trans_head", trans_head),
            ("rot_head", rot_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

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

    def get_box_predictions(self, query_xyz, point_cloud_dims, pose_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            pose_features: num_layers x num_queries x batch x channel
        """
        # pose_features change to (num_layers x batch) x channel x num_queries
        pose_features = pose_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            pose_features.shape[0],
            pose_features.shape[1],
            pose_features.shape[2],
            pose_features.shape[3],
        )
        pose_features = pose_features.reshape(num_layers * batch, channel, num_queries)


        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        pose_logits = self.mlp_heads["pose_head"](pose_features).transpose(1, 2)
        # rot_logits = self.mlp_heads["rot_head"](pose_features).transpose(1, 2)
        # trans_logits = self.mlp_heads["trans_head"](pose_features).transpose(1, 2)


        # reshape outputs to num_layers x batch x nqueries x noutput
        pose_logits = pose_logits.reshape(num_layers, batch, num_queries, -1)
        # trans_logits = trans_logits.reshape(num_layers, batch, num_queries, -1)
        # rot_logits = rot_logits.reshape(num_layers, batch, num_queries, -1)

        outputs = []
        for l in range(num_layers):
 

            box_prediction = {
                "pose_logits": pose_logits[l],
                # "trans_logits": trans_logits[l],
                # "rot_logits": rot_logits[l],
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]
        # print(outputs['pose_logits'].shape)
        return outputs
        # return {
        #     "outputs": outputs,  # output from last layer of decoder
        #     "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        # }

    def forward(self, pre_enc_xyz, pre_enc_features, tgt_xyz):
        # pre_enc_xyz: B x sampled x 3
        # features: B x feat x sampled
        # print("pre_enc_xyz : ",pre_enc_xyz.shape) 
        # print("pre_enc_features : ",pre_enc_features.shape) 
        # print("tgt_xyz : ",tgt_xyz.shape) 

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        
        enc_xyz, enc_features= self.run_encoder(pre_enc_xyz, pre_enc_features)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        # print("enc_xyz : ",enc_xyz.shape)
        # print("enc_features : ", enc_features.shape) 

        # test here. we dont need encoder to decoder projection. length is same.
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3
        # print("enc_features 2 : ", enc_features.shape) 

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

        query_xyz, query_embed = self.get_query_embeddings(tgt_xyz, point_cloud_dims)  


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

        # # pose_features change to batch x (num_layers x channel) x num_queries
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
                transformer_dict: Dict, label_type: LabelType, mlp: List[int], linear: List[int],
                append_features: bool = True, batch_norm: bool = False, dropout: bool = False, **_kwargs: Any):
        super().__init__()
        self._point_dim = point_dim
        self._append_features = append_features
        self._input_dim = input_dim
        self.transformer_params = transformer_dict
        self._transformer = ModelTransformer(self.transformer_params, encoder_dim=self.transformer_params.enc_dim, decoder_dim=self.transformer_params.dec_dim, num_queries = 1024)

        self._label_type = label_type
        # print("========Output LAYER TEST ==========\n")
        print(label_type)
        # layers

        # dense head
        mlp_layers = [1024, *mlp]
        # print("mlp_layers : ", mlp_layers)
        self.pos_conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)
        # print("output_dim : ", self.conv.output_dim())
        self.pos_linear = LinearMultiLayer(linear, batch_norm=batch_norm,
                                       dropout_keep=dropout, dropout_last=True)
        self.pos_output = nn.Linear(linear[-1], label_type.dim, bias=True)


        self.rot_conv = Conv1dMultiLayer(mlp_layers, batch_norm=batch_norm)
        # print("output_dim : ", self.conv.output_dim())
        self.rot_linear = LinearMultiLayer(linear, batch_norm=batch_norm,
                                       dropout_keep=dropout, dropout_last=True)
        self.rot_output = nn.Linear(linear[-1], label_type.dim, bias=True)


        # init weights
        nn.init.xavier_uniform_(self.pos_output.weight)
        nn.init.xavier_uniform_(self.rot_output.weight)

        # bias
        if label_type.bias is not None:
            for i, v in enumerate(label_type.bias):
                self.pos_output.bias.data[i] = v
                self.rot_output.bias.data[i] = v

    def _output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self._label_type == LabelType.POSE3D_QUAT:
            x[:, 3] = torch.sigmoid(x[:, 3])
            x[:, 4:] = torch.tanh(x[:, 4:])
        elif self._label_type == LabelType.POSE3D_DUAL_QUAT:
            x[:, 0] = torch.sigmoid(x[:, 0])
            x[:, 1:4] = torch.tanh(x[:, 1:4])
        return x

    def output_dim(self) -> int:
        # return self._input_dim - self._point_dim
        return self._label_type.dim

    def forward(self, clouds_dict: Dict) -> torch.Tensor:
        clouds_dict_buf = clouds_dict
        sa_cloud = clouds_dict['set_abstract_cloud'] # 2*B x (3 + feature) x sampled
        batch_dim = int(sa_cloud.shape[0] / 2)
        src_sa_cloud = sa_cloud[:batch_dim, ...] #B x (3+feat) x sampled
        tgt_sa_cloud = sa_cloud[batch_dim:, ...] #B x (3+feat) x sampled

        # print("src_sa_cloud : ", src_sa_cloud.shape)
        # print("tgt_sa_cloud : ", tgt_sa_cloud.shape)

        trans_src_xyz = src_sa_cloud[:, :self._point_dim,:].transpose(1,2).contiguous() # B x sampled x (3)
        trans_feature = src_sa_cloud[:, self._point_dim:,:].contiguous() # B x feat x sampled
        trans_tgt_xyz = tgt_sa_cloud[:, :self._point_dim,:].transpose(1,2).contiguous() # B x sampled x (3)

        # print("trans_src_xyz : ", trans_src_xyz)
        # print("trans_feature : ", trans_feature)
        # print("target_xyz : ", target_xyz)

        # print("trans_src_xyz : ", trans_src_xyz.shape)
        # print("trans_feature : ", trans_feature.shape)
        # xyz torch.Size([2, 4096, 3])
        # feature torch.Size([2, 256, 4096])

        # # xyz: batch x npoints x 3
        # # features: batch x channel x npoints
        # # print(trans_src_xyz)
        prediction = self._transformer(trans_src_xyz, trans_feature, trans_tgt_xyz)
        # print("trans_logits : ", prediction["trans_logits"].shape)
        # print("trans_normalized : ", prediction["trans_normalized"].shape)
        # print("trans_unnormalized : ", prediction["trans_unnormalized"].shape)
        # print("rot_logits : ", prediction["rot_logits"].shape)
        # print(torch.cat((prediction["trans_normalized"], prediction["rot_logits"]), dim=2).shape)
        # dec_feature = dec_feature.transpose(1, 2).contiguous()
        # trans_xyz = trans_xyz.transpose(1,2).contiguous()
        # print("xyz", trans_xyz.shape)

        # trans_out = torch.cat((trans_xyz, dec_feature), dim=1)
        # print("trans_out", trans_out.shape)
        
        pos_x = self.pos_conv(prediction["pose_logits"])
        pos_x, _ = torch.max(pos_x, dim=2)
        # # output shape
        pos_x = self.pos_linear(pos_x)
        pos_x = self.pos_output(pos_x)
        pos_x = self._output_activation(pos_x)

        # rot_x = self.rot_conv(prediction["rot_logits"])
        # rot_x, _ = torch.max(rot_x, dim=2)
        # # output shape
        # rot_x = self.rot_linear(rot_x)
        # rot_x = self.rot_output(rot_x)
        # rot_x = self._output_activation(rot_x)

        # # print("pos_x : ", pos_x)
        # # print("rot_x : ", rot_x)

        # # x = torch.cat(rot_x, pos_x, dim=2)
        # output_dict = {
        #     "trans": pos_x,
        #     "rot": rot_x,
        # }
        clouds_dict_buf['trans'] = pos_x
        clouds_dict_buf['rot'] = pos_x
        # clouds_dict_buf['trans'] = pos_x
        # clouds_dict_buf['trans'] = prediction["trans_logits"]
        # clouds_dict_buf['trans'] = prediction["trans_logits"]


        return clouds_dict_buf
        # return torch.cat((prediction["trans_logits"], prediction["trans_normalized"], prediction["rot_logits"]), dim=2)




class Transformer(DEEPCLRTFModule):
    """Transformer encoder-decoder block for attention mechanism [embedded_flow]."""
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._transformer = TransformerBase(**kwargs)
    def output_dim(self):
        return self._transformer.output_dim()

    def forward(self, clouds: torch.Tensor) -> torch.Tensor:
        return self._transformer(clouds)


class OutputSimple(DEEPCLRTFModule):
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

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, cloud_dict: Dict) -> torch.Tensor:
        cloud_dict_buf = cloud_dict
        x = cloud_dict['motion_embed_clouds']
        # apply pointnet to get final feature vector
        x = self.conv(x)
        x, _ = torch.max(x, dim=2)

        # output shape
        x = self.linear(x)
        x = self.output(x)
        x = self._output_activation(x)

        
        # x = torch.cat(rot_x, pos_x, dim=2)
        cloud_dict_buf['trans'] = x
        cloud_dict_buf['rot'] = x
        
        # print("src_cloud :", cloud_dict_buf['src_cloud'].shape)
        # print("tgt_cloud :", cloud_dict_buf['tgt_cloud'].shape)
        # print("set_abstract_cloud :", cloud_dict_buf['set_abstract_cloud'].shape)
        # print("motion_embed_clouds :", cloud_dict_buf['motion_embed_clouds'].shape)
        # print("trans :", cloud_dict_buf['trans'])
        # print("rot :", cloud_dict_buf['rot'])

        return cloud_dict_buf

        # return x

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


class DEEPCLRTFLoss(DEEPCLRTFModule, metaclass=abc.ABCMeta):
    """Abstract base class for loss calculation modules."""
    def __init__(self):
        super().__init__()

    def output_dim(self) -> int:
        return 1

    @abc.abstractmethod
    def get_weights(self) -> Dict:
        raise NotImplementedError


class TransformLoss(DEEPCLRTFLoss):
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


class TransformUncertaintyLoss(DEEPCLRTFLoss):
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


class AccumulatedLoss(DEEPCLRTFLoss):
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


def init_module(cfg: Config, *args: Any, **kwargs: Any) -> DEEPCLRTFModule:
    """Initialize DEEPCLRTFModule from config."""
    return factory(DEEPCLRTFModule, cfg.name, *args, **cfg.params, **kwargs)


def init_loss_module(cfg: Config, label_type: LabelType, *args: Any, **kwargs: Any) -> DEEPCLRTFLoss:
    """Initialize DEEPCLRTFLoss from config."""
    return factory(DEEPCLRTFLoss, cfg.name, *args, label_type=label_type, **cfg.params, **kwargs)


def init_optional_module(cfg: Optional[Config], *args: Any, **kwargs: Any) -> Optional[DEEPCLRTFModule]:
    """Initialize optional DEEPCLRTFModule from config."""
    if cfg is None:
        return None
    else:
        return factory(DEEPCLRTFModule, cfg.name, *args, **cfg.params, **kwargs)


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



# class DEEPCLRTF(BaseModel):
#     """Main DeepCLR network."""
#     _loss_layer: Optional[DEEPCLRTFLoss]

#     def __init__(self, input_dim: int, label_type: LabelType, cloud_features: Config,
#                  merge: Config, output: Config, transform: Optional[Config] = None,
#                  loss: Optional[Config] = None, **kwargs: Any):
#         super().__init__()

#         self._input_dim = input_dim

#         transform_layer = init_optional_module(transform, input_dim=input_dim, **kwargs)
#         transform_layer_output_dim = input_dim if transform_layer is None else transform_layer.output_dim()

#         cloud_feat_layer = init_module(cloud_features, input_dim=transform_layer_output_dim, **kwargs)
#         merge_layer = init_module(merge, input_dim=cloud_feat_layer.output_dim(), **kwargs)
#         print("merge_layer.output_dim() : ", merge_layer.output_dim())
#         output_layer = init_module(output, input_dim=merge_layer.output_dim(), label_type=label_type, **kwargs)

#         if transform_layer is None:
#             self._cloud_layers = nn.Sequential(cloud_feat_layer)
#             self._merge_layers = nn.Sequential(merge_layer, output_layer)
#         else:
#             self._cloud_layers = nn.Sequential(transform_layer, cloud_feat_layer)
#             self._merge_layers = nn.Sequential(merge_layer, output_layer)

#         if loss is not None:
#             if isinstance(loss, list):
#                 loss_modules = [init_loss_module(loss_cfg, label_type, **kwargs) for loss_cfg in loss]
#                 self._loss_layer = AccumulatedLoss(loss_modules)
#             else:
#                 self._loss_layer = init_loss_module(loss, label_type, **kwargs)
#         else:
#             self._loss_layer = None

#     def get_input_dim(self) -> int:
#         return self._input_dim

#     def has_loss(self) -> bool:
#         return self._loss_layer is not None

#     def get_loss_weights(self) -> Dict:
#         if self._loss_layer is not None:
#             return self._loss_layer.get_weights()
#         else:
#             return {}

#     def forward(self, x: torch.Tensor, is_feat: bool = False, m: Optional[torch.Tensor] = None,
#                 y: Optional[torch.Tensor] = None, debug: bool = False)\
#             -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
#         # cloud features
#         if not is_feat:
#             x = self.cloud_features(x, m=m)

#         # merge
#         model_output = self._merge_layers(x)
#         y_pred, model_aux = split_output(model_output)

#         # loss
#         if self._loss_layer is not None and y is not None:
#             loss_output = self._loss_layer(y_pred, y, **model_aux)
#             loss, loss_aux = split_output(loss_output)
#             debug_output = {**model_aux, **loss_aux, 'x_aug': x} if debug else None
#         else:
#             loss = None
#             debug_output = None

#         return y_pred, loss, debug_output

#     def cloud_features(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # apply transforms
#         if m is not None:
#             dim = m.shape[-1] - 1
#             x[:, :, :dim] = tgm.transform_points(m, x[:, :, :dim])

#         # format clouds for pointnet2
#         x = x.transpose(1, 2)

#         # forward pass
#         x = self._cloud_layers(x)
#         return x



class DEEPCLRTF(BaseModel):
    """Main DEEPCLRTF network."""
    _loss_layer: Optional[DEEPCLRTFLoss]

    def __init__(self, input_dim: int, label_type: LabelType, cloud_features: Config,
                 transformer: Config, transform: Optional[Config] = None,
                 loss: Optional[Config] = None, **kwargs: Any):
        super().__init__()

        self._input_dim = input_dim

        transform_layer = init_optional_module(transform, input_dim=input_dim, **kwargs)
        transform_layer_output_dim = input_dim if transform_layer is None else transform_layer.output_dim()
        cloud_feat_layer = init_module(cloud_features, input_dim=transform_layer_output_dim, **kwargs)
        # merge_layer = init_module(merge, input_dim=cloud_feat_layer.output_dim(), **kwargs)
        deep_transformer_layer = init_module(transformer, input_dim=cloud_feat_layer.output_dim(), label_type=label_type, **kwargs)
        # output_layer = init_module(output, input_dim=deep_transformer_layer.output_dim(), label_type=label_type, **kwargs)
        # print("transform_layer_output_dim : ", transform_layer_output_dim)
        # print("cloud_feat_layer_output_dim : ", cloud_feat_layer.output_dim())
        # print("merge_layer_output_dim : ", merge_layer.output_dim())
        # print("deep_transformer_layer : ", deep_transformer_layer.output_dim())
        # print("output_layer_output_dim : ", output_layer.output_dim())
        # transform_layer_output_dim :  4
        # cloud_feat_layer_output_dim :  67
        # merge_layer_output_dim :  259
        # output_layer_output_dim :  8

        if transform_layer is None:
            self._cloud_layers = nn.Sequential(cloud_feat_layer)
            self._merge_layers = nn.Sequential(deep_transformer_layer)
        else:
            self._cloud_layers = nn.Sequential(transform_layer, cloud_feat_layer)
            self._merge_layers = nn.Sequential(deep_transformer_layer)

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
        # print(model_output)       
        loss = None
        debug_output = None
        # y_pred, model_aux = split_output(model_output)

        # # loss
        # if self._loss_layer is not None and y is not None:
        #     loss_output = self._loss_layer(y_pred, y, **model_aux)
        #     loss, loss_aux = split_output(loss_output)
        #     debug_output = {**model_aux, **loss_aux, 'x_aug': x} if debug else None
        # else:
        #     loss = None
        #     debug_output = None

        # print("y_pred: ", y_pred) 
        # # print("loss: ", loss ) 
        # # print("debug_output: ", debug_output ) 
        # # loss:  None
        # # debug_output:  None

        return model_output, loss, debug_output

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

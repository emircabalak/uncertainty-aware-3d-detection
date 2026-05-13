"""
OpenPCDet-compatible PointPillars backbone modules.

The layer structure (and state_dict key names) here match OpenPCDet's official
`pointpillar_7728.pth` checkpoint exactly, so backbone weights can be loaded
directly. The detection head is NOT loaded — we attach our own evidential head
on top of the backbone features.

Components (in the order points flow through them):
    1. PillarVFE        — per-point MLP + max-pool inside each pillar
    2. PointPillarScatter — scatter pillar features to a 2D BEV pseudo-image
    3. BaseBEVBackbone   — multi-scale 2D CNN producing rich BEV features

Reference: Lang et al., "PointPillars" (CVPR 2019)
           OpenPCDet repo: https://github.com/open-mmlab/OpenPCDet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class PFNLayer(nn.Module):
    """Pillar Feature Network Layer — shared Linear + BN + ReLU across points.

    Matches OpenPCDet state_dict keys: `linear.weight`, `norm.weight`,
    `norm.bias`, `norm.running_mean`, `norm.running_var`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_norm: bool = True, last_layer: bool = True):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if use_norm:
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N_voxels, max_points, in_channels)
        Returns:
            (N_voxels, out_channels) if last_layer else
            (N_voxels, max_points, out_channels)
        """
        x = self.linear(inputs)  # (N, P, C_out)
        N, P, C = x.shape
        if self.use_norm:
            x = self.norm(x.view(-1, C)).view(N, P, C)
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=False)[0]  # (N, C)
            return x_max
        return x


class PillarVFE(nn.Module):
    """Encodes raw points inside each pillar into a fixed-size feature vector.

    Input augmentation (matches OpenPCDet default for KITTI PointPillars):
        Each point becomes a 10-dim vector:
            [x, y, z, intensity,                     # raw
             x - pillar_mean_x, y - pillar_mean_y, z - pillar_mean_z,   # cluster
             x - pillar_center_x, y - pillar_center_y, z - pillar_center_z]  # center

    Args:
        num_point_features: Raw feature count per point (e.g. 4 for KITTI).
        num_filters: Output channels per PFN layer (default [64] — single layer).
        voxel_size: (vx, vy, vz) in meters.
        point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax).
    """

    def __init__(
        self,
        num_point_features: int = 4,
        num_filters: List[int] = [64],
        voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0),
        point_cloud_range: Tuple[float, ...] = (0, -39.68, -3, 69.12, 39.68, 1),
        use_norm: bool = True,
        use_absolute_xyz: bool = True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.use_absolute_xyz = use_absolute_xyz

        in_channels = num_point_features + (6 if use_absolute_xyz else 3)
        self.num_filters = num_filters
        num_filters = [in_channels] + list(num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_c = num_filters[i]
            out_c = num_filters[i + 1]
            last_layer = (i == len(num_filters) - 2)
            pfn_layers.append(
                PFNLayer(in_c, out_c, use_norm=use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.output_channels = num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num: torch.Tensor, max_num: int,
                               axis: int = 0) -> torch.Tensor:
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int, device=actual_num.device
        ).view(max_num_shape)
        return actual_num.int() > max_num

    def forward(self, batch_dict: Dict) -> Dict:
        """Augment raw voxel features and run PFN.

        Args:
            batch_dict with:
                - voxels: (N_voxels, max_points, num_point_features)
                - voxel_num_points: (N_voxels,)
                - voxel_coords: (N_voxels, 4)  [batch_idx, z, y, x]

        Sets:
            - pillar_features: (N_voxels, output_channels)
        """
        voxel_features = batch_dict["voxels"]       # (N, P, C_in)
        voxel_num_points = batch_dict["voxel_num_points"]  # (N,)
        coords = batch_dict["voxel_coords"]         # (N, 4)

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1).clamp(min=1.0)
        f_cluster = voxel_features[:, :, :3] - points_mean  # (N, P, 3)

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].type_as(voxel_features).unsqueeze(1) * self.voxel_x
            + self.x_offset
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].type_as(voxel_features).unsqueeze(1) * self.voxel_y
            + self.y_offset
        )
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].type_as(voxel_features).unsqueeze(1) * self.voxel_z
            + self.z_offset
        )

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        features = torch.cat(features, dim=-1)  # (N, P, 10)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        batch_dict["pillar_features"] = features
        return batch_dict


class PointPillarScatter(nn.Module):
    """Places each pillar's feature vector at its (y, x) BEV location.

    No learnable parameters — pure scatter op. Matches OpenPCDet naming.
    """

    def __init__(self, num_bev_features: int, grid_size: Tuple[int, int, int]):
        super().__init__()
        self.num_bev_features = num_bev_features
        self.nx, self.ny, self.nz = grid_size

    def forward(self, batch_dict: Dict) -> Dict:
        pillar_features = batch_dict["pillar_features"]  # (N, C)
        coords = batch_dict["voxel_coords"].long()       # (N, 4) [b, z, y, x]
        batch_size = batch_dict["batch_size"]

        batch_spatial_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features, self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype, device=pillar_features.device,
            )
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = (
                this_coords[:, 1] * self.ny * self.nx +
                this_coords[:, 2] * self.nx +
                this_coords[:, 3]
            )
            pillars = pillar_features[batch_mask, :].t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(
            batch_size, self.num_bev_features * self.nz, self.ny, self.nx
        )
        batch_dict["spatial_features"] = batch_spatial_features
        return batch_dict


class BaseBEVBackbone(nn.Module):
    """2D CNN backbone for BEV features. Matches OpenPCDet state_dict naming.

    Default config (matches `pointpillar_7728.pth`):
        LAYER_NUMS:           [3, 5, 5]
        LAYER_STRIDES:        [2, 2, 2]
        NUM_FILTERS:          [64, 128, 256]
        UPSAMPLE_STRIDES:     [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]
    Output channels: sum(NUM_UPSAMPLE_FILTERS) = 384.
    """

    def __init__(
        self,
        input_channels: int = 64,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [2, 2, 2],
        num_filters: List[int] = [64, 128, 256],
        upsample_strides: List[int] = [1, 2, 4],
        num_upsample_filters: List[int] = [128, 128, 128],
    ):
        super().__init__()
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filters)

        num_levels = len(layer_nums)
        c_in_list = [input_channels] + list(num_filters[:-1])

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], 3,
                          stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], 3,
                              padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            stride = upsample_strides[idx]
            if stride > 1 or (stride == 1 and num_upsample_filters[idx] != num_filters[idx]):
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        stride, stride=stride, bias=False,
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx],
                                   eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))
            else:
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(num_filters[idx], num_upsample_filters[idx],
                              1, stride=1, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[idx],
                                   eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

        self.num_bev_features = sum(num_upsample_filters)
        self.output_channels = self.num_bev_features

    def forward(self, batch_dict: Dict) -> Dict:
        spatial_features = batch_dict["spatial_features"]
        ups = []
        x = spatial_features
        for i, (block, deblock) in enumerate(zip(self.blocks, self.deblocks)):
            x = block(x)
            ups.append(deblock(x))
        x = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        batch_dict["spatial_features_2d"] = x
        return batch_dict


class PretrainedPointPillarsBackbone(nn.Module):
    """Full pretrained backbone = VFE + Scatter + BEV backbone.

    State dict keys match OpenPCDet's `pointpillar_7728.pth` for these three
    submodules, so `load_state_dict(strict=False)` will populate them from
    the checkpoint and skip everything else (head, roi, etc).
    """

    def __init__(
        self,
        num_point_features: int = 4,
        voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0),
        point_cloud_range: Tuple[float, ...] = (0, -39.68, -3, 69.12, 39.68, 1),
        grid_size: Tuple[int, int, int] = (432, 496, 1),  # (X, Y, Z)
    ):
        super().__init__()
        self.vfe = PillarVFE(
            num_point_features=num_point_features,
            num_filters=[64],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.map_to_bev_module = PointPillarScatter(
            num_bev_features=self.vfe.output_channels,
            grid_size=grid_size,
        )
        self.backbone_2d = BaseBEVBackbone(
            input_channels=self.vfe.output_channels,
        )
        self.output_channels = self.backbone_2d.num_bev_features  # 384

    def forward(self, batch_dict: Dict) -> Dict:
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        return batch_dict

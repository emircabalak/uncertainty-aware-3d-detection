"""
Uncertainty-aware CenterPoint: Full detector pipeline.

This module integrates OpenPCDet's CenterPoint with our uncertainty components:
  - Evidential detection head (replaces standard CenterHead)
  - MC Dropout wrapper (for inference-time uncertainty)
  - Uncertainty-aware NMS (post-processing)

Supports two modes:
  1. Evidential: Single forward pass, predicts NIG parameters
  2. MC Dropout: Multiple stochastic passes, aggregates variance
  3. Combined: Evidential training + MC Dropout inference (best of both)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .evidential_head import EvidentialCenterHead
from .mc_dropout import MCDropoutWrapper, inject_dropout, enable_dropout
from .uncertainty_nms import uncertainty_aware_nms


class VoxelFeatureEncoder(nn.Module):
    """Mean Voxel Feature Encoding (VFE).

    Simplified version: takes raw points in each voxel and computes their mean.
    This is the standard approach used in many voxel-based detectors.
    """

    def __init__(self, num_point_features: int = 4):
        super().__init__()
        self.num_point_features = num_point_features

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_num_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: (num_voxels, max_points_per_voxel, C)
            voxel_num_points: (num_voxels,)

        Returns:
            (num_voxels, C) Mean features per voxel.
        """
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = voxel_num_points.float().unsqueeze(-1).clamp(min=1.0)
        points_mean = points_mean / normalizer
        return points_mean


class PillarBackbone(nn.Module):
    """PointPillars-style backbone — no spconv needed.

    Instead of 3D convolutions, this approach:
      1. Treats each vertical column (pillar) as a unit
      2. Encodes pillar features with a simple PointNet (shared MLP)
      3. Scatters features onto a 2D BEV pseudo-image
      4. Applies standard 2D convolutions

    This is memory-efficient and works on any GPU without special libraries.
    Performance is comparable to sparse 3D conv for KITTI-scale detection.

    Reference: Lang et al., "PointPillars" (CVPR 2019)
    """

    def __init__(
        self,
        input_channels: int = 4,
        grid_size: Tuple[int, int, int] = (432, 496, 1),
        pillar_features: int = 64,
    ):
        super().__init__()
        self.nz, self.ny, self.nx = grid_size

        self.pillar_encoder = nn.Sequential(
            nn.Linear(input_channels, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pillar_features, bias=False),
            nn.BatchNorm1d(pillar_features),
            nn.ReLU(inplace=True),
        )

        self.output_channels = pillar_features
        self.backbone_stride = 1  # No spatial downsampling yet (2D backbone handles it)

    def forward(self, batch_dict: Dict) -> Dict:
        """Encode voxels and scatter to BEV pseudo-image.

        Args:
            batch_dict with:
                - voxel_features: (N_voxels, C) mean-encoded voxel features
                - voxel_coords: (N_voxels, 4) [batch_idx, z, y, x]
                - batch_size: int

        Adds:
            - spatial_features: (B, pillar_features, ny, nx) BEV feature map
        """
        voxel_features = batch_dict["voxel_features"]  # (N, C)
        voxel_coords = batch_dict["voxel_coords"].long()  # (N, 4)
        batch_size = batch_dict["batch_size"]

        pillar_features = self.pillar_encoder(voxel_features)  # (N, 64)

        bev = torch.zeros(
            (batch_size, self.output_channels, self.ny, self.nx),
            device=pillar_features.device, dtype=pillar_features.dtype,
        )

        batch_idx = voxel_coords[:, 0]
        y_idx = voxel_coords[:, 2].clamp(0, self.ny - 1)
        x_idx = voxel_coords[:, 3].clamp(0, self.nx - 1)

        bev[batch_idx, :, y_idx, x_idx] = pillar_features

        batch_dict["spatial_features"] = bev
        return batch_dict


class BEVBackbone2D(nn.Module):
    """2D backbone for processing BEV (Bird's Eye View) features.

    Multi-scale feature extraction with top-down pathway, similar to FPN.
    Produces the spatial feature map that feeds into the detection head.
    """

    def __init__(
        self,
        input_channels: int = 128,
        layer_nums: List[int] = [5, 5],
        layer_strides: List[int] = [1, 2],
        num_filters: List[int] = [128, 256],
        upsample_strides: List[int] = [1, 2],
        num_upsample_filters: List[int] = [256, 256],
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.dropout_rate = dropout_rate

        for i in range(len(layer_nums)):
            in_ch = input_channels if i == 0 else num_filters[i - 1]
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(in_ch, num_filters[i], 3, stride=layer_strides[i],
                          bias=False),
                nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            for _ in range(layer_nums[i]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[i], num_filters[i], 3, padding=1,
                              bias=False),
                    nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
                if dropout_rate > 0:
                    cur_layers.append(nn.Dropout2d(p=dropout_rate))

            self.blocks.append(nn.Sequential(*cur_layers))

            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[i], num_upsample_filters[i],
                    upsample_strides[i], stride=upsample_strides[i], bias=False,
                ),
                nn.BatchNorm2d(num_upsample_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.output_channels = sum(num_upsample_filters)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        ups = []
        x = spatial_features
        for i, (block, deblock) in enumerate(zip(self.blocks, self.deblocks)):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)


class UncertaintyCenterPoint(nn.Module):
    """Complete Uncertainty-aware CenterPoint detector.

    Pipeline:
        Raw Points → Voxelization → VFE → 3D Sparse Backbone →
        2D BEV Backbone → Evidential Detection Head → Uncertainty-aware NMS

    Args:
        config: Dictionary with model configuration.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        pc_range = config.get("point_cloud_range",
                              [0, -39.68, -3, 69.12, 39.68, 1])
        voxel_size = config.get("voxel_size", [0.05, 0.05, 0.1])
        self.point_cloud_range = np.array(pc_range)
        self.voxel_size = np.array(voxel_size)

        grid_size = (
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int64)  # (X, Y, Z)
        self.grid_size = grid_size

        num_point_features = config.get("num_point_features", 4)

        self.vfe = VoxelFeatureEncoder(num_point_features)

        pillar_features = config.get("pillar_features", 64)
        self.backbone_3d = PillarBackbone(
            input_channels=num_point_features,
            grid_size=grid_size[::-1].tolist(),  # (Z, Y, X)
            pillar_features=pillar_features,
        )

        bev_input_ch = self.backbone_3d.output_channels
        dropout_rate = config.get("dropout_rate", 0.1)

        self.backbone_2d = BEVBackbone2D(
            input_channels=bev_input_ch,
            layer_strides=[1, 2],
            num_filters=[64, 128],
            upsample_strides=[1, 2],
            num_upsample_filters=[128, 128],
            dropout_rate=dropout_rate,
        )

        num_classes = config.get("num_classes", 3)
        self.head = EvidentialCenterHead(
            input_channels=self.backbone_2d.output_channels,
            num_classes=num_classes,
            bbox_code_size=config.get("bbox_code_size", 7),
        )

        self.uncertainty_mode = config.get("uncertainty_mode", "evidential")
        self.mc_forward_passes = config.get("mc_forward_passes", 20)

        self.feature_map_stride = 1

        # Detection config
        self.score_threshold = config.get("score_threshold", 0.1)
        self.nms_iou_threshold = config.get("nms_iou_threshold", 0.7)
        self.uncertainty_penalty = config.get("uncertainty_penalty", 0.3)
        self.max_detections = config.get("max_detections", 500)

    def forward(self, batch_dict: Dict) -> Dict:
        """Training forward pass.

        Args:
            batch_dict: Dictionary with:
                - voxels: (N, max_points, C) voxel point features
                - voxel_num_points: (N,) actual points per voxel
                - voxel_coords: (N, 4) [batch_idx, z, y, x]
                - batch_size: int

        Returns:
            Dictionary with predictions and losses (if targets provided).
        """
        voxel_features = self.vfe(
            batch_dict["voxels"],
            batch_dict["voxel_num_points"],
        )
        batch_dict["voxel_features"] = voxel_features

        batch_dict = self.backbone_3d(batch_dict)

        spatial_features_2d = self.backbone_2d(batch_dict["spatial_features"])
        batch_dict["spatial_features_2d"] = spatial_features_2d

        pred_dict = self.head(spatial_features_2d)
        batch_dict["pred_dict"] = pred_dict

        return batch_dict

    @torch.no_grad()
    def predict(self, batch_dict: Dict) -> Dict:
        """Inference with uncertainty estimation.

        Returns detections with per-box uncertainty estimates.
        """
        batch_dict = self.forward(batch_dict)
        pred_dict = batch_dict["pred_dict"]

        detections = self._decode_predictions(pred_dict)

        for det in detections:
            unc = self.head.extract_uncertainties_for_detections(
                pred_dict, det["boxes"], det["scores"],
                feature_map_stride=self.feature_map_stride,
                voxel_size=(self.voxel_size[0], self.voxel_size[1]),
                point_cloud_range=tuple(self.point_cloud_range),
            )
            det.update(unc)

            nms_result = uncertainty_aware_nms(
                boxes=det["boxes"],
                scores=det["scores"],
                labels=det["labels"],
                uncertainty=unc["total_bbox_uncertainty"],
                iou_threshold=self.nms_iou_threshold,
                score_threshold=self.score_threshold,
                uncertainty_penalty=self.uncertainty_penalty,
                max_detections=self.max_detections,
            )
            det["final_boxes"] = nms_result["boxes"]
            det["final_scores"] = nms_result["scores"]
            det["final_labels"] = nms_result["labels"]
            det["final_uncertainty"] = nms_result["uncertainty"]

        batch_dict["detections"] = detections
        return batch_dict

    def _decode_predictions(
        self, pred_dict: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode heatmap + regression into bounding box detections.

        Uses CenterPoint-style decoding: find peaks in heatmap,
        extract bbox parameters at peak locations.
        """
        heatmap = pred_dict["heatmap"]  # (B, K, H, W)
        bbox_pred = pred_dict["bbox_pred"]  # (B, 7, H, W)
        B, K, H, W = heatmap.shape

        heatmap = torch.sigmoid(heatmap)

        heatmap_pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        peak_mask = (heatmap == heatmap_pool) & (heatmap > self.score_threshold)

        detections = []
        for b in range(B):
            batch_boxes = []
            batch_scores = []
            batch_labels = []

            for cls in range(K):
                peaks = peak_mask[b, cls].nonzero(as_tuple=False)  # (N, 2) y, x

                if peaks.shape[0] == 0:
                    continue

                ys, xs = peaks[:, 0], peaks[:, 1]
                scores = heatmap[b, cls, ys, xs]

                boxes_at_peaks = bbox_pred[b, :, ys, xs].T  # (N, 7)

                boxes_at_peaks[:, 0] += xs.float() * self.voxel_size[0] * \
                    self.feature_map_stride + self.point_cloud_range[0]
                boxes_at_peaks[:, 1] += ys.float() * self.voxel_size[1] * \
                    self.feature_map_stride + self.point_cloud_range[1]

                labels = torch.full_like(scores, cls + 1, dtype=torch.long)

                batch_boxes.append(boxes_at_peaks)
                batch_scores.append(scores)
                batch_labels.append(labels)

            if batch_boxes:
                detections.append({
                    "boxes": torch.cat(batch_boxes, dim=0),
                    "scores": torch.cat(batch_scores, dim=0),
                    "labels": torch.cat(batch_labels, dim=0),
                })
            else:
                device = heatmap.device
                detections.append({
                    "boxes": torch.zeros((0, 7), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })

        return detections

"""
Evidential 3D detector built on top of OpenPCDet's pretrained PointPillars.

Pipeline:
    Raw Points → PillarVFE → Scatter → BaseBEVBackbone → EvidentialCenterHead
                 └──────── pretrained ────────┘         └─── from scratch ───┘

Key differences vs. Model 1 (uncertainty_3d_detection):
    - Backbone has 3× more capacity (3 scales vs 2, wider filters)
    - Input features: 10 (with x_c, y_c, z_c, x_p, y_p, z_p) vs 4
    - Backbone output: 384 channels vs 256
    - Backbone is initialized from pointpillar_7728.pth (80-epoch pretrained)
    - MC Dropout path removed (known to fail on models not trained with dropout)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .pretrained_backbone import PretrainedPointPillarsBackbone
from .evidential_head import EvidentialCenterHead
from .uncertainty_nms import uncertainty_aware_nms


class EvidentialPretrainedDetector(nn.Module):
    """Pretrained-backbone + evidential-head detector.

    Args:
        config: Dict with keys described in configs/pretrained_kitti.yaml.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        pc_range = config.get(
            "point_cloud_range", [0, -39.68, -3, 69.12, 39.68, 1]
        )
        voxel_size = config.get("voxel_size", [0.16, 0.16, 4.0])
        self.point_cloud_range = np.array(pc_range)
        self.voxel_size = np.array(voxel_size)

        grid_size_xyz = (
            (self.point_cloud_range[3:] - self.point_cloud_range[:3])
            / self.voxel_size
        ).astype(np.int64)  # (X, Y, Z)
        self.grid_size = grid_size_xyz

        num_point_features = config.get("num_point_features", 4)

        self.backbone = PretrainedPointPillarsBackbone(
            num_point_features=num_point_features,
            voxel_size=tuple(self.voxel_size.tolist()),
            point_cloud_range=tuple(self.point_cloud_range.tolist()),
            grid_size=tuple(grid_size_xyz.tolist()),
        )

        num_classes = config.get("num_classes", 3)
        self.head = EvidentialCenterHead(
            input_channels=self.backbone.output_channels,  # 384
            num_classes=num_classes,
            bbox_code_size=config.get("bbox_code_size", 7),
        )

        # Feature map is at 1/2 the pillar grid resolution.
        # OpenPCDet's BaseBEVBackbone uses LAYER_STRIDES=[2,2,2] and
        # UPSAMPLE_STRIDES=[1,2,4], so every upsampled branch lands at
        # stride 2 relative to the input BEV grid.
        # (Input 496x432 → output 248x216.)
        self.feature_map_stride = 2

        # Detection config
        self.score_threshold = config.get("score_threshold", 0.1)
        self.nms_iou_threshold = config.get("nms_iou_threshold", 0.7)
        self.uncertainty_penalty = config.get("uncertainty_penalty", 0.3)
        self.max_detections = config.get("max_detections", 500)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def freeze_backbone(self, verbose: bool = True):
        """Freeze pretrained backbone weights (stage 1 fine-tuning)."""
        frozen = 0
        for p in self.backbone.parameters():
            p.requires_grad = False
            frozen += p.numel()
        # BN layers inside backbone should stay in eval mode when frozen,
        # otherwise running stats drift. We'll handle that in train() method.
        self._backbone_frozen = True
        if verbose:
            print(f"[freeze] Backbone frozen ({frozen:,} params). "
                  f"Only head is trainable.")

    def unfreeze_backbone(self, verbose: bool = True):
        """Unfreeze backbone (stage 2 full fine-tune)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False
        if verbose:
            print(f"[unfreeze] Backbone unfrozen — full fine-tune enabled.")

    def train(self, mode: bool = True):
        """Override train() to keep frozen-backbone BN in eval mode."""
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            # Put all BN layers inside backbone back into eval
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        return self

    def forward(self, batch_dict: Dict) -> Dict:
        """Training forward pass.

        Args:
            batch_dict with:
                - voxels: (N_voxels, max_points, num_point_features)
                - voxel_num_points: (N_voxels,)
                - voxel_coords: (N_voxels, 4)
                - batch_size: int
        """
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict["spatial_features_2d"]
        pred_dict = self.head(spatial_features_2d)
        batch_dict["pred_dict"] = pred_dict
        return batch_dict

    @torch.no_grad()
    def predict(self, batch_dict: Dict) -> Dict:
        """Inference with uncertainty estimation."""
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
        """Decode heatmap + bbox regression into detections.

        Identical to Model 1's decoder — reused because head output format
        is the same.
        """
        heatmap = pred_dict["heatmap"]          # (B, K, H, W)
        bbox_pred = pred_dict["bbox_pred"]      # (B, 7, H, W)
        B, K, H, W = heatmap.shape

        heatmap = torch.sigmoid(heatmap)
        heatmap_pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        peak_mask = (heatmap == heatmap_pool) & (heatmap > self.score_threshold)

        detections = []
        for b in range(B):
            batch_boxes, batch_scores, batch_labels = [], [], []
            for cls in range(K):
                peaks = peak_mask[b, cls].nonzero(as_tuple=False)
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

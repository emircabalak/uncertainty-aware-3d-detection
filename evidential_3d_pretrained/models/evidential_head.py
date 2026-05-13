"""
Evidential Deep Learning head for 3D object detection.

Replaces standard regression outputs with Normal-Inverse-Gamma (NIG) parameters,
enabling single-pass uncertainty estimation for both aleatoric and epistemic
components.

Reference: Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
           Sensoy et al., "Evidential Deep Learning" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class EvidentialCenterHead(nn.Module):
    """CenterPoint-style detection head with evidential uncertainty outputs.

    Instead of predicting (x,y,z,w,l,h,θ) directly, we predict NIG distribution
    parameters (γ, ν, α, β) for each regression target:
        - γ (gamma): predicted mean (same as standard regression output)
        - ν (nu): virtual observation count (>0, controls epistemic uncertainty)
        - α (alpha): shape parameter (>1, controls degrees of freedom)
        - β (beta): scale parameter (>0, controls aleatoric uncertainty)

    For classification, we use Dirichlet distribution instead of softmax.

    Args:
        input_channels: Number of input feature channels from BEV backbone.
        num_classes: Number of object classes.
        bbox_code_size: Dimension of bbox regression (default 7: x,y,z,dx,dy,dz,heading).
        num_heatmap_convs: Number of convolution layers in each prediction head.
        share_conv_channel: Intermediate channel size in prediction heads.
        min_nu: Minimum value for ν to ensure numerical stability.
        min_alpha_offset: α = 1 + softplus(raw) + min_alpha_offset, ensuring α > 1.
    """

    def __init__(
        self,
        input_channels: int = 512,
        num_classes: int = 3,
        bbox_code_size: int = 7,
        num_heatmap_convs: int = 2,
        share_conv_channel: int = 64,
        min_nu: float = 1e-3,
        min_alpha_offset: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_code_size = bbox_code_size
        self.min_nu = min_nu
        self.min_alpha_offset = min_alpha_offset

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, share_conv_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = self._make_head(
            share_conv_channel, num_classes, num_heatmap_convs
        )

        self.bbox_head = self._make_head(
            share_conv_channel, bbox_code_size, num_heatmap_convs
        )

        self.nu_head = self._make_head(
            share_conv_channel, bbox_code_size, num_heatmap_convs
        )

        self.alpha_head = self._make_head(
            share_conv_channel, bbox_code_size, num_heatmap_convs
        )

        self.beta_head = self._make_head(
            share_conv_channel, bbox_code_size, num_heatmap_convs
        )

        self.dirichlet_head = self._make_head(
            share_conv_channel, num_classes, num_heatmap_convs
        )

        self._init_weights()

    def _make_head(
        self, in_channels: int, out_channels: int, num_convs: int
    ) -> nn.Sequential:
        layers = []
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        pi = 0.1
        nn.init.constant_(self.heatmap_head[-1].bias, -np.log((1 - pi) / pi))

        for head in [self.nu_head, self.alpha_head, self.beta_head]:
            nn.init.normal_(head[-1].weight, mean=0, std=0.001)
            nn.init.constant_(head[-1].bias, 0)

    def forward(
        self, spatial_features_2d: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing detection + uncertainty predictions.

        Args:
            spatial_features_2d: BEV features from backbone, (B, C, H, W).

        Returns:
            Dictionary with:
                - heatmap: (B, num_classes, H, W) Center heatmap
                - bbox_pred: (B, 7, H, W) Mean bbox regression (γ)
                - nu: (B, 7, H, W) Virtual evidence count (ν > 0)
                - alpha: (B, 7, H, W) Shape parameter (α > 1)
                - beta: (B, 7, H, W) Scale parameter (β > 0)
                - dirichlet_alpha: (B, num_classes, H, W) Dirichlet concentrations
                - aleatoric_uncertainty: (B, 7, H, W) β / (α - 1)
                - epistemic_uncertainty: (B, 7, H, W) β / (ν * (α - 1))
        """
        x = self.shared_conv(spatial_features_2d)

        heatmap = self.heatmap_head(x)
        bbox_pred = self.bbox_head(x)  # γ: predicted mean

        nu = F.softplus(self.nu_head(x)) + self.min_nu  # ν > 0
        alpha = F.softplus(self.alpha_head(x)) + 1.0 + self.min_alpha_offset  # α > 1
        beta = F.softplus(self.beta_head(x)) + 1e-6  # β > 0

        dirichlet_alpha = F.softplus(self.dirichlet_head(x)) + 1.0  # α_k > 1

        aleatoric = beta / (alpha - 1.0)

        epistemic = beta / (nu * (alpha - 1.0))

        return {
            "heatmap": heatmap,
            "bbox_pred": bbox_pred,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "dirichlet_alpha": dirichlet_alpha,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
        }

    def extract_uncertainties_for_detections(
        self,
        pred_dict: Dict[str, torch.Tensor],
        det_boxes: torch.Tensor,
        det_scores: torch.Tensor,
        feature_map_stride: int = 1,
        voxel_size: Tuple[float, float] = (0.16, 0.16),
        point_cloud_range: Tuple[float, ...] = (0, -39.68, -3, 69.12, 39.68, 1),
    ) -> Dict[str, torch.Tensor]:
        """Extract per-detection uncertainty from spatial uncertainty maps.

        Maps each detected box center back to the feature map coordinate
        and samples the uncertainty value at that location.

        Args:
            pred_dict: Output from forward().
            det_boxes: (N, 7) Detected bounding boxes.
            det_scores: (N,) Detection confidence scores.
            feature_map_stride: Spatial stride of feature map.
            voxel_size: (vx, vy) voxel dimensions.
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)

        Returns:
            Dictionary with per-detection uncertainties:
                - aleatoric: (N, 7) per-param aleatoric uncertainty
                - epistemic: (N, 7) per-param epistemic uncertainty
                - total_bbox_uncertainty: (N,) scalar uncertainty per detection
                - classification_uncertainty: (N,) Dirichlet-based cls uncertainty
        """
        if det_boxes.shape[0] == 0:
            device = det_boxes.device
            return {
                "aleatoric": torch.zeros((0, 7), device=device),
                "epistemic": torch.zeros((0, 7), device=device),
                "total_bbox_uncertainty": torch.zeros(0, device=device),
                "classification_uncertainty": torch.zeros(0, device=device),
            }

        aleatoric_map = pred_dict["aleatoric_uncertainty"]  # (B, 7, H, W)
        epistemic_map = pred_dict["epistemic_uncertainty"]  # (B, 7, H, W)
        dirichlet_alpha = pred_dict["dirichlet_alpha"]  # (B, K, H, W)

        x_offset = point_cloud_range[0]
        y_offset = point_cloud_range[1]
        stride_x = voxel_size[0] * feature_map_stride
        stride_y = voxel_size[1] * feature_map_stride

        feat_x = ((det_boxes[:, 0] - x_offset) / stride_x).long()
        feat_y = ((det_boxes[:, 1] - y_offset) / stride_y).long()

        H, W = aleatoric_map.shape[2], aleatoric_map.shape[3]
        feat_x = feat_x.clamp(0, W - 1)
        feat_y = feat_y.clamp(0, H - 1)

        aleatoric = aleatoric_map[0, :, feat_y, feat_x].T  # (N, 7)
        epistemic = epistemic_map[0, :, feat_y, feat_x].T  # (N, 7)

        bbox_range = torch.tensor(
            [70.0, 70.0, 4.0, 5.0, 5.0, 3.0, 2 * np.pi],
            device=det_boxes.device,
        )
        norm_aleatoric = aleatoric / (bbox_range ** 2 + 1e-6)
        norm_epistemic = epistemic / (bbox_range ** 2 + 1e-6)
        total_bbox_unc = (norm_aleatoric + norm_epistemic).mean(dim=1)

        dir_alpha = dirichlet_alpha[0, :, feat_y, feat_x].T  # (N, K)
        alpha_0 = dir_alpha.sum(dim=1)  # (N,)
        cls_uncertainty = (
            self.num_classes / alpha_0
        )  # Higher alpha_0 = more confident

        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_bbox_uncertainty": total_bbox_unc,
            "classification_uncertainty": cls_uncertainty,
        }

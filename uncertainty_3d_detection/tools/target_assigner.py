"""
CenterPoint target assignment.

Maps ground truth boxes to feature map locations for computing
regression and classification losses at the correct spatial positions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List


class CenterPointTargetAssigner:
    """Assigns GT boxes to feature map grid cells for CenterPoint-style training.

    For each GT object, the center is projected onto the BEV feature map,
    and regression/classification targets are set at that grid cell.

    Args:
        voxel_size: [vx, vy, vz] in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        feature_map_stride: Total spatial stride from input to feature map.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        feature_map_stride: int = 1,
        num_classes: int = 3,
    ):
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.feature_map_stride = feature_map_stride
        self.num_classes = num_classes

        self.stride_x = voxel_size[0] * feature_map_stride
        self.stride_y = voxel_size[1] * feature_map_stride

    def assign(
        self,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask: torch.Tensor,
        feature_map_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Assign targets for evidential head training.

        Creates dense target maps where each GT center location has
        regression targets, and all other locations are masked out.

        Args:
            gt_boxes: (B, M, 7) GT boxes.
            gt_classes: (B, M) GT class labels (1-indexed).
            gt_mask: (B, M) validity mask.
            feature_map_size: (H, W) of feature map.

        Returns:
            Dictionary with:
                - target_boxes: (B, 7, H, W) regression targets at center locs
                - target_labels: (B, H, W) class labels at center locs (0 = bg)
                - mask: (B, H, W) binary mask (1 = has GT object)
                - center_indices: list of (batch_idx, y, x) for GT centers
        """
        B = gt_boxes.shape[0]
        H, W = feature_map_size
        device = gt_boxes.device

        target_boxes = torch.zeros((B, 7, H, W), device=device)
        target_labels = torch.zeros((B, H, W), dtype=torch.long, device=device)
        mask = torch.zeros((B, H, W), device=device)

        center_indices = []

        for b in range(B):
            for m in range(gt_boxes.shape[1]):
                if gt_mask[b, m] < 0.5:
                    continue

                box = gt_boxes[b, m]  # (7,)
                cls = gt_classes[b, m].item()

                fx = (box[0].item() - self.point_cloud_range[0]) / self.stride_x
                fy = (box[1].item() - self.point_cloud_range[1]) / self.stride_y

                ix = int(fx)
                iy = int(fy)

                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    continue

                target_boxes[b, 0, iy, ix] = box[0] - ix * self.stride_x - self.point_cloud_range[0]
                target_boxes[b, 1, iy, ix] = box[1] - iy * self.stride_y - self.point_cloud_range[1]
                target_boxes[b, 2, iy, ix] = box[2]  # z absolute
                target_boxes[b, 3, iy, ix] = box[3]  # dx
                target_boxes[b, 4, iy, ix] = box[4]  # dy
                target_boxes[b, 5, iy, ix] = box[5]  # dz
                target_boxes[b, 6, iy, ix] = box[6]  # heading

                target_labels[b, iy, ix] = cls
                mask[b, iy, ix] = 1.0

                center_indices.append((b, iy, ix))

        return {
            "target_boxes": target_boxes,
            "target_labels": target_labels,
            "mask": mask,
            "center_indices": center_indices,
        }

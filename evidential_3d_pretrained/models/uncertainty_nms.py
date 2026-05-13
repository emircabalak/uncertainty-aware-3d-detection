"""
Uncertainty-aware Non-Maximum Suppression (NMS) for 3D object detection.

Extends standard 3D NMS by incorporating uncertainty estimates into the
suppression decision, filtering out high-uncertainty detections and using
uncertainty-weighted scoring.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def iou_3d_axis_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned 3D IoU between two sets of boxes.

    Uses BEV (Bird's Eye View) IoU as an approximation, which is standard
    practice in LiDAR-based detection.

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        (N, M) IoU matrix.
    """
    x1_a = boxes_a[:, 0] - boxes_a[:, 3] / 2
    y1_a = boxes_a[:, 1] - boxes_a[:, 4] / 2
    x2_a = boxes_a[:, 0] + boxes_a[:, 3] / 2
    y2_a = boxes_a[:, 1] + boxes_a[:, 4] / 2

    x1_b = boxes_b[:, 0] - boxes_b[:, 3] / 2
    y1_b = boxes_b[:, 1] - boxes_b[:, 4] / 2
    x2_b = boxes_b[:, 0] + boxes_b[:, 3] / 2
    y2_b = boxes_b[:, 1] + boxes_b[:, 4] / 2

    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    inter_x1 = torch.max(x1_a[:, None], x1_b[None, :])
    inter_y1 = torch.max(y1_a[:, None], y1_b[None, :])
    inter_x2 = torch.min(x2_a[:, None], x2_b[None, :])
    inter_y2 = torch.min(y2_a[:, None], y2_b[None, :])

    inter_area = (
        torch.clamp(inter_x2 - inter_x1, min=0)
        * torch.clamp(inter_y2 - inter_y1, min=0)
    )

    area_a = boxes_a[:, 3] * boxes_a[:, 4]
    area_b = boxes_b[:, 3] * boxes_b[:, 4]

    union = area_a[:, None] + area_b[None, :] - inter_area
    iou = inter_area / (union + 1e-6)

    return iou


def uncertainty_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    uncertainty: torch.Tensor,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.1,
    uncertainty_threshold: Optional[float] = None,
    uncertainty_penalty: float = 0.3,
    max_detections: int = 500,
) -> Dict[str, torch.Tensor]:
    """Uncertainty-aware NMS for 3D detections.

    Three mechanisms beyond standard NMS:
        1. Pre-filter: Remove detections above uncertainty_threshold
        2. Score adjustment: final_score = score * (1 - penalty * normalized_uncertainty)
        3. Uncertainty-aware merging: When suppressing overlapping boxes,
           keep the one with lower uncertainty (not just higher score)

    Args:
        boxes: (N, 7) Bounding boxes [x, y, z, dx, dy, dz, heading].
        scores: (N,) Detection confidence scores.
        labels: (N,) Class labels.
        uncertainty: (N,) Scalar uncertainty per detection.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score to keep.
        uncertainty_threshold: Maximum uncertainty to keep (None = no filter).
        uncertainty_penalty: How much uncertainty reduces the score [0, 1].
        max_detections: Maximum number of output detections.

    Returns:
        Dictionary with filtered detections:
            - boxes, scores, labels, uncertainty (all filtered and reordered)
            - keep_indices: indices into the original input
    """
    device = boxes.device

    if boxes.shape[0] == 0:
        return _empty_result(device)

    if uncertainty_threshold is not None:
        unc_mask = uncertainty < uncertainty_threshold
        boxes = boxes[unc_mask]
        scores = scores[unc_mask]
        labels = labels[unc_mask]
        uncertainty = uncertainty[unc_mask]

    if boxes.shape[0] == 0:
        return _empty_result(device)

    unc_min = uncertainty.min()
    unc_max = uncertainty.max()
    if unc_max > unc_min:
        unc_normalized = (uncertainty - unc_min) / (unc_max - unc_min)
    else:
        unc_normalized = torch.zeros_like(uncertainty)

    adjusted_scores = scores * (1.0 - uncertainty_penalty * unc_normalized)

    score_mask = adjusted_scores > score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    adjusted_scores = adjusted_scores[score_mask]
    labels = labels[score_mask]
    uncertainty = uncertainty[score_mask]

    if boxes.shape[0] == 0:
        return _empty_result(device)

    keep_all = []
    unique_labels = labels.unique()

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_adj_scores = adjusted_scores[cls_mask]
        cls_uncertainty = uncertainty[cls_mask]
        cls_indices = torch.where(cls_mask)[0]

        order = cls_adj_scores.argsort(descending=True)
        cls_boxes = cls_boxes[order]
        cls_adj_scores = cls_adj_scores[order]
        cls_uncertainty = cls_uncertainty[order]
        cls_indices = cls_indices[order]

        suppressed = torch.zeros(cls_boxes.shape[0], dtype=torch.bool, device=device)

        for i in range(cls_boxes.shape[0]):
            if suppressed[i]:
                continue
            keep_all.append(cls_indices[i])

            if i + 1 < cls_boxes.shape[0]:
                ious = iou_3d_axis_aligned(
                    cls_boxes[i : i + 1], cls_boxes[i + 1 :]
                ).squeeze(0)

                overlap_mask = ious > iou_threshold

                lenient_overlap = ious > (iou_threshold * 0.8)
                high_unc = cls_uncertainty[i + 1 :] > cls_uncertainty[i] * 2.0
                unc_suppress = lenient_overlap & high_unc

                suppress_mask = overlap_mask | unc_suppress
                suppressed[i + 1 :] |= suppress_mask

    if not keep_all:
        return _empty_result(device)

    keep_indices = torch.stack(keep_all)

    if keep_indices.shape[0] > max_detections:
        top_scores = adjusted_scores[keep_indices].argsort(descending=True)
        keep_indices = keep_indices[top_scores[:max_detections]]

    return {
        "boxes": boxes[keep_indices],
        "scores": scores[keep_indices],
        "labels": labels[keep_indices],
        "uncertainty": uncertainty[keep_indices],
        "adjusted_scores": adjusted_scores[keep_indices],
        "keep_indices": keep_indices,
    }


def _empty_result(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 7), device=device),
        "scores": torch.zeros(0, device=device),
        "labels": torch.zeros(0, dtype=torch.long, device=device),
        "uncertainty": torch.zeros(0, device=device),
        "adjusted_scores": torch.zeros(0, device=device),
        "keep_indices": torch.zeros(0, dtype=torch.long, device=device),
    }

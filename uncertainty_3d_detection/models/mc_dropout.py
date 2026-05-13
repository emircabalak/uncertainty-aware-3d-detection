"""
Monte Carlo Dropout wrapper for 3D object detection models.

Enables epistemic uncertainty estimation by keeping dropout active at inference
and running multiple stochastic forward passes. The variance across passes
captures model uncertainty.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


def enable_dropout(model: nn.Module) -> None:
    """Force all dropout layers into training mode (active) while keeping
    everything else in eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def inject_dropout(model: nn.Module, dropout_rate: float = 0.1,
                   target_layers: Optional[List[str]] = None) -> nn.Module:
    """Inject dropout layers after ReLU/BatchNorm in specified submodules.

    This is needed because many pretrained 3D detectors don't include dropout
    by default. We surgically add it to backbone and neck layers.

    Args:
        model: The detector model.
        dropout_rate: Dropout probability.
        target_layers: List of submodule name prefixes to inject into.
            If None, injects into all Conv2d -> BN -> ReLU sequences.

    Returns:
        Modified model with dropout layers injected.
    """
    for name, module in model.named_modules():
        if target_layers and not any(name.startswith(t) for t in target_layers):
            continue

        if isinstance(module, nn.Sequential):
            new_modules = []
            for child in module.children():
                new_modules.append(child)
                if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.GELU)):
                    new_modules.append(nn.Dropout2d(p=dropout_rate))
            for i, new_mod in enumerate(new_modules):
                if i < len(list(module.children())):
                    continue
                module.add_module(f"dropout_{i}", new_mod)

    return model


class MCDropoutWrapper(nn.Module):
    """Wraps a 3D object detector to support MC Dropout inference.

    During training, behaves exactly like the base model.
    During inference, performs T stochastic forward passes and aggregates
    predictions with uncertainty estimates.

    Args:
        model: Base 3D detector (e.g., CenterPoint from OpenPCDet).
        num_forward_passes: Number of MC samples at inference (T).
        dropout_rate: Dropout probability for injected layers.
        inject_targets: Submodule prefixes to inject dropout into.
    """

    def __init__(
        self,
        model: nn.Module,
        num_forward_passes: int = 20,
        dropout_rate: float = 0.1,
        inject_targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.T = num_forward_passes
        self.dropout_rate = dropout_rate

        if inject_targets is None:
            inject_targets = ["backbone_2d", "dense_head"]
        self.model = inject_dropout(self.model, dropout_rate, inject_targets)

    def forward(self, batch_dict: Dict) -> Dict:
        """Standard forward pass (training mode)."""
        return self.model(batch_dict)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, batch_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Run T stochastic forward passes and aggregate results.

        Returns:
            Dictionary with keys:
                - pred_boxes: (N, 7) Mean predicted boxes [x,y,z,dx,dy,dz,heading]
                - pred_scores: (N,) Mean confidence scores
                - pred_labels: (N,) Predicted class labels
                - bbox_uncertainty: (N, 7) Per-parameter variance of boxes
                - score_uncertainty: (N,) Variance of confidence scores
                - epistemic_uncertainty: (N,) Overall epistemic uncertainty
                    (mean of normalized bbox variances)
                - total_uncertainty: (N,) Combined score + epistemic uncertainty
        """
        self.model.eval()
        enable_dropout(self.model)

        all_boxes = []
        all_scores = []
        all_labels = []

        for t in range(self.T):
            batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                         for k, v in batch_dict.items()}

            result = self.model.predict(batch_copy)
            detections = result.get("detections", [])

            enable_dropout(self.model)

            for det in detections:
                boxes = det.get("final_boxes", det.get("boxes"))
                scores = det.get("final_scores", det.get("scores"))
                labels = det.get("final_labels", det.get("labels"))

                if boxes is not None and boxes.shape[0] > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)

        if not all_boxes:
            return self._empty_predictions(batch_dict)

        return self._aggregate_predictions(all_boxes, all_scores, all_labels)

    def _aggregate_predictions(
        self,
        all_boxes: List[torch.Tensor],
        all_scores: List[torch.Tensor],
        all_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate T forward passes into mean predictions + uncertainty.

        Uses Hungarian matching to align predictions across passes before
        computing statistics, since the order of detections can differ.
        """
        ref_boxes = all_boxes[0]
        ref_scores = all_scores[0]
        ref_labels = all_labels[0]
        N = ref_boxes.shape[0]

        if N == 0:
            device = ref_boxes.device
            return {
                "pred_boxes": ref_boxes,
                "pred_scores": ref_scores,
                "pred_labels": ref_labels,
                "bbox_uncertainty": torch.zeros((0, 7), device=device),
                "score_uncertainty": torch.zeros(0, device=device),
                "epistemic_uncertainty": torch.zeros(0, device=device),
                "total_uncertainty": torch.zeros(0, device=device),
            }

        matched_boxes = [ref_boxes.unsqueeze(0)]
        matched_scores = [ref_scores.unsqueeze(0)]

        for t in range(1, len(all_boxes)):
            aligned_boxes, aligned_scores = self._match_predictions(
                ref_boxes, ref_labels,
                all_boxes[t], all_labels[t], all_scores[t]
            )
            matched_boxes.append(aligned_boxes.unsqueeze(0))
            matched_scores.append(aligned_scores.unsqueeze(0))

        matched_boxes = torch.cat(matched_boxes, dim=0)  # (T, N, 7)
        matched_scores = torch.cat(matched_scores, dim=0)  # (T, N)

        mean_boxes = matched_boxes.mean(dim=0)  # (N, 7)
        mean_scores = matched_scores.mean(dim=0)  # (N,)

        if matched_boxes.shape[0] > 1:
            bbox_variance = matched_boxes.var(dim=0, unbiased=False)  # (N, 7)
            score_variance = matched_scores.var(dim=0, unbiased=False)  # (N,)
        else:
            bbox_variance = torch.zeros_like(mean_boxes)
            score_variance = torch.zeros_like(mean_scores)

        bbox_range = torch.tensor(
            [70.0, 70.0, 4.0, 5.0, 5.0, 3.0, 2 * np.pi],
            device=mean_boxes.device
        )
        normalized_var = bbox_variance / (bbox_range ** 2 + 1e-6)
        epistemic_uncertainty = normalized_var.mean(dim=1)  # (N,)

        total_uncertainty = 0.5 * score_variance + 0.5 * epistemic_uncertainty

        return {
            "pred_boxes": mean_boxes,
            "pred_scores": mean_scores,
            "pred_labels": ref_labels,
            "bbox_uncertainty": bbox_variance,
            "score_uncertainty": score_variance,
            "epistemic_uncertainty": epistemic_uncertainty,
            "total_uncertainty": total_uncertainty,
        }

    def _match_predictions(
        self,
        ref_boxes: torch.Tensor,
        ref_labels: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match predictions to reference boxes using center distance.

        For each reference box, find the nearest predicted box of the same class.
        If no match is found within threshold, use the reference box with score=0
        (indicating maximum uncertainty for that detection).
        """
        N = ref_boxes.shape[0]
        device = ref_boxes.device
        matched_boxes = ref_boxes.clone()
        matched_scores = torch.zeros(N, device=device)
        match_threshold = 4.0  # meters, center distance

        if pred_boxes.shape[0] == 0:
            return matched_boxes, matched_scores

        ref_centers = ref_boxes[:, :2]  # (N, 2)
        pred_centers = pred_boxes[:, :2]  # (M, 2)
        dists = torch.cdist(ref_centers.float(), pred_centers.float())  # (N, M)

        used = set()
        for i in range(N):
            class_mask = pred_labels == ref_labels[i]
            if not class_mask.any():
                continue

            masked_dists = dists[i].clone()
            masked_dists[~class_mask] = float("inf")
            for u in used:
                masked_dists[u] = float("inf")

            min_dist, min_idx = masked_dists.min(dim=0)
            if min_dist < match_threshold:
                matched_boxes[i] = pred_boxes[min_idx]
                matched_scores[i] = pred_scores[min_idx]
                used.add(min_idx.item())

        return matched_boxes, matched_scores

    def _empty_predictions(self, batch_dict: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        return {
            "pred_boxes": torch.zeros((0, 7), device=device),
            "pred_scores": torch.zeros(0, device=device),
            "pred_labels": torch.zeros(0, dtype=torch.long, device=device),
            "bbox_uncertainty": torch.zeros((0, 7), device=device),
            "score_uncertainty": torch.zeros(0, device=device),
            "epistemic_uncertainty": torch.zeros(0, device=device),
            "total_uncertainty": torch.zeros(0, device=device),
        }

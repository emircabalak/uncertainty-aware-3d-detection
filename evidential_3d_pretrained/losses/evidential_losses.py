"""
Loss functions for Evidential Deep Learning in 3D object detection.

Implements Normal-Inverse-Gamma (NIG) loss for regression uncertainty
and Dirichlet loss for classification uncertainty.

References:
    - Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
    - Sensoy et al., "Evidential Deep Learning to Quantify Classification
      Uncertainty" (NeurIPS 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EvidentialRegressionLoss(nn.Module):
    """Normal-Inverse-Gamma (NIG) negative log-likelihood loss.

    For each regression target, the model predicts (γ, ν, α, β) parameterizing
    a NIG prior over the Gaussian likelihood parameters (μ, σ²).

    The loss consists of:
        1. NIG NLL: penalizes wrong predictions, uncertainty-aware
        2. NIG regularizer: penalizes high evidence (ν) on wrong predictions,
           encouraging the model to be uncertain when it's wrong

    Args:
        kl_weight: Weight for the KL divergence regularizer. Start small
            (0.01) and anneal up during training to avoid premature collapse.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, kl_weight: float = 0.01, reduction: str = "mean"):
        super().__init__()
        self.kl_weight = kl_weight
        self.reduction = reduction

    def forward(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute evidential regression loss.

        Args:
            gamma: (*, D) Predicted mean for each regression param.
            nu: (*, D) Virtual observation count (> 0).
            alpha: (*, D) Shape parameter (> 1).
            beta: (*, D) Scale parameter (> 0).
            target: (*, D) Ground truth regression targets.
            mask: (*,) Optional binary mask (1 = valid, 0 = ignore).

        Returns:
            Scalar loss.
        """
        nll = self._nig_nll(gamma, nu, alpha, beta, target)

        reg = self._nig_reg(gamma, nu, alpha, beta, target)

        loss = nll + self.kl_weight * reg

        if mask is not None:
            if mask.dim() == loss.dim() - 1:
                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            loss = loss * mask
            if self.reduction == "mean":
                num_valid = mask.sum().clamp(min=1)
                return loss.sum() / (num_valid * loss.shape[1] + 1e-6)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _nig_nll(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """NIG negative log-likelihood.

        Matches the reference implementation from Amini et al. (NeurIPS 2020):
            NLL = 0.5*log(π/ν) - α*log(Ω)
                  + (α+0.5)*log(ν*(y-γ)² + Ω)
                  + lgamma(α) - lgamma(α+0.5)
        where Ω = 2β(1+ν).
        """
        residual = target - gamma

        omega = 2.0 * beta * (1.0 + nu)

        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=gamma.device))
            - 0.5 * torch.log(nu + 1e-6)
            - alpha * torch.log(omega + 1e-6)
            + (alpha + 0.5) * torch.log(nu * residual ** 2 + omega + 1e-6)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll  # Already the NLL (positive loss)

    def _nig_reg(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Evidence regularizer: penalize high evidence on wrong predictions.

        |y - γ| * (2ν + α), encouraging ν and α to be small when the
        prediction error is large (model should be uncertain when wrong).
        """
        residual = torch.abs(target - gamma)
        reg = residual * (2.0 * nu + alpha)
        return reg


class DirichletClassificationLoss(nn.Module):
    """Dirichlet-based classification loss for uncertainty estimation.

    Instead of cross-entropy over softmax, we place a Dirichlet prior over
    class probabilities. The concentration parameters α_k are predicted by
    the network, and higher total evidence (Σα_k) means more confidence.

    The loss consists of:
        1. Type II maximum likelihood (Bayes risk of cross-entropy)
        2. KL divergence regularizer to push non-target α_k toward 1

    Args:
        num_classes: Number of object classes.
        kl_weight: Weight for KL regularizer (anneal from 0 to 1 during training).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        num_classes: int = 3,
        kl_weight: float = 0.01,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.reduction = reduction

    def forward(
        self,
        dirichlet_alpha: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Dirichlet classification loss.

        Args:
            dirichlet_alpha: (*, K) Dirichlet concentration parameters (> 0).
            target: (*,) or (*, K) Class labels (long) or one-hot.
            mask: (*,) Optional binary mask.

        Returns:
            Scalar loss.
        """
        if target.dim() == dirichlet_alpha.dim() - 1:
            one_hot = F.one_hot(target.long(), self.num_classes).float()
        else:
            one_hot = target.float()

        S = dirichlet_alpha.sum(dim=-1, keepdim=True)  # (*, 1)

        ml_loss = (
            one_hot * (torch.digamma(S) - torch.digamma(dirichlet_alpha))
        ).sum(dim=-1)

        alpha_tilde = one_hot + (1.0 - one_hot) * dirichlet_alpha
        kl = self._kl_dirichlet(alpha_tilde)

        loss = ml_loss + self.kl_weight * kl

        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-6)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _kl_dirichlet(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence between Dir(alpha) and Dir(1, 1, ..., 1).

        D_KL(Dir(α) || Dir(1)) = log B(1) - log B(α)
            + Σ_k (α_k - 1) * [ψ(α_k) - ψ(Σ α_k)]
        """
        K = alpha.shape[-1]
        ones = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=-1, keepdim=True)

        log_B_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(
            S_alpha.squeeze(-1)
        )
        log_B_ones = torch.lgamma(ones).sum(dim=-1) - torch.lgamma(
            torch.tensor(float(K), device=alpha.device)
        )

        kl = (
            log_B_ones
            - log_B_alpha
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(
                dim=-1
            )
        )
        return kl


class CombinedUncertaintyLoss(nn.Module):
    """Combined loss for training evidential 3D detector.

    Total loss = heatmap_loss + bbox_evidential_loss + cls_dirichlet_loss

    The KL weights are annealed during training:
        kl_weight(epoch) = min(1.0, epoch / annealing_epochs) * max_kl_weight

    Args:
        bbox_evidential_weight: Weight for bbox evidential loss.
        cls_dirichlet_weight: Weight for classification Dirichlet loss.
        max_kl_weight: Maximum KL weight after annealing.
        annealing_epochs: Number of epochs to anneal KL weight from 0 to max.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        bbox_evidential_weight: float = 1.0,
        cls_dirichlet_weight: float = 0.5,
        max_kl_weight: float = 0.1,
        annealing_epochs: int = 40,
        num_classes: int = 3,
    ):
        super().__init__()
        self.bbox_evidential_weight = bbox_evidential_weight
        self.cls_dirichlet_weight = cls_dirichlet_weight
        self.max_kl_weight = max_kl_weight
        self.annealing_epochs = annealing_epochs

        self.bbox_loss = EvidentialRegressionLoss(kl_weight=0.0)
        self.cls_loss = DirichletClassificationLoss(
            num_classes=num_classes, kl_weight=0.0
        )

    def update_kl_weight(self, current_epoch: int) -> float:
        """Anneal KL weight based on current epoch."""
        weight = min(1.0, current_epoch / self.annealing_epochs) * self.max_kl_weight
        self.bbox_loss.kl_weight = weight
        self.cls_loss.kl_weight = weight
        return weight

    def forward(
        self,
        pred_dict: dict,
        target_dict: dict,
        current_epoch: int = 0,
    ) -> dict:
        """Compute combined loss.

        Args:
            pred_dict: From EvidentialCenterHead.forward():
                - bbox_pred, nu, alpha, beta, dirichlet_alpha, heatmap
            target_dict: Ground truth:
                - target_boxes: (B, max_objs, 7)
                - target_labels: (B, max_objs)
                - heatmap_target: (B, num_classes, H, W)
                - mask: (B, max_objs)

        Returns:
            Dictionary with individual and total losses.
        """
        kl_w = self.update_kl_weight(current_epoch)

        center_indices = target_dict.get("center_indices", [])

        if len(center_indices) == 0:
            device = pred_dict["bbox_pred"].device
            zero = torch.tensor(0.0, device=device)
            return {
                "total_loss": zero,
                "bbox_evidential_loss": zero,
                "cls_dirichlet_loss": zero,
                "kl_weight": kl_w,
            }

        b_idx = [c[0] for c in center_indices]
        y_idx = [c[1] for c in center_indices]
        x_idx = [c[2] for c in center_indices]

        gamma = pred_dict["bbox_pred"][:, :, y_idx, x_idx]  # wrong: need per-batch
        gamma_list, nu_list, alpha_list, beta_list = [], [], [], []
        target_list, dir_alpha_list, target_cls_list = [], [], []

        for b, y, x in center_indices:
            gamma_list.append(pred_dict["bbox_pred"][b, :, y, x])
            nu_list.append(pred_dict["nu"][b, :, y, x])
            alpha_list.append(pred_dict["alpha"][b, :, y, x])
            beta_list.append(pred_dict["beta"][b, :, y, x])
            target_list.append(target_dict["target_boxes"][b, :, y, x])
            dir_alpha_list.append(pred_dict["dirichlet_alpha"][b, :, y, x])
            target_cls_list.append(target_dict["target_labels"][b, y, x])

        gamma = torch.stack(gamma_list)       # (N, 7)
        nu = torch.stack(nu_list)             # (N, 7)
        alpha = torch.stack(alpha_list)       # (N, 7)
        beta = torch.stack(beta_list)         # (N, 7)
        target = torch.stack(target_list)     # (N, 7)
        dir_alpha = torch.stack(dir_alpha_list)   # (N, K)
        target_cls = torch.stack(target_cls_list) # (N,)

        bbox_loss = self.bbox_loss(
            gamma=gamma, nu=nu, alpha=alpha, beta=beta,
            target=target, mask=None,
        )

        target_cls_0 = (target_cls - 1).clamp(min=0)
        cls_loss = self.cls_loss(
            dirichlet_alpha=dir_alpha,
            target=target_cls_0,
            mask=None,
        )

        total = (
            self.bbox_evidential_weight * bbox_loss
            + self.cls_dirichlet_weight * cls_loss
        )

        return {
            "total_loss": total,
            "bbox_evidential_loss": bbox_loss,
            "cls_dirichlet_loss": cls_loss,
            "kl_weight": kl_w,
        }

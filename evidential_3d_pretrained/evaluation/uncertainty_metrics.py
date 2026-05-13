"""
Uncertainty quality metrics for 3D object detection.

Evaluates whether the model's uncertainty estimates are meaningful:
- Do high-uncertainty predictions correspond to actual errors?
- Does removing uncertain predictions improve overall accuracy?
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def uncertainty_auroc(
    uncertainty: np.ndarray,
    correctness: np.ndarray,
) -> Dict[str, float]:
    """AUROC for uncertainty-based error detection.

    Treats uncertainty as a predictor of incorrect detections.
    High AUROC means the model is uncertain when it's wrong and
    confident when it's right.

    Args:
        uncertainty: (N,) Predicted uncertainty per detection.
        correctness: (N,) Binary: 1=correct (TP), 0=incorrect (FP).

    Returns:
        Dictionary with AUROC score and FPR/TPR curves.
    """
    if len(np.unique(correctness)) < 2:
        return {"auroc": 0.5, "fpr": np.array([0, 1]), "tpr": np.array([0, 1])}

    error_labels = 1 - correctness

    auroc = roc_auc_score(error_labels, uncertainty)
    fpr, tpr, thresholds = roc_curve(error_labels, uncertainty)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def sparsification_plot(
    scores: np.ndarray,
    uncertainty: np.ndarray,
    iou_with_gt: np.ndarray,
    iou_threshold: float = 0.7,
    num_steps: int = 20,
) -> Dict[str, np.ndarray]:
    """Sparsification analysis: remove most uncertain predictions first.

    The key insight: if uncertainty is well-calibrated, removing the most
    uncertain detections first should improve precision faster than
    removing random detections.

    This produces the "oracle" and "uncertainty" sparsification curves.
    The gap between them indicates room for improvement.

    Args:
        scores: (N,) Detection confidence scores.
        uncertainty: (N,) Predicted uncertainty values.
        iou_with_gt: (N,) IoU of each detection with its best-matching GT box.
        iou_threshold: IoU threshold for counting as TP.
        num_steps: Number of removal steps.

    Returns:
        Dictionary with:
            - fractions_removed: (num_steps,) fraction of detections removed
            - precision_by_uncertainty: precision at each step (uncertainty ordering)
            - precision_by_random: expected precision at each step (random ordering)
            - precision_by_oracle: precision at each step (perfect ordering)
            - ause: Area Under Sparsification Error (lower = better)
    """
    N = len(scores)
    if N == 0:
        return _empty_sparsification(num_steps)

    correct = (iou_with_gt >= iou_threshold).astype(float)
    fractions = np.linspace(0, 0.95, num_steps)

    unc_order = np.argsort(-uncertainty)  # Most uncertain first
    precision_unc = []
    for frac in fractions:
        n_remove = int(frac * N)
        keep = unc_order[n_remove:]
        if len(keep) == 0:
            precision_unc.append(precision_unc[-1] if precision_unc else 0)
        else:
            precision_unc.append(correct[keep].mean())

    oracle_order = np.lexsort((iou_with_gt, correct))
    precision_oracle = []
    for frac in fractions:
        n_remove = int(frac * N)
        keep = oracle_order[n_remove:]
        if len(keep) == 0:
            precision_oracle.append(precision_oracle[-1] if precision_oracle else 0)
        else:
            precision_oracle.append(correct[keep].mean())

    precision_random = []
    base_precision = correct.mean()
    for frac in fractions:
        n_keep = N - int(frac * N)
        if n_keep == 0:
            precision_random.append(precision_random[-1] if precision_random else 0)
        else:
            precision_random.append(base_precision)

    precision_unc = np.array(precision_unc)
    precision_oracle = np.array(precision_oracle)
    precision_random = np.array(precision_random)

    sparsification_error = precision_oracle - precision_unc
    ause = np.trapz(sparsification_error, fractions)

    return {
        "fractions_removed": fractions,
        "precision_by_uncertainty": precision_unc,
        "precision_by_random": precision_random,
        "precision_by_oracle": precision_oracle,
        "ause": ause,
    }


def plot_sparsification(
    sparsification_data: Dict[str, np.ndarray],
    title: str = "Sparsification Plot",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize sparsification analysis.

    Shows how precision changes as we remove detections by uncertainty
    vs. randomly vs. with oracle knowledge.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    fracs = sparsification_data["fractions_removed"]

    ax.plot(fracs, sparsification_data["precision_by_oracle"],
            "g-", linewidth=2, label="Oracle (best possible)")
    ax.plot(fracs, sparsification_data["precision_by_uncertainty"],
            "b-", linewidth=2, label="Our uncertainty")
    ax.plot(fracs, sparsification_data["precision_by_random"],
            "r--", linewidth=2, label="Random")

    ax.fill_between(
        fracs,
        sparsification_data["precision_by_oracle"],
        sparsification_data["precision_by_uncertainty"],
        alpha=0.15, color="blue",
        label=f"AUSE = {sparsification_data['ause']:.4f}",
    )

    ax.set_xlabel("Fraction of detections removed", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def negative_log_likelihood(
    predicted_mean: np.ndarray,
    predicted_var: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Gaussian NLL: measures if predicted variance matches actual errors.

    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]

    Lower NLL = better uncertainty calibration.

    Args:
        predicted_mean: (N, D) Predicted values.
        predicted_var: (N, D) Predicted variance (must be > 0).
        targets: (N, D) Ground truth values.

    Returns:
        Mean NLL across all predictions and dimensions.
    """
    predicted_var = np.clip(predicted_var, 1e-6, None)
    nll = 0.5 * (
        np.log(predicted_var) + (targets - predicted_mean) ** 2 / predicted_var
    )
    return float(nll.mean())


def _empty_sparsification(num_steps: int) -> Dict[str, np.ndarray]:
    fracs = np.linspace(0, 0.95, num_steps)
    zeros = np.zeros(num_steps)
    return {
        "fractions_removed": fracs,
        "precision_by_uncertainty": zeros,
        "precision_by_random": zeros,
        "precision_by_oracle": zeros,
        "ause": 0.0,
    }

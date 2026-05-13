"""
Calibration metrics for uncertainty evaluation.

Measures how well the predicted uncertainty/confidence aligns with
actual detection accuracy. A well-calibrated model's 90% confidence
predictions should be correct ~90% of the time.

References:
    - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class ExpectedCalibrationError:
    """Expected Calibration Error (ECE).

    ECE = Σ (|B_m| / N) * |accuracy(B_m) - confidence(B_m)|

    where B_m is the set of predictions in confidence bin m.

    Lower ECE = better calibrated. Perfect calibration = 0.

    Args:
        num_bins: Number of equal-width confidence bins.
    """

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins

    def compute(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> dict:
        """Compute ECE and related calibration statistics.

        Args:
            confidences: (N,) Predicted confidence/score for each detection.
            correctness: (N,) Binary: 1 if detection is correct (TP), 0 if FP.

        Returns:
            Dictionary with:
                - ece: Expected Calibration Error
                - mce: Maximum Calibration Error
                - bin_confidences: (num_bins,) mean confidence per bin
                - bin_accuracies: (num_bins,) mean accuracy per bin
                - bin_counts: (num_bins,) samples per bin
        """
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_confidences = np.zeros(self.num_bins)
        bin_accuracies = np.zeros(self.num_bins)
        bin_counts = np.zeros(self.num_bins)

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = mask.sum()

            if count > 0:
                bin_confidences[i] = confidences[mask].mean()
                bin_accuracies[i] = correctness[mask].mean()
                bin_counts[i] = count

        total = confidences.shape[0]
        ece = np.sum(
            (bin_counts / total) * np.abs(bin_accuracies - bin_confidences)
        )

        nonempty = bin_counts > 0
        if nonempty.any():
            mce = np.max(np.abs(bin_accuracies[nonempty] - bin_confidences[nonempty]))
        else:
            mce = 0.0

        return {
            "ece": ece,
            "mce": mce,
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }


def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot reliability diagram showing calibration quality.

    A perfectly calibrated model follows the diagonal (identity line).
    Bars above diagonal = overconfident, below = underconfident.

    Args:
        confidences: (N,) Predicted confidence scores.
        correctness: (N,) Binary correctness labels.
        num_bins: Number of confidence bins.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    ece_calc = ExpectedCalibrationError(num_bins)
    results = ece_calc.compute(confidences, correctness)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    bin_centers = np.linspace(1 / (2 * num_bins), 1 - 1 / (2 * num_bins), num_bins)
    width = 1.0 / num_bins

    ax1.bar(bin_centers, results["bin_accuracies"], width=width * 0.8,
            color="#3274A1", edgecolor="black", linewidth=0.5,
            label="Accuracy", alpha=0.8)

    gaps = results["bin_accuracies"] - results["bin_confidences"]
    gap_colors = ["#E1812C" if g < 0 else "#3A923A" for g in gaps]
    ax1.bar(bin_centers, np.abs(gaps), bottom=np.minimum(
        results["bin_accuracies"], results["bin_confidences"]),
        width=width * 0.8, color=gap_colors, alpha=0.3, label="Gap")

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"{title}\nECE = {results['ece']:.4f}, "
                  f"MCE = {results['mce']:.4f}", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, results["bin_counts"], width=width * 0.8,
            color="#3274A1", edgecolor="black", linewidth=0.5, alpha=0.6)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Samples per bin", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved to {save_path}")

    return fig

"""
Evaluation script for Uncertainty-aware CenterPoint.

Runs inference on KITTI val set with both standard detection metrics (mAP)
and uncertainty quality metrics (ECE, AUROC, Sparsification).

Supports:
  - Evidential single-pass inference
  - MC Dropout multi-pass inference
  - Combined mode
  - Full uncertainty quality analysis

Usage:
    python tools/evaluate.py --config configs/centerpoint_kitti.yaml \
        --checkpoint output/.../best_model.pth \
        --mode evidential
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.uncertainty_centerpoint import UncertaintyCenterPoint
from models.mc_dropout import MCDropoutWrapper
from tools.kitti_dataset import KITTIDataset, collate_batch, LABEL_MAP
from evaluation.calibration import ExpectedCalibrationError, reliability_diagram
from evaluation.uncertainty_metrics import (
    uncertainty_auroc,
    sparsification_plot,
    plot_sparsification,
    negative_log_likelihood,
)
from visualization.vis_utils import (
    visualize_bev_uncertainty,
    plot_uncertainty_histogram,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="evidential",
                        choices=["evidential", "mc_dropout", "combined"])
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_viz", type=int, default=10,
                        help="Number of samples to visualize")
    return parser.parse_args()


def compute_iou_with_gt(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Compute BEV IoU between each prediction and its best-matching GT box.

    Returns: (N,) IoU values for each predicted box.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros(pred_boxes.shape[0])

    ious = np.zeros(pred_boxes.shape[0])

    for i in range(pred_boxes.shape[0]):
        px, py, _, pdx, pdy = pred_boxes[i, :5]

        best_iou = 0
        for j in range(gt_boxes.shape[0]):
            gx, gy, _, gdx, gdy = gt_boxes[j, :5]

            inter_x = max(0, min(px + pdx/2, gx + gdx/2) - max(px - pdx/2, gx - gdx/2))
            inter_y = max(0, min(py + pdy/2, gy + gdy/2) - max(py - pdy/2, gy - gdy/2))
            inter_area = inter_x * inter_y
            union_area = pdx * pdy + gdx * gdy - inter_area

            iou = inter_area / (union_area + 1e-6)
            best_iou = max(best_iou, iou)

        ious[i] = best_iou

    return ious


def compute_ap(
    scores: np.ndarray,
    is_tp: np.ndarray,
    num_gt: int,
    num_recall_points: int = 41,
) -> float:
    """Compute Average Precision using KITTI-style 41-point interpolation."""
    if num_gt == 0:
        return 0.0

    sorted_indices = np.argsort(-scores)
    is_tp_sorted = is_tp[sorted_indices]

    tp_cumsum = np.cumsum(is_tp_sorted)
    fp_cumsum = np.cumsum(1 - is_tp_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    recall_points = np.linspace(0, 1, num_recall_points)
    interp_precision = np.zeros(num_recall_points)

    for i, r in enumerate(recall_points):
        mask = recall >= r
        if mask.any():
            interp_precision[i] = precision[mask].max()

    return interp_precision.mean()


def evaluate_detection(
    all_pred_boxes, all_pred_scores, all_pred_labels,
    all_gt_boxes, all_gt_classes,
    iou_thresholds: dict,
    classes: list,
) -> dict:
    """Compute mAP for each class and difficulty."""
    results = {}

    for cls_name in classes:
        cls_id = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}[cls_name]
        iou_thresh = iou_thresholds[cls_name]

        all_scores = []
        all_tp = []
        total_gt = 0

        for i in range(len(all_pred_boxes)):
            pred_mask = all_pred_labels[i] == cls_id
            gt_mask = all_gt_classes[i] == cls_id

            pred_boxes = all_pred_boxes[i][pred_mask]
            pred_scores = all_pred_scores[i][pred_mask]
            gt_boxes = all_gt_boxes[i][gt_mask]

            total_gt += gt_boxes.shape[0]

            if pred_boxes.shape[0] == 0:
                continue

            ious = compute_iou_with_gt(pred_boxes, gt_boxes)
            is_tp = (ious >= iou_thresh).astype(float)

            all_scores.append(pred_scores)
            all_tp.append(is_tp)

        if all_scores:
            scores = np.concatenate(all_scores)
            tp = np.concatenate(all_tp)
            ap = compute_ap(scores, tp, total_gt)
        else:
            ap = 0.0

        results[cls_name] = {"AP": ap, "num_gt": total_gt}

    results["mAP"] = np.mean([v["AP"] for v in results.values() if "AP" in v])
    return results


@torch.no_grad()
def run_evaluation(args, config):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"].copy()
    model_cfg["voxel_size"] = config["voxel"]["voxel_size"]
    model_cfg["point_cloud_range"] = config["voxel"]["point_cloud_range"]
    model = UncertaintyCenterPoint(model_cfg)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")

    if args.mode in ["mc_dropout", "combined"]:
        model = MCDropoutWrapper(
            model, num_forward_passes=args.mc_passes,
            dropout_rate=config["model"].get("dropout_rate", 0.1),
        )
        print(f"MC Dropout mode: {args.mc_passes} forward passes")

    from torch.utils.data import DataLoader
    val_dataset = KITTIDataset(
        data_path=config["data"]["data_path"],
        split="val",
        classes=config["data"]["classes"],
        point_cloud_range=config["voxel"]["point_cloud_range"],
        voxel_size=config["voxel"]["voxel_size"],
        max_points_per_voxel=config["voxel"]["max_points_per_voxel"],
        max_num_voxels=config["voxel"]["max_num_voxels"]["eval"],
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_batch,
    )

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_pred_uncertainty = []
    all_gt_boxes = []
    all_gt_classes = []
    all_points = []

    print(f"\nRunning inference on {len(val_dataset)} samples...")

    for batch_dict in tqdm(val_loader, desc="Evaluating"):
        for key in ["voxels", "voxel_coords", "voxel_num_points",
                     "gt_boxes", "gt_classes", "gt_mask"]:
            if key in batch_dict and isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to(device)

        if args.mode == "mc_dropout":
            result = model.predict_with_uncertainty(batch_dict)
            pred_boxes = result["pred_boxes"].cpu().numpy()
            pred_scores = result["pred_scores"].cpu().numpy()
            pred_labels = result["pred_labels"].cpu().numpy()
            uncertainty = result["total_uncertainty"].cpu().numpy()
        else:
            result = model.predict(batch_dict)
            det = result["detections"][0]
            pred_boxes = det.get("final_boxes", det["boxes"]).cpu().numpy()
            pred_scores = det.get("final_scores", det["scores"]).cpu().numpy()
            pred_labels = det.get("final_labels", det["labels"]).cpu().numpy()
            uncertainty = det.get("final_uncertainty",
                                  det.get("total_bbox_uncertainty",
                                           torch.zeros(pred_boxes.shape[0])
                                           )).cpu().numpy()

        gt_boxes = batch_dict["gt_boxes"][0].cpu().numpy()
        gt_classes = batch_dict["gt_classes"][0].cpu().numpy()
        gt_mask = batch_dict["gt_mask"][0].cpu().numpy()

        valid_gt = gt_mask > 0.5
        gt_boxes = gt_boxes[valid_gt]
        gt_classes = gt_classes[valid_gt]

        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_pred_labels.append(pred_labels)
        all_pred_uncertainty.append(uncertainty)
        all_gt_boxes.append(gt_boxes)
        all_gt_classes.append(gt_classes)

    # ===================== Detection Metrics =====================
    print("\n" + "=" * 60)
    print("DETECTION RESULTS (mAP)")
    print("=" * 60)

    iou_thresholds = config["eval"]["iou_thresholds"]
    det_results = evaluate_detection(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_classes,
        iou_thresholds, config["data"]["classes"],
    )

    for cls_name in config["data"]["classes"]:
        r = det_results[cls_name]
        print(f"  {cls_name:12s}: AP = {r['AP']:.4f}  (GT: {r['num_gt']})")
    print(f"  {'mAP':12s}: {det_results['mAP']:.4f}")

    # ===================== Uncertainty Metrics =====================
    print("\n" + "=" * 60)
    print("UNCERTAINTY QUALITY METRICS")
    print("=" * 60)

    all_scores_flat = np.concatenate(all_pred_scores)
    all_unc_flat = np.concatenate(all_pred_uncertainty)

    all_ious = []
    for i in range(len(all_pred_boxes)):
        ious = compute_iou_with_gt(all_pred_boxes[i], all_gt_boxes[i])
        all_ious.append(ious)
    all_ious_flat = np.concatenate(all_ious)

    is_correct = (all_ious_flat >= 0.5).astype(float)

    if len(all_scores_flat) > 0 and len(np.unique(is_correct)) >= 2:
        ece_calc = ExpectedCalibrationError(num_bins=15)
        ece_result = ece_calc.compute(all_scores_flat, is_correct)
        print(f"  ECE: {ece_result['ece']:.4f}")
        print(f"  MCE: {ece_result['mce']:.4f}")

        auroc_result = uncertainty_auroc(all_unc_flat, is_correct)
        print(f"  AUROC (uncertainty→error): {auroc_result['auroc']:.4f}")

        spars_result = sparsification_plot(
            all_scores_flat, all_unc_flat, all_ious_flat,
            iou_threshold=0.5, num_steps=20,
        )
        print(f"  AUSE (sparsification error): {spars_result['ause']:.4f}")

        tp_unc = all_unc_flat[is_correct == 1]
        fp_unc = all_unc_flat[is_correct == 0]
        if len(tp_unc) > 0 and len(fp_unc) > 0:
            print(f"  TP uncertainty (mean±std): {tp_unc.mean():.4f} ± {tp_unc.std():.4f}")
            print(f"  FP uncertainty (mean±std): {fp_unc.mean():.4f} ± {fp_unc.std():.4f}")

        eval_dir = config["paths"]["eval_dir"]
        os.makedirs(eval_dir, exist_ok=True)

        reliability_diagram(
            all_scores_flat, is_correct, num_bins=15,
            title=f"Reliability Diagram ({args.mode})",
            save_path=os.path.join(eval_dir, "reliability_diagram.png"),
        )

        plot_sparsification(
            spars_result,
            title=f"Sparsification Plot ({args.mode})",
            save_path=os.path.join(eval_dir, "sparsification_plot.png"),
        )

        if len(tp_unc) > 0 and len(fp_unc) > 0:
            plot_uncertainty_histogram(
                tp_unc, fp_unc,
                title=f"Uncertainty Distribution ({args.mode})",
                save_path=os.path.join(eval_dir, "uncertainty_histogram.png"),
            )

        print(f"\nPlots saved to: {eval_dir}")

    # ===================== Visualization =====================
    if args.visualize:
        viz_dir = config["paths"]["viz_dir"]
        os.makedirs(viz_dir, exist_ok=True)
        print(f"\nSaving {args.num_viz} BEV visualizations to {viz_dir}...")

        for i in range(min(args.num_viz, len(all_pred_boxes))):
            if all_pred_boxes[i].shape[0] > 0:
                # Get points from dataset
                sample = val_dataset[i]
                points = sample["points"]

                visualize_bev_uncertainty(
                    points=points,
                    boxes=all_pred_boxes[i],
                    scores=all_pred_scores[i],
                    labels=all_pred_labels[i],
                    uncertainty=all_pred_uncertainty[i],
                    gt_boxes=all_gt_boxes[i],
                    title=f"Sample {val_dataset.sample_ids[i]}",
                    save_path=os.path.join(viz_dir, f"bev_{val_dataset.sample_ids[i]}.png"),
                )

    # ===================== Summary =====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  mAP: {det_results['mAP']:.4f}")
    if len(all_scores_flat) > 0 and len(np.unique(is_correct)) >= 2:
        print(f"  ECE: {ece_result['ece']:.4f}")
        print(f"  AUROC: {auroc_result['auroc']:.4f}")
        print(f"  AUSE: {spars_result['ause']:.4f}")
    print("=" * 60)

    return det_results


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_evaluation(args, config)


if __name__ == "__main__":
    main()

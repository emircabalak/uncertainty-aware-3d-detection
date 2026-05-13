"""
Evaluation script for Evidential Pretrained Detector.

Runs inference on KITTI val set with:
  - KITTI-style mAP (3-class, 41-point AP interpolation)
  - Uncertainty quality metrics (ECE, AUROC, Sparsification/AUSE)
  - Reliability diagram + sparsification plot + uncertainty histogram
  - Optional BEV visualizations with uncertainty-colored boxes

MC Dropout path intentionally removed — pretrained backbone was not trained
with dropout, so post-hoc injection corrupts predictions (verified on Model 1,
mAP dropped from 37% to 0.6%). Evidential single-pass uncertainty is the
supported mode.

Usage:
    python tools/evaluate.py --config configs/pretrained_kitti.yaml \
        --checkpoint output/pretrained_kitti_evidential/checkpoints/best_model.pth
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

from models.uncertainty_detector import EvidentialPretrainedDetector
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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_viz", type=int, default=10)
    return parser.parse_args()


def compute_iou_with_gt(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
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


def compute_ap(scores, is_tp, num_gt, num_recall_points=41):
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


def evaluate_detection(all_pred_boxes, all_pred_scores, all_pred_labels,
                       all_gt_boxes, all_gt_classes,
                       iou_thresholds, classes):
    results = {}
    for cls_name in classes:
        cls_id = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}[cls_name]
        iou_thresh = iou_thresholds[cls_name]
        all_scores, all_tp, total_gt = [], [], 0
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

    # Build model
    model_cfg = config["model"].copy()
    model_cfg["voxel_size"] = config["voxel"]["voxel_size"]
    model_cfg["point_cloud_range"] = config["voxel"]["point_cloud_range"]
    model = EvidentialPretrainedDetector(model_cfg)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if "stage" in checkpoint:
        print(f"  Training stage when saved: {checkpoint['stage']}")

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
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=collate_batch)

    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_pred_uncertainty = []
    all_gt_boxes, all_gt_classes = [], []

    print(f"\nRunning inference on {len(val_dataset)} samples...")
    for batch_dict in tqdm(val_loader, desc="Evaluating"):
        for key in ["voxels", "voxel_coords", "voxel_num_points",
                     "gt_boxes", "gt_classes", "gt_mask"]:
            if key in batch_dict and isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to(device)

        result = model.predict(batch_dict)
        det = result["detections"][0]
        pred_boxes = det.get("final_boxes", det["boxes"]).cpu().numpy()
        pred_scores = det.get("final_scores", det["scores"]).cpu().numpy()
        pred_labels = det.get("final_labels", det["labels"]).cpu().numpy()
        uncertainty = det.get("final_uncertainty",
                              det.get("total_bbox_uncertainty",
                                      torch.zeros(pred_boxes.shape[0]))).cpu().numpy()

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

    print("\n" + "=" * 60)
    print("DETECTION RESULTS (mAP)")
    print("=" * 60)
    det_results = evaluate_detection(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_classes,
        config["eval"]["iou_thresholds"], config["data"]["classes"],
    )
    for cls_name in config["data"]["classes"]:
        r = det_results[cls_name]
        print(f"  {cls_name:12s}: AP = {r['AP']:.4f}  (GT: {r['num_gt']})")
    print(f"  {'mAP':12s}: {det_results['mAP']:.4f}")

    print("\n" + "=" * 60)
    print("UNCERTAINTY QUALITY METRICS")
    print("=" * 60)
    all_scores_flat = np.concatenate(all_pred_scores) if all_pred_scores else np.array([])
    all_unc_flat = np.concatenate(all_pred_uncertainty) if all_pred_uncertainty else np.array([])
    all_ious = [compute_iou_with_gt(all_pred_boxes[i], all_gt_boxes[i])
                for i in range(len(all_pred_boxes))]
    all_ious_flat = np.concatenate(all_ious) if all_ious else np.array([])
    is_correct = (all_ious_flat >= 0.5).astype(float) if all_ious_flat.size else np.array([])

    ece_result = auroc_result = spars_result = None
    if all_scores_flat.size > 0 and len(np.unique(is_correct)) >= 2:
        ece_result = ExpectedCalibrationError(num_bins=15).compute(
            all_scores_flat, is_correct)
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
            title="Reliability Diagram (evidential, pretrained)",
            save_path=os.path.join(eval_dir, "reliability_diagram.png"),
        )
        plot_sparsification(
            spars_result,
            title="Sparsification Plot (evidential, pretrained)",
            save_path=os.path.join(eval_dir, "sparsification_plot.png"),
        )
        if len(tp_unc) > 0 and len(fp_unc) > 0:
            plot_uncertainty_histogram(
                tp_unc, fp_unc,
                title="Uncertainty Distribution (evidential, pretrained)",
                save_path=os.path.join(eval_dir, "uncertainty_histogram.png"),
            )
        print(f"\nPlots saved to: {eval_dir}")

    if args.visualize:
        viz_dir = config["paths"]["viz_dir"]
        os.makedirs(viz_dir, exist_ok=True)
        print(f"\nSaving {args.num_viz} BEV visualizations to {viz_dir}...")
        for i in range(min(args.num_viz, len(all_pred_boxes))):
            if all_pred_boxes[i].shape[0] > 0:
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
                    save_path=os.path.join(
                        viz_dir, f"bev_{val_dataset.sample_ids[i]}.png"),
                )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: EvidentialPretrainedDetector (pretrained backbone)")
    print(f"  mAP: {det_results['mAP']:.4f}")
    if ece_result:
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

"""
Training script for Uncertainty-aware CenterPoint on KITTI.

Supports:
  - Training from scratch or resuming from checkpoint
  - Evidential loss with KL annealing
  - OneCycle learning rate schedule
  - TensorBoard and W&B logging
  - Automatic checkpointing with best-model tracking
  - Multi-GPU via DataParallel (if available)

Usage:
    python tools/train.py --config configs/centerpoint_kitti.yaml
    python tools/train.py --config configs/centerpoint_kitti.yaml --resume path/to/ckpt.pth
"""

import os
import sys
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.uncertainty_centerpoint import UncertaintyCenterPoint
from models.evidential_head import EvidentialCenterHead
from losses.evidential_losses import CombinedUncertaintyLoss
from tools.kitti_dataset import KITTIDataset, collate_batch
from tools.target_assigner import CenterPointTargetAssigner


def parse_args():
    parser = argparse.ArgumentParser(description="Train Uncertainty CenterPoint")
    parser.add_argument("--config", type=str,
                        default="configs/centerpoint_kitti.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Load model weights only (no optimizer/scheduler), start fresh training")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str,
                        default="uncertainty-3d-detection")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_dataloader(config: dict, split: str) -> DataLoader:
    """Build KITTI dataloader for specified split."""
    data_cfg = config["data"]
    voxel_cfg = config["voxel"]
    train_cfg = config["train"]

    is_train = split == "train"
    augmentation = data_cfg.get("augmentation") if is_train else None

    dataset = KITTIDataset(
        data_path=data_cfg["data_path"],
        split=split,
        classes=data_cfg["classes"],
        point_cloud_range=voxel_cfg["point_cloud_range"],
        voxel_size=voxel_cfg["voxel_size"],
        max_points_per_voxel=voxel_cfg["max_points_per_voxel"],
        max_num_voxels=voxel_cfg["max_num_voxels"]["train" if is_train else "eval"],
        augmentation=augmentation,
        gt_database_path=augmentation.get("gt_sampling", {}).get(
            "db_info_path") if augmentation else None,
    )

    batch_size = train_cfg["batch_size"] if is_train else config["eval"]["batch_size"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader


def build_optimizer(model: nn.Module, config: dict):
    """Build optimizer from config."""
    opt_cfg = config["train"]["optimizer"]
    if opt_cfg["type"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )
    elif opt_cfg["type"].lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    return optimizer


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Build LR scheduler from config."""
    sched_cfg = config["train"]["scheduler"]
    total_epochs = config["train"]["epochs"]

    if sched_cfg["type"] == "one_cycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sched_cfg["max_lr"],
            total_steps=total_epochs * steps_per_epoch,
            pct_start=sched_cfg["pct_start"],
            div_factor=sched_cfg["div_factor"],
            final_div_factor=sched_cfg["final_div_factor"],
        )
    elif sched_cfg["type"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs * steps_per_epoch,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return scheduler


def generate_heatmap_target(
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int,
    feature_map_size: tuple,
    voxel_size: list,
    point_cloud_range: list,
    feature_map_stride: int,
    gaussian_sigma: float = 1.0,
) -> torch.Tensor:
    """Generate Gaussian heatmap targets for CenterPoint.

    Places a 2D Gaussian at each GT object center on the BEV feature map.

    Args:
        gt_boxes: (B, M, 7) GT boxes in LiDAR coordinates.
        gt_classes: (B, M) GT class labels (1-indexed).
        gt_mask: (B, M) validity mask.
        num_classes: Number of classes.
        feature_map_size: (H, W) of the BEV feature map.
        voxel_size: [vx, vy, vz].
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        feature_map_stride: Spatial stride of feature map.
        gaussian_sigma: Standard deviation of Gaussian in pixels.

    Returns:
        heatmap: (B, num_classes, H, W) target heatmap.
    """
    B = gt_boxes.shape[0]
    H, W = feature_map_size
    device = gt_boxes.device

    heatmap = torch.zeros((B, num_classes, H, W), device=device)

    stride_x = voxel_size[0] * feature_map_stride
    stride_y = voxel_size[1] * feature_map_stride

    for b in range(B):
        for m in range(gt_boxes.shape[1]):
            if gt_mask[b, m] < 0.5:
                continue

            cls = int(gt_classes[b, m].item()) - 1  # Convert to 0-indexed
            if cls < 0 or cls >= num_classes:
                continue

            cx = (gt_boxes[b, m, 0].item() - point_cloud_range[0]) / stride_x
            cy = (gt_boxes[b, m, 1].item() - point_cloud_range[1]) / stride_y

            ix, iy = int(cx), int(cy)
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            dx = gt_boxes[b, m, 3].item() / stride_x
            dy = gt_boxes[b, m, 4].item() / stride_y
            radius = max(int(np.ceil(max(dx, dy) / 2)), 1)

            sigma = max(radius / 3, gaussian_sigma)
            size = 2 * radius + 1
            x_grid = torch.arange(size, device=device).float() - radius
            y_grid = torch.arange(size, device=device).float() - radius
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
            gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

            x_start = max(0, ix - radius)
            x_end = min(W, ix + radius + 1)
            y_start = max(0, iy - radius)
            y_end = min(H, iy + radius + 1)

            g_x_start = max(0, radius - ix)
            g_x_end = g_x_start + (x_end - x_start)
            g_y_start = max(0, radius - iy)
            g_y_end = g_y_start + (y_end - y_start)

            heatmap[b, cls, y_start:y_end, x_start:x_end] = torch.max(
                heatmap[b, cls, y_start:y_end, x_start:x_end],
                gaussian[g_y_start:g_y_end, g_x_start:g_x_end],
            )

    return heatmap


class FocalLoss(nn.Module):
    """Focal loss for heatmap training (handles class imbalance)."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = (
            -((1 - target) ** self.beta)
            * (pred ** self.alpha)
            * torch.log(1 - pred)
            * neg_mask
        )

        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: CombinedUncertaintyLoss,
    focal_loss: FocalLoss,
    target_assigner: CenterPointTargetAssigner,
    epoch: int,
    config: dict,
    device: torch.device,
    logger=None,
    save_fn=None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss_sum = 0
    heatmap_loss_sum = 0
    bbox_loss_sum = 0
    cls_loss_sum = 0
    num_batches = 0

    loss_cfg = config["train"]["loss"]
    intra_save_interval = config["train"]["checkpoint"].get("intra_epoch_save_interval", 200)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch",
                dynamic_ncols=True, leave=True)

    for batch_idx, batch_dict in enumerate(pbar):
        for key in ["voxels", "voxel_coords", "voxel_num_points",
                     "gt_boxes", "gt_classes", "gt_mask"]:
            if key in batch_dict and isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to(device)

        output = model(batch_dict)
        pred_dict = output["pred_dict"]

        heatmap = pred_dict["heatmap"]
        B, K, H, W = heatmap.shape

        heatmap_target = generate_heatmap_target(
            gt_boxes=batch_dict["gt_boxes"],
            gt_classes=batch_dict["gt_classes"],
            gt_mask=batch_dict["gt_mask"],
            num_classes=config["model"]["num_classes"],
            feature_map_size=(H, W),
            voxel_size=config["voxel"]["voxel_size"],
            point_cloud_range=config["voxel"]["point_cloud_range"],
            feature_map_stride=model.feature_map_stride
            if not isinstance(model, nn.DataParallel)
            else model.module.feature_map_stride,
        )

        target_dict = target_assigner.assign(
            batch_dict["gt_boxes"],
            batch_dict["gt_classes"],
            batch_dict["gt_mask"],
            feature_map_size=(H, W),
        )

        hm_loss = focal_loss(heatmap, heatmap_target)

        evidential_losses = loss_fn(pred_dict, target_dict, current_epoch=epoch)

        total_loss = (
            loss_cfg["heatmap_weight"] * hm_loss
            + evidential_losses["total_loss"]
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        total_loss_sum += total_loss.item()
        heatmap_loss_sum += hm_loss.item()
        bbox_loss_sum += evidential_losses["bbox_evidential_loss"].item()
        cls_loss_sum += evidential_losses["cls_dirichlet_loss"].item()
        num_batches += 1

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "loss": f"{total_loss.item():.3f}",
            "hm": f"{hm_loss.item():.3f}",
            "bbox": f"{evidential_losses['bbox_evidential_loss'].item():.3f}",
            "cls": f"{evidential_losses['cls_dirichlet_loss'].item():.3f}",
            "lr": f"{lr:.5f}",
        })

        if logger:
            step = epoch * len(dataloader) + batch_idx
            logger.add_scalar("train/total_loss", total_loss.item(), step)
            logger.add_scalar("train/heatmap_loss", hm_loss.item(), step)
            logger.add_scalar("train/bbox_loss",
                              evidential_losses["bbox_evidential_loss"].item(), step)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        if save_fn and (batch_idx + 1) % intra_save_interval == 0:
            step_metrics = {
                "total_loss": total_loss_sum / num_batches,
                "heatmap_loss": heatmap_loss_sum / num_batches,
            }
            save_fn(epoch, step_metrics, tag=f"epoch{epoch+1}_step{batch_idx+1}")

    avg_metrics = {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
        "bbox_loss": bbox_loss_sum / max(num_batches, 1),
        "cls_loss": cls_loss_sum / max(num_batches, 1),
    }

    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    focal_loss: FocalLoss,
    target_assigner: CenterPointTargetAssigner,
    config: dict,
    device: torch.device,
) -> dict:
    """Run validation and compute metrics."""
    model.eval()
    total_loss_sum = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Validation", unit="batch",
                dynamic_ncols=True, leave=False)

    for batch_dict in pbar:
        for key in ["voxels", "voxel_coords", "voxel_num_points",
                     "gt_boxes", "gt_classes", "gt_mask"]:
            if key in batch_dict and isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = batch_dict[key].to(device)

        output = model(batch_dict)
        pred_dict = output["pred_dict"]

        heatmap = pred_dict["heatmap"]
        B, K, H, W = heatmap.shape

        heatmap_target = generate_heatmap_target(
            gt_boxes=batch_dict["gt_boxes"],
            gt_classes=batch_dict["gt_classes"],
            gt_mask=batch_dict["gt_mask"],
            num_classes=config["model"]["num_classes"],
            feature_map_size=(H, W),
            voxel_size=config["voxel"]["voxel_size"],
            point_cloud_range=config["voxel"]["point_cloud_range"],
            feature_map_stride=model.feature_map_stride
            if not isinstance(model, nn.DataParallel)
            else model.module.feature_map_stride,
        )

        hm_loss = focal_loss(heatmap, heatmap_target)
        total_loss_sum += hm_loss.item()
        num_batches += 1
        pbar.set_postfix({"val_loss": f"{total_loss_sum / num_batches:.4f}"})

    return {
        "val_loss": total_loss_sum / max(num_batches, 1),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """Save training checkpoint for resuming."""
    state = {
        "epoch": epoch,
        "model_state_dict": (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"  Checkpoint saved: {save_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["epochs"] = args.epochs

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    for path_key in ["output_dir", "log_dir", "checkpoint_dir", "eval_dir", "viz_dir"]:
        os.makedirs(config["paths"][path_key], exist_ok=True)

    print("\n--- Building dataloaders ---")
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")

    print("\n--- Building model ---")
    model_cfg = config["model"].copy()
    model_cfg["voxel_size"] = config["voxel"]["voxel_size"]
    model_cfg["point_cloud_range"] = config["voxel"]["point_cloud_range"]
    model = UncertaintyCenterPoint(model_cfg)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    loss_fn = CombinedUncertaintyLoss(
        bbox_evidential_weight=config["train"]["loss"]["bbox_evidential_weight"],
        cls_dirichlet_weight=config["train"]["loss"]["cls_dirichlet_weight"],
        max_kl_weight=config["train"]["loss"]["max_kl_weight"],
        annealing_epochs=config["train"]["loss"]["kl_annealing_epochs"],
        num_classes=config["model"]["num_classes"],
    )

    focal_loss = FocalLoss()

    target_assigner = CenterPointTargetAssigner(
        voxel_size=config["voxel"]["voxel_size"],
        point_cloud_range=config["voxel"]["point_cloud_range"],
        feature_map_stride=1,  # BEVBackbone2D upsamples back to pillar grid resolution
        num_classes=config["model"]["num_classes"],
    )

    logger = None
    try:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(config["paths"]["log_dir"])
    except ImportError:
        print("TensorBoard not available, skipping logging")

    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=config)
            wandb.watch(model)
        except ImportError:
            print("W&B not available")

    start_epoch = 0
    best_val_loss = float("inf")

    if args.finetune and os.path.exists(args.finetune):
        print(f"\nFinetuning from: {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights (epoch {checkpoint['epoch']+1}), starting fresh training for {config['train']['epochs']} epochs")
    else:
        resume_path = args.resume or config["train"]["checkpoint"].get("resume_from")

        if not resume_path or not os.path.exists(resume_path):
            ckpt_dir = config["paths"]["checkpoint_dir"]
            import glob, re
            def _epoch_num(p):
                m = re.search(r"checkpoint_epoch_(\d+)\.pth", os.path.basename(p))
                return int(m.group(1)) if m else -1

            epoch_ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")),
                key=_epoch_num,
            )
            latest_ckpt = os.path.join(ckpt_dir, "latest.pth")

            best_candidate = None
            if epoch_ckpts:
                best_candidate = epoch_ckpts[-1]
            if os.path.exists(latest_ckpt):
                try:
                    latest_ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
                    latest_epoch = latest_ck.get("epoch", -1)
                    if best_candidate is None:
                        best_candidate = latest_ckpt
                    else:
                        best_epoch_ck = torch.load(best_candidate, map_location="cpu", weights_only=False)
                        best_epoch_num = best_epoch_ck.get("epoch", -1)
                        if latest_epoch > best_epoch_num:
                            best_candidate = latest_ckpt
                except Exception as e:
                    print(f"Warning: failed to inspect latest.pth ({e}), ignoring")

            if best_candidate:
                resume_path = best_candidate
                print(f"\nAuto-resume: found {resume_path}")

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            saved_epoch = checkpoint["epoch"]
            if "latest" in os.path.basename(resume_path):
                start_epoch = saved_epoch
                print(f"Resumed mid-epoch, restarting epoch {start_epoch + 1}")
            else:
                start_epoch = saved_epoch + 1
                print(f"Resumed at epoch {start_epoch + 1}")
            best_val_loss = checkpoint["metrics"].get("val_loss", float("inf"))
        else:
            print("\nNo checkpoint found, starting from scratch.")

    total_epochs = config["train"]["epochs"]
    early_stop_cfg = config["train"].get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 10)
    min_delta = early_stop_cfg.get("min_delta", 0.001)
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {total_epochs} epochs (early stop patience: {patience})")
    print(f"{'='*60}\n")

    def intra_epoch_save(epoch, metrics, tag=""):
        save_checkpoint(
            model, optimizer, scheduler, epoch, metrics,
            os.path.join(config["paths"]["checkpoint_dir"], f"latest.pth"),
        )

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, focal_loss, target_assigner,
            epoch, config, device, logger,
            save_fn=intra_epoch_save,
        )

        val_metrics = validate(
            model, val_loader, focal_loss, target_assigner,
            config, device,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1}/{total_epochs} ({epoch_time:.1f}s) — "
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f}, "
            f"bbox: {train_metrics['bbox_loss']:.4f}, "
            f"cls: {train_metrics['cls_loss']:.4f}) — "
            f"Val Loss: {val_metrics['val_loss']:.4f}\n"
        )

        all_metrics = {**train_metrics, **val_metrics}

        save_checkpoint(
            model, optimizer, scheduler, epoch, all_metrics,
            os.path.join(
                config["paths"]["checkpoint_dir"],
                f"checkpoint_epoch_{epoch+1}.pth",
            ),
        )

        if val_metrics["val_loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                os.path.join(config["paths"]["checkpoint_dir"], "best_model.pth"),
            )
            print(f"  ★ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break

        # W&B logging
        if args.wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })
            except Exception:
                pass

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, total_epochs - 1, all_metrics,
        os.path.join(config["paths"]["checkpoint_dir"], "final_model.pth"),
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    print(f"{'='*60}")

    if logger:
        logger.close()


if __name__ == "__main__":
    main()

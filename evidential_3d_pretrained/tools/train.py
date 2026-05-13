"""
Two-phase fine-tuning for evidential detector with pretrained PointPillars.

Phase 1 (stage 1):  Backbone FROZEN — train evidential head from scratch
                    on frozen pretrained features. Higher LR on head.
                    Duration: ~5 epochs.

Phase 2 (stage 2):  Full fine-tune — backbone and head both trainable.
                    Low LR on backbone (protect pretrained features),
                    moderate LR on head.
                    Duration: ~20 epochs.

Stage is determined from epoch count:
    epoch < stage1.epochs          → stage 1
    epoch >= stage1.epochs         → stage 2

Usage:
    python tools/train.py --config configs/pretrained_kitti.yaml
    python tools/train.py --config configs/pretrained_kitti.yaml --resume path/to/ckpt.pth
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

from models.uncertainty_detector import EvidentialPretrainedDetector
from models.pretrained_loader import download_pretrained, load_pretrained_backbone
from losses.evidential_losses import CombinedUncertaintyLoss
from tools.kitti_dataset import KITTIDataset, collate_batch
from tools.target_assigner import CenterPointTargetAssigner


def parse_args():
    parser = argparse.ArgumentParser(description="Train Evidential Pretrained Detector")
    parser.add_argument("--config", type=str,
                        default="configs/pretrained_kitti.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--skip_pretrained", action="store_true",
                        help="Skip loading pretrained backbone (e.g. for resume)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="evidential-3d-pretrained")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(config: dict, split: str) -> DataLoader:
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=is_train,
    )


def build_optimizer_stage1(model: EvidentialPretrainedDetector, config: dict):
    """Stage 1: only head is trainable — single LR."""
    opt_cfg = config["train"]["stage1"]["optimizer"]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        head_params,
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    return optimizer


def build_optimizer_stage2(model: EvidentialPretrainedDetector, config: dict):
    """Stage 2: full fine-tune — two param groups with different LRs."""
    opt_cfg = config["train"]["stage2"]["optimizer"]
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": opt_cfg["backbone_lr"],
             "name": "backbone"},
            {"params": head_params, "lr": opt_cfg["head_lr"],
             "name": "head"},
        ],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    return optimizer


def build_scheduler_stage1(optimizer, config: dict, steps_per_epoch: int):
    sched_cfg = config["train"]["stage1"]["scheduler"]
    total_steps = config["train"]["stage1"]["epochs"] * steps_per_epoch
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=sched_cfg["max_lr"],
        total_steps=total_steps,
        pct_start=sched_cfg["pct_start"],
        div_factor=sched_cfg["div_factor"],
        final_div_factor=sched_cfg["final_div_factor"],
    )


def build_scheduler_stage2(optimizer, config: dict, steps_per_epoch: int):
    sched_cfg = config["train"]["stage2"]["scheduler"]
    total_steps = config["train"]["stage2"]["epochs"] * steps_per_epoch
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[sched_cfg["max_lr_backbone"], sched_cfg["max_lr_head"]],
        total_steps=total_steps,
        pct_start=sched_cfg["pct_start"],
        div_factor=sched_cfg["div_factor"],
        final_div_factor=sched_cfg["final_div_factor"],
    )


def generate_heatmap_target(
    gt_boxes: torch.Tensor, gt_classes: torch.Tensor, gt_mask: torch.Tensor,
    num_classes: int, feature_map_size: tuple, voxel_size: list,
    point_cloud_range: list, feature_map_stride: int,
    gaussian_sigma: float = 1.0,
) -> torch.Tensor:
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
            cls = int(gt_classes[b, m].item()) - 1
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
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = (-((1 - target) ** self.beta) * (pred ** self.alpha)
                    * torch.log(1 - pred) * neg_mask)
        num_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


def train_one_epoch(
    model, dataloader, optimizer, scheduler,
    loss_fn, focal_loss, target_assigner,
    epoch, config, device, logger=None, save_fn=None,
) -> dict:
    model.train()
    total_loss_sum = 0
    heatmap_loss_sum = 0
    bbox_loss_sum = 0
    cls_loss_sum = 0
    num_batches = 0

    loss_cfg = config["train"]["loss"]
    intra_save_interval = config["train"]["checkpoint"].get(
        "intra_epoch_save_interval", 200)

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

        feat_stride = (model.module.feature_map_stride
                       if isinstance(model, nn.DataParallel)
                       else model.feature_map_stride)

        heatmap_target = generate_heatmap_target(
            gt_boxes=batch_dict["gt_boxes"],
            gt_classes=batch_dict["gt_classes"],
            gt_mask=batch_dict["gt_mask"],
            num_classes=config["model"]["num_classes"],
            feature_map_size=(H, W),
            voxel_size=config["voxel"]["voxel_size"],
            point_cloud_range=config["voxel"]["point_cloud_range"],
            feature_map_stride=feat_stride,
        )

        target_dict = target_assigner.assign(
            batch_dict["gt_boxes"], batch_dict["gt_classes"],
            batch_dict["gt_mask"], feature_map_size=(H, W),
        )

        hm_loss = focal_loss(heatmap, heatmap_target)
        evidential_losses = loss_fn(pred_dict, target_dict, current_epoch=epoch)
        total_loss = (loss_cfg["heatmap_weight"] * hm_loss
                      + evidential_losses["total_loss"])

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=10.0,
        )
        optimizer.step()
        scheduler.step()

        total_loss_sum += total_loss.item()
        heatmap_loss_sum += hm_loss.item()
        bbox_loss_sum += evidential_losses["bbox_evidential_loss"].item()
        cls_loss_sum += evidential_losses["cls_dirichlet_loss"].item()
        num_batches += 1

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = f"{lrs[0]:.5f}" if len(lrs) == 1 else \
                 f"bb={lrs[0]:.5f} hd={lrs[1]:.5f}"
        pbar.set_postfix({
            "loss": f"{total_loss.item():.3f}",
            "hm": f"{hm_loss.item():.3f}",
            "bbox": f"{evidential_losses['bbox_evidential_loss'].item():.3f}",
            "cls": f"{evidential_losses['cls_dirichlet_loss'].item():.3f}",
            "lr": lr_str,
        })

        if logger:
            step = epoch * len(dataloader) + batch_idx
            logger.add_scalar("train/total_loss", total_loss.item(), step)
            logger.add_scalar("train/heatmap_loss", hm_loss.item(), step)
            logger.add_scalar("train/bbox_loss",
                              evidential_losses["bbox_evidential_loss"].item(), step)
            for i, lr in enumerate(lrs):
                logger.add_scalar(f"train/lr_group{i}", lr, step)

        if save_fn and (batch_idx + 1) % intra_save_interval == 0:
            step_metrics = {
                "total_loss": total_loss_sum / num_batches,
                "heatmap_loss": heatmap_loss_sum / num_batches,
            }
            save_fn(epoch, step_metrics, tag=f"epoch{epoch+1}_step{batch_idx+1}")

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "heatmap_loss": heatmap_loss_sum / max(num_batches, 1),
        "bbox_loss": bbox_loss_sum / max(num_batches, 1),
        "cls_loss": cls_loss_sum / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, focal_loss, target_assigner, config, device):
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

        feat_stride = (model.module.feature_map_stride
                       if isinstance(model, nn.DataParallel)
                       else model.feature_map_stride)
        heatmap_target = generate_heatmap_target(
            batch_dict["gt_boxes"], batch_dict["gt_classes"],
            batch_dict["gt_mask"],
            num_classes=config["model"]["num_classes"],
            feature_map_size=(H, W),
            voxel_size=config["voxel"]["voxel_size"],
            point_cloud_range=config["voxel"]["point_cloud_range"],
            feature_map_stride=feat_stride,
        )
        hm_loss = focal_loss(heatmap, heatmap_target)
        total_loss_sum += hm_loss.item()
        num_batches += 1
        pbar.set_postfix({"val_loss": f"{total_loss_sum / num_batches:.4f}"})

    return {"val_loss": total_loss_sum / max(num_batches, 1)}


def save_checkpoint(model, optimizer, scheduler, epoch, stage, metrics, save_path):
    state = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": (model.module.state_dict()
                             if isinstance(model, nn.DataParallel)
                             else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"  Checkpoint saved: {save_path}")


def compute_stage(epoch: int, config: dict) -> int:
    """Return 1 if epoch is in stage 1 range, else 2."""
    s1_epochs = config["train"]["stage1"]["epochs"]
    return 1 if epoch < s1_epochs else 2


def setup_stage(model, config, stage: int, steps_per_epoch: int,
                verbose: bool = True):
    """Configure model/optimizer/scheduler for the given stage."""
    if stage == 1:
        model.freeze_backbone(verbose=verbose)
        optimizer = build_optimizer_stage1(model, config)
        scheduler = build_scheduler_stage1(optimizer, config, steps_per_epoch)
    else:
        model.unfreeze_backbone(verbose=verbose)
        optimizer = build_optimizer_stage2(model, config)
        scheduler = build_scheduler_stage2(optimizer, config, steps_per_epoch)
    return optimizer, scheduler


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size

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
    steps_per_epoch = len(train_loader)

    print("\n--- Building model ---")
    model_cfg = config["model"].copy()
    model_cfg["voxel_size"] = config["voxel"]["voxel_size"]
    model_cfg["point_cloud_range"] = config["voxel"]["point_cloud_range"]
    model = EvidentialPretrainedDetector(model_cfg).to(device)

    pretrained_cfg = config.get("pretrained", {})
    ckpt_path = pretrained_cfg.get("checkpoint_path",
                                   "./pretrained/pointpillar_7728.pth")
    if not args.skip_pretrained and not args.resume:
        if not os.path.exists(ckpt_path):
            print(f"\n--- Downloading pretrained checkpoint ---")
            download_pretrained(ckpt_path)
        print(f"\n--- Loading pretrained backbone ---")
        load_pretrained_backbone(model.backbone, ckpt_path)

    num_total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_total:,}")

    start_epoch = 0
    current_stage = compute_stage(start_epoch, config)
    optimizer, scheduler = setup_stage(model, config, current_stage, steps_per_epoch)
    print(f"\n--- Initial stage: {current_stage} ---")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

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
        # Must match model.feature_map_stride — OpenPCDet backbone
        # produces a feature map at 1/2 the pillar grid resolution.
        feature_map_stride=model.feature_map_stride,
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

    best_val_loss = float("inf")
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
        best_candidate = epoch_ckpts[-1] if epoch_ckpts else None
        if os.path.exists(latest_ckpt):
            try:
                latest_ck = torch.load(latest_ckpt, map_location="cpu",
                                       weights_only=False)
                latest_epoch = latest_ck.get("epoch", -1)
                if best_candidate is None:
                    best_candidate = latest_ckpt
                else:
                    best_ep = torch.load(best_candidate, map_location="cpu",
                                         weights_only=False).get("epoch", -1)
                    if latest_epoch > best_ep:
                        best_candidate = latest_ckpt
            except Exception as e:
                print(f"Warning: could not inspect latest.pth ({e})")
        if best_candidate:
            resume_path = best_candidate
            print(f"\nAuto-resume: found {resume_path}")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_epoch = checkpoint["epoch"]
        saved_stage = checkpoint.get("stage", compute_stage(saved_epoch, config))

        # Determine start_epoch
        if "latest" in os.path.basename(resume_path):
            start_epoch = saved_epoch
            print(f"Resumed mid-epoch, restarting epoch {start_epoch + 1}")
        else:
            start_epoch = saved_epoch + 1
            print(f"Resumed at epoch {start_epoch + 1}")

        # Determine the correct stage for start_epoch
        needed_stage = compute_stage(start_epoch, config)
        if needed_stage != saved_stage:
            print(f"Stage transition {saved_stage} → {needed_stage} detected at resume; "
                  f"rebuilding optimizer/scheduler.")
            optimizer, scheduler = setup_stage(
                model, config, needed_stage, steps_per_epoch)
            current_stage = needed_stage
        else:
            # Same stage — restore optimizer/scheduler state
            optimizer, scheduler = setup_stage(
                model, config, needed_stage, steps_per_epoch, verbose=False)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: could not restore optim/sched state ({e}); "
                      f"using fresh stage setup.")
            current_stage = needed_stage

        best_val_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
    else:
        print("\nNo checkpoint found, starting from scratch (stage 1).")

    total_epochs = (config["train"]["stage1"]["epochs"]
                    + config["train"]["stage2"]["epochs"])
    early_stop_cfg = config["train"].get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 10)
    min_delta = early_stop_cfg.get("min_delta", 0.001)
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Training plan:")
    print(f"  Stage 1 (frozen backbone): {config['train']['stage1']['epochs']} epochs")
    print(f"  Stage 2 (full fine-tune):  {config['train']['stage2']['epochs']} epochs")
    print(f"  Total:                      {total_epochs} epochs")
    print(f"  Early stop patience:        {patience} epochs")
    print(f"{'='*60}\n")

    def intra_epoch_save(epoch, metrics, tag=""):
        save_checkpoint(
            model, optimizer, scheduler, epoch, current_stage, metrics,
            os.path.join(config["paths"]["checkpoint_dir"], "latest.pth"),
        )

    all_metrics = {}
    for epoch in range(start_epoch, total_epochs):
        needed_stage = compute_stage(epoch, config)
        if needed_stage != current_stage:
            print(f"\n{'='*60}")
            print(f"STAGE TRANSITION: {current_stage} → {needed_stage} "
                  f"at epoch {epoch + 1}")
            print(f"{'='*60}")
            optimizer, scheduler = setup_stage(
                model, config, needed_stage, steps_per_epoch)
            current_stage = needed_stage

        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, focal_loss, target_assigner,
            epoch, config, device, logger,
            save_fn=intra_epoch_save,
        )
        val_metrics = validate(
            model, val_loader, focal_loss, target_assigner, config, device,
        )
        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1}/{total_epochs} [stage {current_stage}] "
            f"({epoch_time:.1f}s) — "
            f"Train Loss: {train_metrics['total_loss']:.4f} "
            f"(hm: {train_metrics['heatmap_loss']:.4f}, "
            f"bbox: {train_metrics['bbox_loss']:.4f}, "
            f"cls: {train_metrics['cls_loss']:.4f}) — "
            f"Val Loss: {val_metrics['val_loss']:.4f}\n"
        )

        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(
            model, optimizer, scheduler, epoch, current_stage, all_metrics,
            os.path.join(config["paths"]["checkpoint_dir"],
                         f"checkpoint_epoch_{epoch+1}.pth"),
        )

        if val_metrics["val_loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, current_stage, all_metrics,
                os.path.join(config["paths"]["checkpoint_dir"], "best_model.pth"),
            )
            print(f"  ★ New best model! Val Loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered.")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break

        if args.wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "stage": current_stage,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })
            except Exception:
                pass

    save_checkpoint(
        model, optimizer, scheduler, total_epochs - 1, current_stage, all_metrics,
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

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.uncertainty_detector import EvidentialPretrainedDetector
from tools.kitti_dataset import KITTIDataset, collate_batch
from visualization.vis_utils import visualize_bev_uncertainty


def parse_args():
    p = argparse.ArgumentParser(description="Canlı 3B detector demosu (KITTI val üzerinde).")
    p.add_argument("--config",       type=str,   default="configs/pretrained_kitti.yaml")
    p.add_argument("--checkpoint",   type=str,   default="Model/best_model.pth")
    p.add_argument("--num_frames",   type=int,   default=20)
    p.add_argument("--start",        type=int,   default=0)
    p.add_argument("--delay",        type=float, default=1.2)
    p.add_argument("--mode",         type=str,   default="live", choices=["live", "save", "video"])
    p.add_argument("--save_dir",     type=str,   default="demo_frames")
    p.add_argument("--output_path",  type=str,   default="demo.mp4")
    p.add_argument("--fps",          type=int,   default=2)
    p.add_argument("--gpu",          type=int,   default=0)
    return p.parse_args()


def resolve(p):
    p = Path(p)
    return str(p) if p.is_absolute() else str(PROJECT_ROOT / p)


def load_model(config, checkpoint_path, device):
    model_cfg = config["model"].copy()
    model_cfg["voxel_size"] = config["voxel"]["voxel_size"]
    model_cfg["point_cloud_range"] = config["voxel"]["point_cloud_range"]

    model = EvidentialPretrainedDetector(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[uyarı] {len(missing)} eksik anahtar var.")
    if unexpected:
        print(f"[uyarı] {len(unexpected)} beklenmeyen anahtar var.")
    model.to(device).eval()
    print(f"Checkpoint yüklendi: {checkpoint_path}")
    print(f"  epoch: {ckpt.get('epoch', '?')}, stage: {ckpt.get('stage', '?')}")
    return model


def build_val_dataset(config):
    raw = config["data"]["data_path"]
    data_path = raw if Path(raw).is_absolute() else str(PROJECT_ROOT / raw)
    return KITTIDataset(
        data_path=data_path,
        split="val",
        classes=config["data"]["classes"],
        point_cloud_range=config["voxel"]["point_cloud_range"],
        voxel_size=config["voxel"]["voxel_size"],
        max_points_per_voxel=config["voxel"]["max_points_per_voxel"],
        max_num_voxels=config["voxel"]["max_num_voxels"]["eval"],
    )


@torch.no_grad()
def run_one_sample(model, sample, device):
    batch = collate_batch([sample])
    for k in ["voxels", "voxel_coords", "voxel_num_points", "gt_boxes", "gt_classes", "gt_mask"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    out = model.predict(batch)
    det = out["detections"][0]
    pred_boxes   = det.get("final_boxes",       det["boxes"]).cpu().numpy()
    pred_scores  = det.get("final_scores",      det["scores"]).cpu().numpy()
    pred_labels  = det.get("final_labels",      det["labels"]).cpu().numpy()
    uncertainty  = det.get("final_uncertainty",
                   det.get("total_bbox_uncertainty",
                           torch.zeros(pred_boxes.shape[0]))).cpu().numpy()
    return pred_boxes, pred_scores, pred_labels, uncertainty


def render_frame(points, gt_boxes, pred_boxes, pred_scores, pred_labels,
                 uncertainty, sample_id, point_cloud_range, save_path=None):
    return visualize_bev_uncertainty(
        points=points,
        boxes=pred_boxes,
        scores=pred_scores,
        labels=pred_labels,
        uncertainty=uncertainty,
        gt_boxes=gt_boxes if gt_boxes.shape[0] > 0 else None,
        title=f"KITTI val/{sample_id}  |  {pred_boxes.shape[0]} tahmin",
        save_path=save_path,
        point_cloud_range=tuple(point_cloud_range),
    )


def make_video(frame_paths, output_path, fps):
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            print("[hata] imageio yok. `pip install imageio[ffmpeg]` çalıştırın.")
            return False
    ext = os.path.splitext(output_path)[1].lower()
    try:
        if ext in [".mp4", ".avi", ".mov"]:
            with imageio.get_writer(output_path, fps=fps, codec="libx264") as w:
                for fp in frame_paths:
                    w.append_data(imageio.imread(fp))
        else:
            imageio.mimsave(output_path, [imageio.imread(fp) for fp in frame_paths], duration=1.0 / fps)
        print(f"\nVideo oluşturuldu: {output_path}")
        return True
    except Exception as e:
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        print(f"[uyarı] Video başarısız ({e}). GIF olarak kaydediliyor: {gif_path}")
        imageio.mimsave(gif_path, [imageio.imread(fp) for fp in frame_paths], duration=1.0 / fps)
        return True


def main():
    args = parse_args()

    if args.mode in ("save", "video"):
        matplotlib.use("Agg")

    args.config      = resolve(args.config)
    args.checkpoint  = resolve(args.checkpoint)
    args.save_dir    = resolve(args.save_dir)
    args.output_path = resolve(args.output_path)

    if not os.path.exists(args.config):
        print(f"[hata] Config bulunamadı: {args.config}")
        sys.exit(1)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Cihaz: {device} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        print("Cihaz: CPU")

    model   = load_model(config, args.checkpoint, device)
    dataset = build_val_dataset(config)

    end_idx = min(args.start + args.num_frames, len(dataset))
    indices = range(args.start, end_idx)
    print(f"\nDemo: val[{args.start}:{end_idx}] — {len(indices)} örnek\n")

    if args.mode in ("save", "video"):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == "live":
        plt.ion()

    saved_paths = []
    pc_range = config["voxel"]["point_cloud_range"]

    for k, idx in enumerate(indices):
        sample    = dataset[idx]
        sample_id = sample["sample_id"]

        t0 = time.time()
        pred_boxes, pred_scores, pred_labels, uncertainty = run_one_sample(model, sample, device)
        infer_ms = (time.time() - t0) * 1000

        gt_boxes = sample["gt_boxes"]
        if gt_boxes.shape[0] > 0 and gt_boxes.shape[1] > 7:
            gt_boxes = gt_boxes[:, :7]

        save_path = None
        if args.mode in ("save", "video"):
            save_path = os.path.join(args.save_dir, f"frame_{k:04d}_{sample_id}.png")

        fig = render_frame(
            points=sample["points"], gt_boxes=gt_boxes,
            pred_boxes=pred_boxes, pred_scores=pred_scores,
            pred_labels=pred_labels, uncertainty=uncertainty,
            sample_id=sample_id, point_cloud_range=pc_range,
            save_path=save_path,
        )

        msg = (f"[{k+1:3d}/{len(indices)}] sample={sample_id} "
               f"tahmin={pred_boxes.shape[0]:3d} GT={gt_boxes.shape[0]:3d} inf={infer_ms:5.1f}ms")
        if uncertainty.size > 0:
            msg += f" unc=[{uncertainty.min():.3f}, {uncertainty.max():.3f}]"
        print(msg)

        if save_path:
            saved_paths.append(save_path)

        if args.mode == "live":
            plt.pause(args.delay)
            plt.close(fig)
        else:
            plt.close(fig)

    if args.mode == "live":
        plt.ioff()
        print("\nDemo tamamlandı.")

    if args.mode == "video" and saved_paths:
        make_video(saved_paths, args.output_path, args.fps)


if __name__ == "__main__":
    main()

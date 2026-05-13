"""
Helpers for downloading and loading OpenPCDet's pretrained PointPillars
checkpoint into our backbone.

The official checkpoint is `pointpillar_7728.pth` from OpenPCDet (KITTI 3D
PointPillars, 77.28 mAP on Car moderate). We only load the backbone portion
— VFE, map_to_bev_module, and backbone_2d — and discard the anchor-based
detection head (`dense_head.*`). The evidential head is trained from scratch.

Download sources (try in order):
    1. Pre-supplied local path (if you've already downloaded it)
    2. gdown from Google Drive (OpenPCDet official)
    3. HuggingFace mirror (fallback)
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


GDRIVE_FILE_ID = "1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm"
HF_MIRROR_URL = (
    "https://huggingface.co/OpenPCDet/pointpillar_kitti/resolve/main/"
    "pointpillar_7728.pth"
)


BACKBONE_PREFIXES = ("vfe.", "map_to_bev_module.", "backbone_2d.")


def download_pretrained(dest_path: str, verbose: bool = True) -> str:
    """Download pointpillar_7728.pth if not already at dest_path.

    Returns:
        Absolute path to the downloaded checkpoint.
    """
    dest_path = os.path.abspath(dest_path)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        if verbose:
            print(f"[pretrained] Already present: {dest_path} "
                  f"({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        import gdown
        if verbose:
            print(f"[pretrained] Downloading from Google Drive (id={GDRIVE_FILE_ID})...")
        gdown.download(id=GDRIVE_FILE_ID, output=dest_path, quiet=not verbose)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            return dest_path
    except Exception as e:
        if verbose:
            print(f"[pretrained] gdown failed: {e}")

    try:
        if verbose:
            print(f"[pretrained] Falling back to HuggingFace mirror: {HF_MIRROR_URL}")
        torch.hub.download_url_to_file(HF_MIRROR_URL, dest_path, progress=verbose)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            return dest_path
    except Exception as e:
        if verbose:
            print(f"[pretrained] HF mirror failed: {e}")

    raise RuntimeError(
        f"Could not download pretrained checkpoint. Please download "
        f"pointpillar_7728.pth manually from OpenPCDet and place it at {dest_path}. "
        f"Google Drive ID: {GDRIVE_FILE_ID}"
    )


def load_pretrained_backbone(
    backbone: nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Load pretrained weights into our PointPillars backbone.

    Args:
        backbone: An instance of PretrainedPointPillarsBackbone (or any module
                  with `vfe.`, `map_to_bev_module.`, `backbone_2d.` submodules).
        checkpoint_path: Path to pointpillar_7728.pth.
        verbose: Print loading stats.

    Returns:
        Dict with keys:
            - "loaded": list of parameter names successfully loaded
            - "skipped_pretrained": pretrained keys that had no target
            - "missing_target": target params that pretrained didn't cover
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}. "
            "Call download_pretrained() first."
        )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model_state" in raw:
        pretrained_sd = raw["model_state"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        pretrained_sd = raw["state_dict"]
    else:
        pretrained_sd = raw  # assume it's already the state_dict

    backbone_sd = {
        k: v for k, v in pretrained_sd.items()
        if k.startswith(BACKBONE_PREFIXES)
    }

    skipped_pretrained = [
        k for k in pretrained_sd.keys() if not k.startswith(BACKBONE_PREFIXES)
    ]

    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)

    loaded = [k for k in backbone_sd.keys() if k not in unexpected]

    if verbose:
        print(f"[pretrained] Loaded {len(loaded)} backbone params from "
              f"{checkpoint_path}")
        print(f"[pretrained] Skipped {len(skipped_pretrained)} non-backbone "
              f"pretrained keys (head weights, etc.)")
        if unexpected:
            print(f"[pretrained] {len(unexpected)} unexpected pretrained keys "
                  f"(not in our backbone): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if missing:
            print(f"[pretrained] {len(missing)} backbone params NOT covered by "
                  f"pretrained: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return {
        "loaded": loaded,
        "skipped_pretrained": skipped_pretrained,
        "missing_target": list(missing),
        "unexpected": list(unexpected),
    }


def freeze_backbone(model: nn.Module, verbose: bool = True) -> int:
    """Freeze backbone parameters (VFE + scatter + backbone_2d). Evidential
    head remains trainable.

    Returns:
        Number of parameters frozen.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(f"backbone.{p}") or name.startswith(p)
               for p in BACKBONE_PREFIXES):
            param.requires_grad = False
            frozen += param.numel()
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] Frozen {frozen:,} backbone params. "
              f"Trainable: {trainable:,} / Total: {total:,}")
    return frozen


def unfreeze_all(model: nn.Module, verbose: bool = True) -> None:
    """Unfreeze all parameters (for full fine-tune phase)."""
    for param in model.parameters():
        param.requires_grad = True
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[unfreeze] All {trainable:,} params now trainable.")

"""
3D visualization utilities for uncertainty-aware detection results.

Provides point cloud + bounding box + uncertainty visualization using
Open3D and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple


def uncertainty_to_color(
    uncertainty: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlGn_r",
) -> np.ndarray:
    """Convert uncertainty values to RGB colors.

    Low uncertainty → Green, High uncertainty → Red.

    Args:
        uncertainty: (N,) uncertainty values.
        vmin: Min value for normalization (default: auto).
        vmax: Max value for normalization (default: auto).
        cmap: Matplotlib colormap name.

    Returns:
        (N, 3) RGB colors in [0, 1].
    """
    if vmin is None:
        vmin = uncertainty.min()
    if vmax is None:
        vmax = uncertainty.max()

    normalized = np.clip((uncertainty - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colormap = cm.get_cmap(cmap)
    colors = colormap(normalized)[:, :3]  # Drop alpha
    return colors


def visualize_3d_with_uncertainty(
    points: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    gt_boxes: Optional[np.ndarray] = None,
    title: str = "3D Detection with Uncertainty",
    save_path: Optional[str] = None,
    point_size: float = 1.0,
) -> None:
    """Visualize 3D point cloud with detection boxes colored by uncertainty.

    Uses Open3D for interactive 3D visualization.

    Args:
        points: (N, 3+) Point cloud (x, y, z, ...).
        boxes: (M, 7) Predicted bounding boxes.
        scores: (M,) Detection scores.
        labels: (M,) Class labels.
        uncertainty: (M,) Per-box uncertainty values.
        gt_boxes: (K, 7) Optional ground truth boxes (shown in blue).
        title: Window title.
        save_path: If provided, save screenshot instead of interactive display.
        point_size: Point rendering size.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Falling back to BEV matplotlib plot.")
        visualize_bev_uncertainty(
            points, boxes, scores, labels, uncertainty,
            gt_boxes=gt_boxes, title=title, save_path=save_path,
        )
        return

    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    point_colors = np.ones((len(points), 3)) * 0.6  # Gray
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(pcd)

    if boxes.shape[0] > 0:
        box_colors = uncertainty_to_color(uncertainty)
        for i in range(boxes.shape[0]):
            bbox_lines = create_3d_bbox_lineset(
                boxes[i], color=box_colors[i].tolist()
            )
            geometries.append(bbox_lines)

    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        for i in range(gt_boxes.shape[0]):
            bbox_lines = create_3d_bbox_lineset(
                gt_boxes[i], color=[0.0, 0.0, 1.0]
            )
            geometries.append(bbox_lines)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    for g in geometries:
        vis.add_geometry(g)

    ctrl = vis.get_view_control()
    ctrl.set_front([0, 0, 1])
    ctrl.set_up([0, 1, 0])
    ctrl.set_lookat([35, 0, 0])
    ctrl.set_zoom(0.3)

    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"Saved 3D visualization to {save_path}")
    else:
        vis.run()
        vis.destroy_window()


def create_3d_bbox_lineset(
    box: np.ndarray, color: List[float] = [1, 0, 0]
) -> "o3d.geometry.LineSet":
    """Create Open3D LineSet for a 3D bounding box.

    Args:
        box: (7,) [x, y, z, dx, dy, dz, heading]
        color: RGB color [0-1].

    Returns:
        Open3D LineSet geometry.
    """
    import open3d as o3d

    x, y, z, dx, dy, dz, heading = box
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    corners = np.array([
        [-dx/2, -dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [ dx/2,  dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ])

    R = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1],
    ])

    corners = (R @ corners.T).T + np.array([x, y, z])

    lines = [
        [0,1],[1,2],[2,3],[3,0],  # Bottom
        [4,5],[5,6],[6,7],[7,4],  # Top
        [0,4],[1,5],[2,6],[3,7],  # Verticals
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def visualize_bev_uncertainty(
    points: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    gt_boxes: Optional[np.ndarray] = None,
    title: str = "BEV Detection with Uncertainty",
    save_path: Optional[str] = None,
    point_cloud_range: Tuple = (0, -39.68, -3, 69.12, 39.68, 1),
) -> plt.Figure:
    """Bird's Eye View visualization with uncertainty-colored boxes.

    Matplotlib-based — works without Open3D, also good for papers/reports.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4])
    )
    pts = points[mask]
    ax.scatter(pts[:, 0], pts[:, 1], c="gray", s=0.1, alpha=0.3)

    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        for box in gt_boxes:
            rect = _box_to_bev_rect(box)
            ax.plot(*rect.T, "b--", linewidth=1.5, label="GT" if box is gt_boxes[0] else "")

    if boxes.shape[0] > 0:
        colors = uncertainty_to_color(uncertainty)
        for i, box in enumerate(boxes):
            rect = _box_to_bev_rect(box)
            ax.plot(*rect.T, color=colors[i], linewidth=2)
            ax.text(
                box[0], box[1] + 1,
                f"{scores[i]:.2f}\nu:{uncertainty[i]:.3f}",
                fontsize=6, ha="center", color=colors[i],
            )

        sm = plt.cm.ScalarMappable(
            cmap="RdYlGn_r",
            norm=plt.Normalize(vmin=uncertainty.min(), vmax=uncertainty.max()),
        )
        plt.colorbar(sm, ax=ax, label="Uncertainty", shrink=0.8)

    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    if gt_boxes is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"BEV visualization saved to {save_path}")

    return fig


def _box_to_bev_rect(box: np.ndarray) -> np.ndarray:
    """Convert 3D box to BEV rectangle corners for plotting.

    Returns (5, 2) array — 4 corners + closing point.
    """
    x, y, z, dx, dy, dz, heading = box
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    corners = np.array([
        [-dx/2, -dy/2],
        [ dx/2, -dy/2],
        [ dx/2,  dy/2],
        [-dx/2,  dy/2],
        [-dx/2, -dy/2],  # Close rectangle
    ])

    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (R @ corners.T).T + np.array([x, y])

    return corners


def plot_uncertainty_histogram(
    uncertainty_tp: np.ndarray,
    uncertainty_fp: np.ndarray,
    title: str = "Uncertainty Distribution: TP vs FP",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot uncertainty distributions for true/false positive detections.

    Well-calibrated uncertainty should show clear separation:
    FP detections should have higher uncertainty than TP detections.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(uncertainty_tp.min(), uncertainty_fp.min()),
        max(uncertainty_tp.max(), uncertainty_fp.max()),
        50,
    )

    ax.hist(uncertainty_tp, bins=bins, alpha=0.6, label="True Positives",
            color="#3A923A", density=True, edgecolor="black", linewidth=0.5)
    ax.hist(uncertainty_fp, bins=bins, alpha=0.6, label="False Positives",
            color="#E1812C", density=True, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Uncertainty", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    tp_mean = uncertainty_tp.mean()
    fp_mean = uncertainty_fp.mean()
    ax.axvline(tp_mean, color="#3A923A", linestyle="--", linewidth=1.5)
    ax.axvline(fp_mean, color="#E1812C", linestyle="--", linewidth=1.5)
    ax.text(0.02, 0.95, f"TP mean: {tp_mean:.4f}\nFP mean: {fp_mean:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

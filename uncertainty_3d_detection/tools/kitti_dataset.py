"""
KITTI 3D Object Detection dataset loader.

Handles loading of LiDAR point clouds, calibration files, and labels
from the KITTI dataset. Supports data augmentation for training.

KITTI directory structure expected:
    kitti/
    ├── training/
    │   ├── velodyne/      # .bin LiDAR point clouds
    │   ├── calib/         # .txt calibration files
    │   ├── label_2/       # .txt 3D bounding box labels
    │   └── image_2/       # .png images (optional)
    └── ImageSets/
        ├── train.txt
        └── val.txt
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import pickle


CLASS_MAP = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
LABEL_MAP = {v: k for k, v in CLASS_MAP.items()}

DIFFICULTY_LEVELS = {
    "Easy": {"min_height": 40, "max_occlusion": 0, "max_truncation": 0.15},
    "Moderate": {"min_height": 25, "max_occlusion": 1, "max_truncation": 0.30},
    "Hard": {"min_height": 25, "max_occlusion": 2, "max_truncation": 0.50},
}


class KITTIDataset(Dataset):
    """KITTI 3D Object Detection dataset.

    Args:
        data_path: Root path to KITTI dataset.
        split: 'train', 'val', or 'test'.
        classes: List of class names to detect.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        max_points_per_voxel: Maximum points allowed in one voxel.
        max_num_voxels: Maximum total voxels.
        augmentation: Dictionary of augmentation configs (None for eval).
        gt_database_path: Path to GT database for GT sampling augmentation.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        classes: List[str] = None,
        point_cloud_range: List[float] = None,
        voxel_size: List[float] = None,
        max_points_per_voxel: int = 5,
        max_num_voxels: int = 16000,
        augmentation: Optional[Dict] = None,
        gt_database_path: Optional[str] = None,
    ):
        self.data_path = data_path
        self.split = split
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]
        self.point_cloud_range = np.array(
            point_cloud_range or [0, -39.68, -3, 69.12, 39.68, 1]
        )
        self.voxel_size = np.array(voxel_size or [0.05, 0.05, 0.1])
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_voxels = max_num_voxels
        self.augmentation = augmentation
        self.training = split == "train"

        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3])
            / self.voxel_size
        ).astype(np.int64)

        split_file = os.path.join(data_path, "ImageSets", f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f.readlines()]
        else:
            vel_dir = os.path.join(data_path, "training", "velodyne")
            if os.path.exists(vel_dir):
                self.sample_ids = sorted([
                    f.replace(".bin", "") for f in os.listdir(vel_dir)
                    if f.endswith(".bin")
                ])
            else:
                self.sample_ids = []

        self.gt_database = None
        if gt_database_path and os.path.exists(gt_database_path):
            with open(gt_database_path, "rb") as f:
                self.gt_database = pickle.load(f)

        print(f"KITTI {split} set: {len(self.sample_ids)} samples loaded")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict:
        sample_id = self.sample_ids[index]

        points = self.load_point_cloud(sample_id)

        labels = self.load_labels(sample_id)

        calib = self.load_calibration(sample_id)

        gt_boxes, gt_classes, gt_names = self.labels_to_lidar(labels, calib)

        points = self.filter_points(points)

        if self.training and self.augmentation:
            points, gt_boxes, gt_classes, gt_names = self.apply_augmentation(
                points, gt_boxes, gt_classes, gt_names
            )

        voxels, voxel_coords, voxel_num_points = self.voxelize(points)

        return {
            "sample_id": sample_id,
            "points": points,
            "voxels": voxels,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "gt_names": gt_names,
            "calib": calib,
        }

    def load_point_cloud(self, sample_id: str) -> np.ndarray:
        """Load LiDAR point cloud from .bin file.

        Returns: (N, 4) array with [x, y, z, intensity].
        """
        bin_path = os.path.join(
            self.data_path, "training", "velodyne", f"{sample_id}.bin"
        )
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def load_labels(self, sample_id: str) -> List[Dict]:
        """Load 3D bounding box labels from KITTI label file.

        Each label has: type, truncated, occluded, alpha, bbox_2d (4),
        dimensions (h, w, l), location (x, y, z in camera coords), rotation_y
        """
        label_path = os.path.join(
            self.data_path, "training", "label_2", f"{sample_id}.txt"
        )

        if not os.path.exists(label_path):
            return []

        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                obj_type = parts[0]
                if obj_type not in self.classes:
                    continue

                labels.append({
                    "type": obj_type,
                    "truncated": float(parts[1]),
                    "occluded": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bbox_2d": [float(x) for x in parts[4:8]],
                    "dimensions": [float(x) for x in parts[8:11]],  # h, w, l
                    "location": [float(x) for x in parts[11:14]],    # x, y, z (cam)
                    "rotation_y": float(parts[14]),
                })

        return labels

    def load_calibration(self, sample_id: str) -> Dict:
        """Load KITTI calibration matrices.

        Returns dict with P0, P1, P2, P3, R0_rect, Tr_velo_to_cam matrices.
        """
        calib_path = os.path.join(
            self.data_path, "training", "calib", f"{sample_id}.txt"
        )

        calib = {}
        if not os.path.exists(calib_path):
            # Return identity-like defaults
            calib["Tr_velo_to_cam"] = np.eye(4)[:3, :]
            calib["R0_rect"] = np.eye(4)
            return calib

        with open(calib_path, "r") as f:
            for line in f.readlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                calib[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )

        # Parse into matrices
        if "Tr_velo_to_cam" in calib:
            V2C = calib["Tr_velo_to_cam"].reshape(3, 4)
            calib["Tr_velo_to_cam"] = V2C

        if "R0_rect" in calib:
            R0 = np.eye(4)
            R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
            calib["R0_rect"] = R0

        return calib

    def labels_to_lidar(
        self, labels: List[Dict], calib: Dict
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert KITTI camera-coord labels to LiDAR coordinates.

        Args:
            labels: List of label dicts from load_labels().
            calib: Calibration dict from load_calibration().

        Returns:
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading] in LiDAR
            gt_classes: (M,) class indices
            gt_names: list of class name strings
        """
        if not labels:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                [],
            )

        boxes = []
        classes = []
        names = []

        V2C = calib["Tr_velo_to_cam"]
        R0 = calib.get("R0_rect", np.eye(4))

        V2C_4x4 = np.eye(4)
        V2C_4x4[:3, :] = V2C
        C2V = np.linalg.inv(R0 @ V2C_4x4)

        for label in labels:
            h, w, l = label["dimensions"]
            x, y, z = label["location"]
            ry = label["rotation_y"]

            cam_center = np.array([x, y - h / 2, z, 1.0])  # bottom center -> center
            lidar_center = (C2V @ cam_center)[:3]

            heading = -(ry + np.pi / 2)

            boxes.append([
                lidar_center[0], lidar_center[1], lidar_center[2],
                l, w, h,
                heading,
            ])
            classes.append(CLASS_MAP[label["type"]])
            names.append(label["type"])

        gt_boxes = np.array(boxes, dtype=np.float32)
        gt_classes = np.array(classes, dtype=np.int64)

        return gt_boxes, gt_classes, names

    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """Remove points outside the detection range."""
        mask = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] <= self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] <= self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] <= self.point_cloud_range[5])
        )
        return points[mask]

    def voxelize(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert point cloud to voxel representation.

        Uses simple hash-based voxelization (no external dependencies needed).

        Args:
            points: (N, 4) filtered point cloud.

        Returns:
            voxels: (V, max_points, 4) point features per voxel
            coords: (V, 3) voxel coordinates [z, y, x]
            num_points: (V,) actual point count per voxel
        """
        voxel_indices = np.floor(
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

        valid = (
            (voxel_indices >= 0).all(axis=1)
            & (voxel_indices[:, 0] < self.grid_size[0])
            & (voxel_indices[:, 1] < self.grid_size[1])
            & (voxel_indices[:, 2] < self.grid_size[2])
        )
        points = points[valid]
        voxel_indices = voxel_indices[valid]

        Gx, Gy, Gz = self.grid_size
        voxel_hash = (
            voxel_indices[:, 0] * (Gy * Gz)
            + voxel_indices[:, 1] * Gz
            + voxel_indices[:, 2]
        )

        unique_hashes, inverse, counts = np.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )

        num_voxels = min(len(unique_hashes), self.max_num_voxels)
        voxels = np.zeros(
            (num_voxels, self.max_points_per_voxel, points.shape[1]),
            dtype=np.float32,
        )
        coords = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points = np.zeros(num_voxels, dtype=np.int32)

        for i in range(num_voxels):
            mask = inverse == i
            pts = points[mask]

            n = min(len(pts), self.max_points_per_voxel)
            if len(pts) > self.max_points_per_voxel:
                choice = np.random.choice(len(pts), self.max_points_per_voxel,
                                          replace=False)
                pts = pts[choice]
                n = self.max_points_per_voxel

            voxels[i, :n] = pts[:n]
            num_points[i] = n

            idx = np.where(mask)[0][0]
            coords[i] = voxel_indices[idx][::-1]  # x,y,z -> z,y,x

        return voxels, coords, num_points

    def apply_augmentation(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Apply training data augmentation.

        Supports: random flip, rotation, scaling, and GT sampling.
        """
        aug = self.augmentation

        if (
            aug.get("gt_sampling", {}).get("enabled", False)
            and self.gt_database is not None
        ):
            points, gt_boxes, gt_classes, gt_names = self.gt_sampling(
                points, gt_boxes, gt_classes, gt_names
            )

        if aug.get("random_flip", {}).get("enabled", False):
            if np.random.random() < aug["random_flip"].get("probability", 0.5):
                points[:, 1] = -points[:, 1]
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6]  # Flip heading

        if aug.get("random_rotation", {}).get("enabled", False):
            rot_range = aug["random_rotation"]["range"]
            angle = np.random.uniform(rot_range[0], rot_range[1])
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points[:, :2] = points[:, :2] @ rot_matrix.T

            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_matrix.T
                gt_boxes[:, 6] += angle

        if aug.get("random_scaling", {}).get("enabled", False):
            scale_range = aug["random_scaling"]["range"]
            scale = np.random.uniform(scale_range[0], scale_range[1])
            points[:, :3] *= scale
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :6] *= scale

        return points, gt_boxes, gt_classes, gt_names

    def gt_sampling(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Ground truth database sampling augmentation.

        Randomly pastes GT objects from other scenes into the current scene.
        This is a key augmentation for 3D detection — significantly boosts
        performance especially for rare classes (Pedestrian, Cyclist).
        """
        if self.gt_database is None:
            return points, gt_boxes, gt_classes, gt_names

        sample_groups = self.augmentation.get(
            "gt_sampling", {}
        ).get("sample_groups", {})

        new_boxes = []
        new_classes = []
        new_names = []
        new_points_list = []

        for class_name, max_samples in sample_groups.items():
            if class_name not in self.gt_database:
                continue

            current_count = sum(1 for n in gt_names if n == class_name)
            num_to_sample = max(0, max_samples - current_count)

            if num_to_sample <= 0:
                continue

            db_entries = self.gt_database[class_name]
            if len(db_entries) == 0:
                continue

            sampled_indices = np.random.choice(
                len(db_entries), min(num_to_sample, len(db_entries)), replace=False
            )

            for idx in sampled_indices:
                entry = db_entries[idx]
                sampled_box = entry["box3d_lidar"]
                sampled_points = entry.get("points")

                if sampled_points is None:
                    continue

                if gt_boxes.shape[0] > 0:
                    dists = np.linalg.norm(
                        gt_boxes[:, :2] - sampled_box[:2], axis=1
                    )
                    if dists.min() < 1.0:
                        continue

                new_boxes.append(sampled_box)
                new_classes.append(CLASS_MAP[class_name])
                new_names.append(class_name)
                new_points_list.append(sampled_points)

        if new_boxes:
            new_boxes = np.array(new_boxes, dtype=np.float32)
            new_classes = np.array(new_classes, dtype=np.int64)

            gt_boxes = np.concatenate([gt_boxes, new_boxes], axis=0)
            gt_classes = np.concatenate([gt_classes, new_classes], axis=0)
            gt_names = gt_names + new_names
            points = np.concatenate([points] + new_points_list, axis=0)

        return points, gt_boxes, gt_classes, gt_names


def collate_batch(batch_list: List[Dict]) -> Dict:
    """Custom collate function for DataLoader.

    Merges individual samples into a batch, handling variable-size
    voxel counts by adding batch index to coordinates.
    """
    batch_size = len(batch_list)

    max_gt = max(sample["gt_boxes"].shape[0] for sample in batch_list)
    if max_gt == 0:
        max_gt = 1

    all_voxels = []
    all_coords = []
    all_num_points = []
    gt_boxes_batch = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
    gt_classes_batch = np.zeros((batch_size, max_gt), dtype=np.int64)
    gt_mask_batch = np.zeros((batch_size, max_gt), dtype=np.float32)
    sample_ids = []

    for i, sample in enumerate(batch_list):
        voxels = sample["voxels"]
        coords = sample["voxel_coords"]

        all_voxels.append(voxels)

        batch_coords = np.pad(
            coords, ((0, 0), (1, 0)),
            mode="constant", constant_values=i,
        )
        all_coords.append(batch_coords)
        all_num_points.append(sample["voxel_num_points"])

        n_gt = sample["gt_boxes"].shape[0]
        if n_gt > 0:
            gt_boxes_batch[i, :n_gt] = sample["gt_boxes"][:max_gt]
            gt_classes_batch[i, :n_gt] = sample["gt_classes"][:max_gt]
            gt_mask_batch[i, :n_gt] = 1.0

        sample_ids.append(sample["sample_id"])

    return {
        "voxels": torch.from_numpy(np.concatenate(all_voxels, axis=0)),
        "voxel_coords": torch.from_numpy(
            np.concatenate(all_coords, axis=0)
        ).int(),
        "voxel_num_points": torch.from_numpy(
            np.concatenate(all_num_points, axis=0)
        ),
        "batch_size": batch_size,
        "gt_boxes": torch.from_numpy(gt_boxes_batch),
        "gt_classes": torch.from_numpy(gt_classes_batch),
        "gt_mask": torch.from_numpy(gt_mask_batch),
        "sample_ids": sample_ids,
    }

"""
nuScenes Dataset Class for PyTorch
===================================
Custom dataset loader for nuScenes 3D object detection.
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


# nuScenes detection classes
NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

NUM_CLASSES = 10

# Point cloud range for nuScenes (in meters)
# x: front/back, y: left/right, z: up/down
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Voxel size for PointPillars
VOXEL_SIZE = [0.2, 0.2, 8.0]  # x, y, z in meters


def load_points_from_file(lidar_path: str) -> np.ndarray:
    """
    Load point cloud from .bin file.
    
    nuScenes LiDAR format: x, y, z, intensity, ring_index
    
    Args:
        lidar_path: Path to .bin file
    
    Returns:
        Point cloud array (N, 4) with x, y, z, intensity
    """
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape(-1, 5)  # x, y, z, intensity, ring
    points = points[:, :4]  # Keep only x, y, z, intensity
    return points


def filter_points_in_range(points: np.ndarray, point_cloud_range: List[float]) -> np.ndarray:
    """Filter points within the specified range."""
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4]) &
        (points[:, 2] >= point_cloud_range[2]) &
        (points[:, 2] <= point_cloud_range[5])
    )
    return points[mask]


def points_to_voxels(
    points: np.ndarray,
    voxel_size: List[float],
    point_cloud_range: List[float],
    max_points_per_voxel: int = 32,
    max_voxels: int = 16000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert point cloud to voxels for PointPillars.
    
    Args:
        points: Point cloud (N, 4)
        voxel_size: Voxel dimensions [x, y, z]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel: Maximum points in each voxel
        max_voxels: Maximum number of voxels
    
    Returns:
        voxels: (M, max_points_per_voxel, 4)
        coordinates: (M, 3) voxel coordinates
        num_points: (M,) number of points in each voxel
    """
    # Calculate grid size
    grid_size = np.array([
        (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0],
        (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1],
        (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2],
    ]).astype(np.int32)
    
    # Calculate voxel indices for each point
    voxel_indices = np.floor(
        (points[:, :3] - np.array(point_cloud_range[:3])) / np.array(voxel_size)
    ).astype(np.int32)
    
    # Filter out-of-range indices
    mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_size[0]) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_size[1]) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_size[2])
    )
    points = points[mask]
    voxel_indices = voxel_indices[mask]
    
    # Create unique voxel keys
    voxel_keys = (
        voxel_indices[:, 0] * grid_size[1] * grid_size[2] +
        voxel_indices[:, 1] * grid_size[2] +
        voxel_indices[:, 2]
    )
    
    # Get unique voxels
    unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    # Limit number of voxels
    if len(unique_keys) > max_voxels:
        selected = np.random.choice(len(unique_keys), max_voxels, replace=False)
        unique_keys = unique_keys[selected]
        
        # Create mask for points in selected voxels
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        point_mask = np.array([voxel_keys[i] in key_to_idx for i in range(len(voxel_keys))])
        points = points[point_mask]
        voxel_indices = voxel_indices[point_mask]
        voxel_keys = voxel_keys[point_mask]
        _, inverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    num_voxels = len(unique_keys)
    
    # Initialize outputs
    voxels = np.zeros((num_voxels, max_points_per_voxel, 4), dtype=np.float32)
    coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points = np.zeros(num_voxels, dtype=np.int32)
    
    # Fill voxels
    for i, point in enumerate(points):
        voxel_idx = inverse_indices[i]
        
        if num_points[voxel_idx] < max_points_per_voxel:
            voxels[voxel_idx, num_points[voxel_idx]] = point
            num_points[voxel_idx] += 1
            coordinates[voxel_idx] = voxel_indices[i]
    
    return voxels, coordinates, num_points


def encode_boxes(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        boxes: Ground truth boxes (N, 7) [x, y, z, w, l, h, yaw]
        anchors: Anchor boxes (N, 7)
    
    Returns:
        Encoded boxes (N, 7)
    """
    # Diagonal of anchor base
    anchor_diag = np.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
    # Encode
    encoded = np.zeros_like(boxes)
    encoded[:, 0] = (boxes[:, 0] - anchors[:, 0]) / anchor_diag
    encoded[:, 1] = (boxes[:, 1] - anchors[:, 1]) / anchor_diag
    encoded[:, 2] = (boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    encoded[:, 3] = np.log(boxes[:, 3] / anchors[:, 3])
    encoded[:, 4] = np.log(boxes[:, 4] / anchors[:, 4])
    encoded[:, 5] = np.log(boxes[:, 5] / anchors[:, 5])
    encoded[:, 6] = np.sin(boxes[:, 6] - anchors[:, 6])
    
    return encoded


class NuScenesDataset(Dataset):
    """
    nuScenes 3D Object Detection Dataset.
    
    Each sample contains:
        - Point cloud from LiDAR
        - 3D bounding box annotations
    """
    
    def __init__(
        self,
        data_dir: str,
        info_path: str,
        point_cloud_range: List[float] = POINT_CLOUD_RANGE,
        voxel_size: List[float] = VOXEL_SIZE,
        max_points_per_voxel: int = 32,
        max_voxels: int = 16000,
        augment: bool = True,
        split: str = 'train'
    ):
        """
        Initialize nuScenes dataset.
        
        Args:
            data_dir: Path to nuScenes root
            info_path: Path to preprocessed info pickle file
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: Voxel dimensions [x, y, z]
            max_points_per_voxel: Maximum points per voxel
            max_voxels: Maximum number of voxels
            augment: Apply data augmentation
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.augment = augment and split == 'train'
        self.split = split
        
        # Load info
        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)
        
        # Calculate grid size
        self.grid_size = np.array([
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ])
        
        print(f"Loaded {len(self.infos)} samples for {split}")
        print(f"Grid size: {self.grid_size}")
    
    def __len__(self) -> int:
        return len(self.infos)
    
    def _augment_points(self, points: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to points and boxes."""
        # Random rotation around z-axis
        if np.random.random() < 0.5:
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            points[:, :3] = points[:, :3] @ rot_matrix.T
            boxes[:, :3] = boxes[:, :3] @ rot_matrix.T
            boxes[:, 6] += angle
        
        # Random flip along x-axis
        if np.random.random() < 0.5:
            points[:, 1] = -points[:, 1]
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
            boxes[:, :6] *= scale
        
        return points, boxes
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - voxels: (M, max_points_per_voxel, 4)
                - coordinates: (M, 3)
                - num_points: (M,)
                - gt_boxes: (N, 7) [x, y, z, w, l, h, yaw]
                - gt_labels: (N,)
        """
        info = self.infos[idx]
        
        # Load point cloud
        lidar_path = info['lidar_path']
        if not os.path.isabs(lidar_path):
            lidar_path = str(self.data_dir / lidar_path)
        
        points = load_points_from_file(lidar_path)
        
        # Filter points in range
        points = filter_points_in_range(points, self.point_cloud_range)
        
        # Get ground truth boxes
        gt_boxes = []
        gt_labels = []
        
        for ann in info['annotations']:
            # Box: [x, y, z, w, l, h, yaw]
            center = ann['center']
            wlh = ann['wlh']
            yaw = ann['yaw']
            
            box = [center[0], center[1], center[2], wlh[0], wlh[1], wlh[2], yaw]
            gt_boxes.append(box)
            gt_labels.append(ann['class_id'])
        
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros((0,), dtype=np.int64)
        
        # Data augmentation
        if self.augment and len(gt_boxes) > 0:
            points, gt_boxes = self._augment_points(points, gt_boxes)
        
        # Convert to voxels
        voxels, coordinates, num_points = points_to_voxels(
            points, self.voxel_size, self.point_cloud_range,
            self.max_points_per_voxel, self.max_voxels
        )
        
        return {
            'voxels': torch.from_numpy(voxels),
            'coordinates': torch.from_numpy(coordinates),
            'num_points': torch.from_numpy(num_points),
            'gt_boxes': torch.from_numpy(gt_boxes),
            'gt_labels': torch.from_numpy(gt_labels),
            'sample_token': info['sample_token'],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-size voxels.
    """
    voxels_list = []
    coordinates_list = []
    num_points_list = []
    gt_boxes_list = []
    gt_labels_list = []
    batch_indices = []
    
    for i, sample in enumerate(batch):
        voxels_list.append(sample['voxels'])
        
        # Add batch index to coordinates
        coords = sample['coordinates']
        batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
        coordinates_list.append(torch.cat([batch_idx, coords], dim=1))
        
        num_points_list.append(sample['num_points'])
        gt_boxes_list.append(sample['gt_boxes'])
        gt_labels_list.append(sample['gt_labels'])
    
    return {
        'voxels': torch.cat(voxels_list, dim=0),
        'coordinates': torch.cat(coordinates_list, dim=0),
        'num_points': torch.cat(num_points_list, dim=0),
        'gt_boxes': gt_boxes_list,  # Keep as list (variable size per sample)
        'gt_labels': gt_labels_list,
        'batch_size': len(batch),
    }


def get_dataloaders(
    data_dir: str,
    processed_dir: str = None,
    batch_size: int = 2,
    num_workers: int = 4,
    point_cloud_range: List[float] = POINT_CLOUD_RANGE,
    voxel_size: List[float] = VOXEL_SIZE,
    max_voxels: int = 16000
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    if processed_dir is None:
        processed_dir = Path(data_dir) / 'processed'
    else:
        processed_dir = Path(processed_dir)
    
    train_dataset = NuScenesDataset(
        data_dir=data_dir,
        info_path=str(processed_dir / 'nuscenes_infos_train.pkl'),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=max_voxels,
        augment=True,
        split='train'
    )
    
    val_dataset = NuScenesDataset(
        data_dir=data_dir,
        info_path=str(processed_dir / 'nuscenes_infos_val.pkl'),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=max_voxels,
        augment=False,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Test dataset
    processed_dir = Path(args.data_dir) / 'processed'
    
    if (processed_dir / 'nuscenes_infos_train.pkl').exists():
        dataset = NuScenesDataset(
            data_dir=args.data_dir,
            info_path=str(processed_dir / 'nuscenes_infos_train.pkl'),
            augment=False
        )
        
        sample = dataset[0]
        print(f"Voxels shape: {sample['voxels'].shape}")
        print(f"Coordinates shape: {sample['coordinates'].shape}")
        print(f"GT boxes shape: {sample['gt_boxes'].shape}")
        print(f"GT labels: {sample['gt_labels']}")
    else:
        print(f"Please run prepare_nuscenes.py first")

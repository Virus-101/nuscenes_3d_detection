"""
3D Object Detection Inference Script
=====================================
Run inference on point cloud data.

Usage:
    python inference.py --weights best.pth --source sample.bin       # Single file
    python inference.py --weights best.pth --source lidar_folder/    # Folder
    python inference.py --weights best.pth --data_dir /path/to/nuscenes --sample_token xxx
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys
from tqdm import tqdm
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent))

from models.pointpillars import get_model
from utils.dataset import (
    load_points_from_file, filter_points_in_range, points_to_voxels,
    NUSCENES_CLASSES, NUM_CLASSES, POINT_CLOUD_RANGE, VOXEL_SIZE
)


# Visualization imports
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. 3D visualization disabled.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Detector3D:
    """3D Object Detection inference class."""
    
    def __init__(
        self,
        weights_path: str,
        model_name: str = None,
        device: str = 'cuda',
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5
    ):
        """
        Initialize detector.
        
        Args:
            weights_path: Path to model weights
            model_name: Model architecture
            device: Device to run on
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device)
        
        if model_name is None:
            config = checkpoint.get('config', {})
            model_name = config.get('model', 'pointpillars_lite')
        
        # Create model
        self.model = get_model(model_name, num_classes=NUM_CLASSES)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model: {model_name}")
        print(f"Best loss from training: {checkpoint.get('best_loss', 'N/A')}")
        
        # Get voxel parameters from model
        self.voxel_size = self.model.voxel_size
        self.point_cloud_range = self.model.point_cloud_range
        self.grid_size = self.model.grid_size
    
    def preprocess(self, points: np.ndarray) -> dict:
        """Preprocess point cloud for inference."""
        # Filter points in range
        points = filter_points_in_range(points, self.point_cloud_range)
        
        # Convert to voxels
        voxels, coordinates, num_points = points_to_voxels(
            points, self.voxel_size, self.point_cloud_range,
            max_points_per_voxel=32, max_voxels=16000
        )
        
        # Add batch index to coordinates
        batch_idx = np.zeros((coordinates.shape[0], 1), dtype=np.int32)
        coordinates = np.concatenate([batch_idx, coordinates], axis=1)
        
        return {
            'voxels': torch.from_numpy(voxels).to(self.device),
            'coordinates': torch.from_numpy(coordinates).to(self.device),
            'num_points': torch.from_numpy(num_points).to(self.device),
            'batch_size': 1,
        }
    
    @torch.no_grad()
    def predict(self, points: np.ndarray) -> dict:
        """
        Run detection on point cloud.
        
        Args:
            points: Point cloud (N, 4) with x, y, z, intensity
        
        Returns:
            Dictionary with boxes, scores, labels
        """
        # Preprocess
        batch = self.preprocess(points)
        
        # Forward pass
        preds = self.model(batch)
        
        # Post-process predictions
        # Note: This is simplified - full implementation needs anchor decoding and NMS
        cls_preds = preds['cls_preds'].sigmoid()  # (1, num_anchors*num_classes, H, W)
        box_preds = preds['box_preds']  # (1, num_anchors*7, H, W)
        
        # Get top predictions
        cls_scores, cls_indices = cls_preds.max(dim=1)  # (1, H, W)
        cls_scores = cls_scores[0].cpu().numpy()
        cls_indices = cls_indices[0].cpu().numpy()
        
        # Find high-confidence detections
        mask = cls_scores > self.conf_threshold
        
        # Placeholder results (real implementation needs proper decoding)
        boxes = []
        scores = []
        labels = []
        
        return {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 7)),
            'scores': np.array(scores) if scores else np.zeros((0,)),
            'labels': np.array(labels) if labels else np.zeros((0,), dtype=np.int32),
            'cls_heatmap': cls_scores,
        }
    
    def detect_file(self, file_path: str) -> dict:
        """Run detection on a .bin file."""
        points = load_points_from_file(file_path)
        return self.predict(points)


def visualize_bev(points: np.ndarray, boxes: np.ndarray = None, 
                  scores: np.ndarray = None, labels: np.ndarray = None,
                  point_cloud_range: list = POINT_CLOUD_RANGE,
                  save_path: str = None):
    """
    Visualize point cloud and detections in Bird's Eye View.
    
    Args:
        points: Point cloud (N, 4)
        boxes: Detected boxes (M, 7) [x, y, z, w, l, h, yaw]
        scores: Detection scores (M,)
        labels: Detection labels (M,)
        point_cloud_range: Point cloud range
        save_path: Path to save visualization
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for visualization")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot points in BEV
    mask = (
        (points[:, 0] >= point_cloud_range[0]) &
        (points[:, 0] <= point_cloud_range[3]) &
        (points[:, 1] >= point_cloud_range[1]) &
        (points[:, 1] <= point_cloud_range[4])
    )
    pts = points[mask]
    
    ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=pts[:, 2], cmap='viridis', alpha=0.5)
    
    # Plot boxes
    if boxes is not None and len(boxes) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        
        for i, box in enumerate(boxes):
            x, y, z, w, l, h, yaw = box
            label = labels[i] if labels is not None else 0
            score = scores[i] if scores is not None else 1.0
            
            # Create rotated rectangle
            rect = Rectangle(
                (-l/2, -w/2), l, w,
                linewidth=2, edgecolor=colors[label], facecolor='none'
            )
            
            # Apply rotation and translation
            t = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            
            # Add label
            class_name = list(NUSCENES_CLASSES.keys())[label]
            ax.text(x, y, f'{class_name}\n{score:.2f}', fontsize=8, ha='center')
    
    ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
    ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Bird\'s Eye View')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_3d(points: np.ndarray, boxes: np.ndarray = None):
    """
    Visualize point cloud and boxes in 3D using Open3D.
    
    Args:
        points: Point cloud (N, 4)
        boxes: Detected boxes (M, 7)
    """
    if not HAS_OPEN3D:
        print("Open3D not available for 3D visualization")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color by height
    colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                            (points[:, 2].max() - points[:, 2].min() + 1e-6))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Add boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x, y, z, w, l, h, yaw = box
            
            # Create box
            bbox = o3d.geometry.OrientedBoundingBox(
                center=[x, y, z],
                R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw]),
                extent=[l, w, h]
            )
            bbox.color = [1, 0, 0]  # Red
            geometries.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)


def process_sample(detector: Detector3D, file_path: str, output_dir: str = None, visualize: bool = False):
    """Process a single point cloud file."""
    print(f"Processing: {file_path}")
    
    # Load points
    points = load_points_from_file(file_path)
    print(f"  Loaded {len(points)} points")
    
    # Detect
    results = detector.predict(points)
    
    print(f"  Detected {len(results['boxes'])} objects")
    
    # Visualize
    if visualize or output_dir:
        if output_dir:
            save_path = Path(output_dir) / f"{Path(file_path).stem}_bev.png"
        else:
            save_path = None
        
        visualize_bev(
            points, results['boxes'], results['scores'], results['labels'],
            save_path=str(save_path) if save_path else None
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='3D Object Detection Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to .bin file or folder')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to nuScenes dataset (for sample token mode)')
    parser.add_argument('--sample_token', type=str, default=None,
                        help='nuScenes sample token')
    parser.add_argument('--model', type=str, default=None,
                        help='Model architecture')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='runs/inference_3d',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--viz_3d', action='store_true',
                        help='Show 3D visualization (requires Open3D)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D OBJECT DETECTION - INFERENCE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create detector
    detector = Detector3D(
        weights_path=args.weights,
        model_name=args.model,
        device=device,
        conf_threshold=args.conf
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process source
    if args.source:
        source_path = Path(args.source)
        
        if source_path.is_file():
            results = process_sample(detector, str(source_path), str(output_dir), args.visualize)
            
            if args.viz_3d:
                points = load_points_from_file(str(source_path))
                visualize_3d(points, results['boxes'])
        
        elif source_path.is_dir():
            bin_files = list(source_path.glob('*.bin'))
            print(f"Found {len(bin_files)} .bin files")
            
            for bin_file in tqdm(bin_files):
                process_sample(detector, str(bin_file), str(output_dir), False)
    
    elif args.data_dir and args.sample_token:
        # Load from nuScenes
        from nuscenes.nuscenes import NuScenes
        
        # Detect version
        if (Path(args.data_dir) / 'v1.0-mini').exists():
            version = 'v1.0-mini'
        else:
            version = 'v1.0-trainval'
        
        nusc = NuScenes(version=version, dataroot=args.data_dir, verbose=False)
        
        sample = nusc.get('sample', args.sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_path = Path(args.data_dir) / lidar_data['filename']
        
        results = process_sample(detector, str(lidar_path), str(output_dir), args.visualize)
        
        if args.viz_3d:
            points = load_points_from_file(str(lidar_path))
            visualize_3d(points, results['boxes'])
    
    else:
        print("Please specify --source or (--data_dir and --sample_token)")
        return
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

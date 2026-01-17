"""
nuScenes Dataset Preparation Script
====================================
Prepares nuScenes dataset for 3D object detection training.

Usage:
    python prepare_nuscenes.py --data_dir /path/to/nuscenes --version v1.0-mini

Expected input structure:
    nuscenes/
    ├── maps/
    ├── samples/
    │   ├── CAM_FRONT/
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   ├── CAM_BACK_RIGHT/
    │   └── LIDAR_TOP/
    ├── sweeps/
    │   └── LIDAR_TOP/
    └── v1.0-mini/ (or v1.0-trainval/)
        ├── attribute.json
        ├── calibrated_sensor.json
        ├── category.json
        ├── ego_pose.json
        ├── instance.json
        ├── log.json
        ├── map.json
        ├── sample.json
        ├── sample_annotation.json
        ├── sample_data.json
        ├── scene.json
        ├── sensor.json
        └── visibility.json
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import json

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion


# nuScenes detection classes (10 classes)
NUSCENES_CLASSES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'trailer': 3,
    'construction_vehicle': 4,
    'pedestrian': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic_cone': 8,
    'barrier': 9,
}

# Map from nuScenes categories to detection classes
CATEGORY_TO_CLASS = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
}

NUM_CLASSES = 10


def verify_dataset(data_dir: Path, version: str) -> bool:
    """Verify nuScenes dataset structure."""
    required_dirs = [
        'samples/LIDAR_TOP',
        'samples/CAM_FRONT',
        version,
    ]
    
    missing = []
    for d in required_dirs:
        if not (data_dir / d).exists():
            missing.append(d)
    
    if missing:
        print("Missing directories:")
        for d in missing:
            print(f"  - {d}")
        return False
    
    return True


def get_lidar_to_ego_transform(nusc, sample_data):
    """Get transformation from LiDAR to ego vehicle frame."""
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    rotation = Quaternion(cs_record['rotation']).rotation_matrix
    translation = np.array(cs_record['translation'])
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform


def get_ego_to_global_transform(nusc, sample_data):
    """Get transformation from ego vehicle to global frame."""
    ego_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    
    rotation = Quaternion(ego_record['rotation']).rotation_matrix
    translation = np.array(ego_record['translation'])
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform


def box_to_corners(box):
    """Convert 3D box to 8 corner points."""
    # Box dimensions
    w, l, h = box.wlh
    
    # 8 corners in box coordinate frame
    corners = np.array([
        [-l/2, -w/2, -h/2],
        [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2],
        [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],
        [l/2, -w/2, h/2],
        [l/2, w/2, h/2],
        [-l/2, w/2, h/2],
    ])
    
    # Rotate and translate
    corners = np.dot(corners, box.rotation_matrix.T) + box.center
    
    return corners


def process_sample(nusc, sample_token: str, data_dir: Path) -> dict:
    """
    Process a single sample and extract info for training.
    
    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        data_dir: Dataset root directory
    
    Returns:
        Dictionary with sample info
    """
    sample = nusc.get('sample', sample_token)
    
    # Get LiDAR data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_path = data_dir / lidar_data['filename']
    
    # Get transformations
    lidar_to_ego = get_lidar_to_ego_transform(nusc, lidar_data)
    ego_to_global = get_ego_to_global_transform(nusc, lidar_data)
    
    # Get annotations
    annotations = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # Get category
        category = ann['category_name']
        if category not in CATEGORY_TO_CLASS:
            continue
        
        class_name = CATEGORY_TO_CLASS[category]
        class_id = NUSCENES_CLASSES[class_name]
        
        # Get box in global frame
        box = nusc.get_box(ann_token)
        
        # Transform to ego frame
        box.translate(-np.array(ego_to_global[:3, 3]))
        box.rotate(Quaternion(matrix=ego_to_global[:3, :3]).inverse)
        
        # Transform to LiDAR frame
        box.translate(-np.array(lidar_to_ego[:3, 3]))
        box.rotate(Quaternion(matrix=lidar_to_ego[:3, :3]).inverse)
        
        # Extract box parameters
        center = box.center  # x, y, z
        wlh = box.wlh  # width, length, height
        yaw = box.orientation.yaw_pitch_roll[0]  # rotation around z-axis
        
        annotations.append({
            'class_name': class_name,
            'class_id': class_id,
            'center': center.tolist(),
            'wlh': wlh.tolist(),
            'yaw': yaw,
            'num_lidar_pts': ann['num_lidar_pts'],
            'visibility': ann['visibility_token'],
        })
    
    # Get camera paths for multi-modal fusion (optional)
    cam_paths = {}
    for cam_name in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        cam_paths[cam_name] = str(data_dir / cam_data['filename'])
    
    return {
        'sample_token': sample_token,
        'lidar_path': str(lidar_path),
        'lidar_to_ego': lidar_to_ego.tolist(),
        'ego_to_global': ego_to_global.tolist(),
        'annotations': annotations,
        'cam_paths': cam_paths,
        'timestamp': sample['timestamp'],
    }


def create_infos(nusc, data_dir: Path, split: str = 'train') -> list:
    """
    Create info dictionaries for all samples in a split.
    
    Args:
        nusc: NuScenes instance
        data_dir: Dataset root directory
        split: 'train' or 'val'
    
    Returns:
        List of info dictionaries
    """
    # Get scene splits
    if split == 'train':
        scenes = [s for s in nusc.scene if s['name'] not in ['scene-0061', 'scene-0553', 
                                                               'scene-0655', 'scene-0757',
                                                               'scene-0796', 'scene-1077',
                                                               'scene-1094', 'scene-1100']]
    else:
        scenes = [s for s in nusc.scene if s['name'] in ['scene-0061', 'scene-0553',
                                                          'scene-0655', 'scene-0757',
                                                          'scene-0796', 'scene-1077',
                                                          'scene-1094', 'scene-1100']]
    
    # For mini dataset, use simple split
    if len(nusc.scene) <= 10:
        if split == 'train':
            scenes = nusc.scene[:8]
        else:
            scenes = nusc.scene[8:]
    
    infos = []
    
    for scene in tqdm(scenes, desc=f'Processing {split} scenes'):
        # Get first sample token
        sample_token = scene['first_sample_token']
        
        while sample_token:
            info = process_sample(nusc, sample_token, data_dir)
            infos.append(info)
            
            # Get next sample
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
    
    return infos


def analyze_dataset(infos: list):
    """Analyze dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal samples: {len(infos)}")
    
    # Count annotations per class
    class_counts = {name: 0 for name in NUSCENES_CLASSES.keys()}
    total_annotations = 0
    
    for info in infos:
        for ann in info['annotations']:
            class_counts[ann['class_name']] += 1
            total_annotations += 1
    
    print(f"Total annotations: {total_annotations}")
    print(f"\nClass distribution:")
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        percentage = count / total_annotations * 100 if total_annotations > 0 else 0
        print(f"  {class_name:20s}: {count:6d} ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Prepare nuScenes dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval'],
                        help='Dataset version')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'processed'
    
    print("="*60)
    print("NUSCENES DATASET PREPARATION")
    print("="*60)
    
    # Print dataset info
    print(f"\nDataset directory: {data_dir}")
    print(f"Version: {args.version}")
    
    # Verify structure
    print("\nVerifying dataset structure...")
    if not verify_dataset(data_dir, args.version):
        print("\nError: Invalid dataset structure!")
        print("\nPlease download from: https://www.nuscenes.org/nuscenes#download")
        print("\nFor mini dataset (~4GB):")
        print("  - v1.0-mini.tar.gz")
        print("\nFor full dataset (~300GB):")
        print("  - v1.0-trainval_meta.tar.gz")
        print("  - v1.0-trainval01_blobs.tar.gz (and more)")
        return
    
    print("✓ Dataset structure verified")
    
    # Initialize nuScenes
    print(f"\nLoading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=str(data_dir), verbose=True)
    
    print(f"\nDataset contains:")
    print(f"  - {len(nusc.scene)} scenes")
    print(f"  - {len(nusc.sample)} samples")
    print(f"  - {len(nusc.sample_annotation)} annotations")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train split
    print("\nProcessing training data...")
    train_infos = create_infos(nusc, data_dir, 'train')
    
    train_path = output_dir / 'nuscenes_infos_train.pkl'
    with open(train_path, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f"Saved {len(train_infos)} train samples to: {train_path}")
    
    # Process val split
    print("\nProcessing validation data...")
    val_infos = create_infos(nusc, data_dir, 'val')
    
    val_path = output_dir / 'nuscenes_infos_val.pkl'
    with open(val_path, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f"Saved {len(val_infos)} val samples to: {val_path}")
    
    # Analyze dataset
    analyze_dataset(train_infos + val_infos)
    
    # Save class info
    class_info = {
        'classes': NUSCENES_CLASSES,
        'category_mapping': CATEGORY_TO_CLASS,
        'num_classes': NUM_CLASSES,
    }
    
    with open(output_dir / 'class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print("\n" + "="*60)
    print("PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update data_dir in configs/nuscenes.yaml")
    print(f"  2. Run: python train.py --data_dir {data_dir}")


if __name__ == '__main__':
    main()

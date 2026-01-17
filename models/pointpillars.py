"""
3D Object Detection Models
==========================
PointPillars and other 3D detection architectures.
Optimized for RTX 4050 (6GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Net (PFN) from PointPillars.
    Converts point cloud voxels to pillar features.
    """
    
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 64,
        voxel_size: List[float] = [0.2, 0.2, 8.0],
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        """
        Args:
            in_channels: Input point features (x, y, z, intensity, xc, yc, zc, xp, yp)
            out_channels: Output pillar features
            voxel_size: Voxel dimensions
            point_cloud_range: Point cloud range
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Point-wise MLP
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
    
    def forward(self, voxels: torch.Tensor, coordinates: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: (M, max_points, 4) - x, y, z, intensity
            coordinates: (M, 4) - batch_idx, z, y, x
            num_points: (M,) - number of points per voxel
        
        Returns:
            Pillar features (M, out_channels)
        """
        # Calculate pillar center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.view(-1, 1, 1).clamp(min=1)
        
        # Offset from pillar center
        f_cluster = voxels[:, :, :3] - points_mean
        
        # Offset from pillar position (in BEV coordinates)
        pillar_x = coordinates[:, 3].float() * self.voxel_size[0] + self.point_cloud_range[0]
        pillar_y = coordinates[:, 2].float() * self.voxel_size[1] + self.point_cloud_range[1]
        
        f_center = torch.zeros_like(voxels[:, :, :2])
        f_center[:, :, 0] = voxels[:, :, 0] - pillar_x.unsqueeze(1)
        f_center[:, :, 1] = voxels[:, :, 1] - pillar_y.unsqueeze(1)
        
        # Concatenate features
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)  # (M, max_points, 9)
        
        # Apply point-wise MLP
        M, P, C = features.shape
        features = features.view(-1, C)
        features = self.linear(features)
        features = self.norm(features)
        features = F.relu(features)
        features = features.view(M, P, -1)
        
        # Max pooling over points
        features = features.max(dim=1)[0]  # (M, out_channels)
        
        return features


class PointPillarsScatter(nn.Module):
    """
    Scatter pillar features to pseudo-image (BEV feature map).
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        grid_size: List[int] = [512, 512, 1]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.nx = grid_size[0]
        self.ny = grid_size[1]
    
    def forward(self, pillar_features: torch.Tensor, coordinates: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            pillar_features: (M, in_channels)
            coordinates: (M, 4) - batch_idx, z, y, x
            batch_size: Batch size
        
        Returns:
            Pseudo-image (batch_size, in_channels, ny, nx)
        """
        # Create batch canvas
        batch_canvas = []
        
        for batch_idx in range(batch_size):
            # Get pillars for this batch
            mask = coordinates[:, 0] == batch_idx
            pillars = pillar_features[mask]
            coords = coordinates[mask]
            
            # Create canvas for this sample
            canvas = torch.zeros(self.in_channels, self.ny * self.nx, 
                               dtype=pillar_features.dtype, device=pillar_features.device)
            
            # Scatter pillars to canvas
            indices = coords[:, 2] * self.nx + coords[:, 3]  # y * nx + x
            indices = indices.long()
            
            # Handle multiple pillars at same location (keep max)
            canvas[:, indices] = pillars.t()
            
            # Reshape to 2D
            canvas = canvas.view(self.in_channels, self.ny, self.nx)
            batch_canvas.append(canvas)
        
        return torch.stack(batch_canvas, dim=0)


class Backbone2D(nn.Module):
    """
    2D CNN backbone for processing BEV feature map.
    Uses multiple scales for multi-scale feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: List[int] = [3, 5, 5],
        layer_strides: List[int] = [2, 2, 2],
        out_channels: List[int] = [64, 128, 256]
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        c_in = in_channels
        
        for i, (num_layers, stride, c_out) in enumerate(zip(layer_nums, layer_strides, out_channels)):
            # Downsample block
            block = [
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            
            for _ in range(num_layers - 1):
                block.extend([
                    nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ])
            
            self.blocks.append(nn.Sequential(*block))
            
            # Upsample block
            deblock = nn.Sequential(
                nn.ConvTranspose2d(c_out, c_out, stride, stride=stride, bias=False),
                nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.deblocks.append(deblock)
            
            c_in = c_out
        
        self.out_channels = sum(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV feature map (B, C, H, W)
        
        Returns:
            Multi-scale features concatenated (B, sum(out_channels), H, W)
        """
        ups = []
        
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        
        # Concatenate multi-scale features
        return torch.cat(ups, dim=1)


class DetectionHead(nn.Module):
    """
    Detection head for predicting 3D bounding boxes.
    """
    
    def __init__(
        self,
        in_channels: int = 448,  # sum of backbone output channels
        num_classes: int = 10,
        num_anchors: int = 2,  # 2 orientations per class
        box_code_size: int = 7  # x, y, z, w, l, h, yaw
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.box_code_size = box_code_size
        
        # Classification head
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        
        # Regression head
        self.conv_box = nn.Conv2d(in_channels, num_anchors * box_code_size, 1)
        
        # Direction classification (for yaw)
        self.conv_dir = nn.Conv2d(in_channels, num_anchors * 2, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        # Classification bias initialization (focal loss)
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.conv_cls.bias, bias_value)
        
        # Other layers
        for m in [self.conv_box, self.conv_dir]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Backbone features (B, C, H, W)
        
        Returns:
            Dictionary with cls_preds, box_preds, dir_preds
        """
        cls_preds = self.conv_cls(x)  # (B, num_anchors * num_classes, H, W)
        box_preds = self.conv_box(x)  # (B, num_anchors * box_code_size, H, W)
        dir_preds = self.conv_dir(x)  # (B, num_anchors * 2, H, W)
        
        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds,
            'dir_preds': dir_preds,
        }


class PointPillars(nn.Module):
    """
    PointPillars 3D Object Detection Model.
    
    Paper: PointPillars: Fast Encoders for Object Detection from Point Clouds
    https://arxiv.org/abs/1812.05784
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        voxel_size: List[float] = [0.2, 0.2, 8.0],
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points_per_voxel: int = 32,
        pillar_features: int = 64,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            1
        ]
        
        # Pillar Feature Net
        self.pfn = PillarFeatureNet(
            in_channels=9,  # x, y, z, intensity, xc, yc, zc, xp, yp
            out_channels=pillar_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range
        )
        
        # Scatter to pseudo-image
        self.scatter = PointPillarsScatter(
            in_channels=pillar_features,
            grid_size=self.grid_size
        )
        
        # 2D Backbone
        self.backbone = Backbone2D(
            in_channels=pillar_features,
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            out_channels=[64, 128, 256]
        )
        
        # Detection Head
        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_anchors=2,
            box_code_size=7
        )
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary with voxels, coordinates, num_points, batch_size
        
        Returns:
            Dictionary with predictions
        """
        voxels = batch['voxels']
        coordinates = batch['coordinates']
        num_points = batch['num_points']
        batch_size = batch['batch_size']
        
        # Extract pillar features
        pillar_features = self.pfn(voxels, coordinates, num_points)
        
        # Scatter to BEV
        bev_features = self.scatter(pillar_features, coordinates, batch_size)
        
        # 2D backbone
        backbone_features = self.backbone(bev_features)
        
        # Detection head
        preds = self.head(backbone_features)
        
        return preds


class PointPillarsLite(nn.Module):
    """
    Lightweight PointPillars for limited VRAM.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        voxel_size: List[float] = [0.25, 0.25, 8.0],  # Larger voxels
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        pillar_features: int = 32,  # Reduced features
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            1
        ]
        
        self.pfn = PillarFeatureNet(
            in_channels=9,
            out_channels=pillar_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range
        )
        
        self.scatter = PointPillarsScatter(
            in_channels=pillar_features,
            grid_size=self.grid_size
        )
        
        # Lighter backbone
        self.backbone = Backbone2D(
            in_channels=pillar_features,
            layer_nums=[2, 3, 3],  # Fewer layers
            layer_strides=[2, 2, 2],
            out_channels=[32, 64, 128]  # Fewer channels
        )
        
        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_anchors=2,
            box_code_size=7
        )
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        voxels = batch['voxels']
        coordinates = batch['coordinates']
        num_points = batch['num_points']
        batch_size = batch['batch_size']
        
        pillar_features = self.pfn(voxels, coordinates, num_points)
        bev_features = self.scatter(pillar_features, coordinates, batch_size)
        backbone_features = self.backbone(bev_features)
        preds = self.head(backbone_features)
        
        return preds


def get_model(
    model_name: str = 'pointpillars',
    num_classes: int = 10,
    voxel_size: List[float] = [0.2, 0.2, 8.0],
    point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
) -> nn.Module:
    """
    Factory function to create 3D detection models.
    
    Args:
        model_name: Model name
        num_classes: Number of detection classes
        voxel_size: Voxel dimensions
        point_cloud_range: Point cloud range
    
    Returns:
        3D detection model
    
    Available models:
        - pointpillars: Standard PointPillars (~3-4GB VRAM)
        - pointpillars_lite: Lightweight version (~2GB VRAM)
    """
    models = {
        'pointpillars': lambda: PointPillars(
            num_classes=num_classes,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            pillar_features=64
        ),
        'pointpillars_lite': lambda: PointPillarsLite(
            num_classes=num_classes,
            voxel_size=[0.25, 0.25, 8.0],
            point_cloud_range=point_cloud_range,
            pillar_features=32
        ),
    }
    
    if model_name not in models:
        available = ', '.join(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return models[model_name]()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Testing PointPillars models...\n")
    
    for name in ['pointpillars', 'pointpillars_lite']:
        model = get_model(name, num_classes=10)
        params = count_parameters(model)
        
        print(f"{name}:")
        print(f"  Parameters: {params:,}")
        
        # Create dummy batch
        batch = {
            'voxels': torch.randn(1000, 32, 4),
            'coordinates': torch.randint(0, 512, (1000, 4)),
            'num_points': torch.randint(1, 32, (1000,)),
            'batch_size': 2,
        }
        batch['coordinates'][:, 0] = torch.randint(0, 2, (1000,))  # batch indices
        
        with torch.no_grad():
            preds = model(batch)
        
        print(f"  Cls preds shape: {preds['cls_preds'].shape}")
        print(f"  Box preds shape: {preds['box_preds'].shape}")
        print()

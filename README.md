# nuScenes 3D Object Detection

Complete 3D object detection training pipeline using PointPillars on nuScenes dataset, optimized for RTX 4050 (6GB VRAM).

## 📁 Project Structure

```
nuscenes_3d_detection/
├── configs/
│   └── nuscenes.yaml              # Training configuration
├── models/
│   ├── __init__.py
│   └── pointpillars.py            # PointPillars architecture
├── scripts/
│   └── prepare_nuscenes.py        # Dataset preparation
├── utils/
│   ├── __init__.py
│   └── dataset.py                 # Dataset classes
├── train.py                       # Training script
├── inference.py                   # Inference script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create conda environment
conda create -n nuscenes3d python=3.10 -y
conda activate nuscenes3d

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Step 2: Download nuScenes Dataset

**Official Website:** https://www.nuscenes.org/

1. **Register** at https://www.nuscenes.org/sign-up
2. **Download** from https://www.nuscenes.org/nuscenes#download

**Recommended for learning (Mini dataset ~4GB):**
- `v1.0-mini.tar.gz`

**Full dataset (~300GB):**
- `v1.0-trainval_meta.tar.gz` (metadata)
- Blob files for samples (images + LiDAR)

```bash
mkdir -p ~/datasets/nuscenes
cd ~/datasets/nuscenes

# Extract mini dataset
tar -xzf v1.0-mini.tar.gz

# Expected structure:
# nuscenes/
# ├── maps/
# ├── samples/
# │   ├── CAM_FRONT/
# │   ├── CAM_FRONT_LEFT/
# │   ├── CAM_FRONT_RIGHT/
# │   ├── CAM_BACK/
# │   ├── CAM_BACK_LEFT/
# │   ├── CAM_BACK_RIGHT/
# │   └── LIDAR_TOP/
# ├── sweeps/
# └── v1.0-mini/
#     └── *.json files
```

### Step 3: Prepare Dataset

```bash
python scripts/prepare_nuscenes.py --data_dir ~/datasets/nuscenes --version v1.0-mini
```

This creates preprocessed pickle files with annotations.

### Step 4: Train

```bash
# Default training (PointPillars Lite)
python train.py --data_dir ~/datasets/nuscenes

# Standard PointPillars (needs more VRAM)
python train.py --data_dir ~/datasets/nuscenes --model pointpillars --batch 1
```

### Step 5: Monitor Training

```bash
tensorboard --logdir runs/detection_3d
```

### Step 6: Run Inference

```bash
# On a .bin file
python inference.py --weights runs/detection_3d/.../best.pth --source sample.bin

# Visualize in BEV
python inference.py --weights ... --source sample.bin --visualize

# 3D visualization (requires Open3D)
python inference.py --weights ... --source sample.bin --viz_3d
```

---

## 📊 Dataset Information

### nuScenes Mini

| Property | Value |
|----------|-------|
| Size | ~4GB |
| Scenes | 10 |
| Samples | ~400 |
| Sensors | 6 cameras + 1 LiDAR + 5 radars |

### nuScenes Full (trainval)

| Property | Value |
|----------|-------|
| Size | ~300GB |
| Scenes | 1000 |
| Samples | 40,000 |
| Train/Val | 850/150 scenes |

### Detection Classes (10)

| ID | Name | Typical Size (w×l×h) |
|----|------|----------------------|
| 0 | car | 1.95 × 4.60 × 1.73 m |
| 1 | truck | 2.51 × 6.93 × 2.84 m |
| 2 | bus | 2.94 × 10.5 × 3.47 m |
| 3 | trailer | 2.90 × 12.3 × 3.87 m |
| 4 | construction_vehicle | 2.73 × 6.37 × 3.19 m |
| 5 | pedestrian | 0.67 × 0.73 × 1.77 m |
| 6 | motorcycle | 0.77 × 2.11 × 1.47 m |
| 7 | bicycle | 0.60 × 1.70 × 1.28 m |
| 8 | traffic_cone | 0.41 × 0.41 × 1.07 m |
| 9 | barrier | 2.53 × 0.50 × 0.98 m |

---

## 🤖 Available Models

| Model | Parameters | VRAM | Description |
|-------|------------|------|-------------|
| `pointpillars` | ~5M | ~3-4GB | Standard PointPillars |
| `pointpillars_lite` | ~2M | **~2GB** | Lightweight version (recommended) |

**For RTX 4050:** Use `pointpillars_lite` with batch size 2

---

## ⚙️ Training Options

```bash
python train.py --help

# Key options:
--data_dir       # Path to nuScenes dataset
--model          # Model architecture (pointpillars, pointpillars_lite)
--epochs         # Number of epochs (default: 80)
--batch          # Batch size (default: 2)
--lr             # Learning rate (default: 0.001)
--max_voxels     # Maximum voxels (reduce if OOM)
--amp            # Mixed precision (default: enabled)
--resume         # Resume from checkpoint
```

### Example Commands

```bash
# Quick test on mini dataset
python train.py --data_dir ~/datasets/nuscenes --model pointpillars_lite --epochs 20

# Standard training for RTX 4050
python train.py --data_dir ~/datasets/nuscenes --model pointpillars_lite --epochs 80 --batch 2

# Reduce memory usage
python train.py --data_dir ~/datasets/nuscenes --max_voxels 10000 --batch 1

# Resume training
python train.py --data_dir ~/datasets/nuscenes --resume runs/detection_3d/.../checkpoint.pth
```

---

## 📈 Expected Results

After training on nuScenes mini with PointPillars Lite:

| Metric | Expected Value |
|--------|---------------|
| mAP | 0.20 - 0.35 |
| NDS | 0.30 - 0.45 |

**Note:** Mini dataset is very small. For better results, use the full trainval set.

Training time on RTX 4050:
- pointpillars_lite (mini): ~2-4 hours
- pointpillars_lite (full): ~24-48 hours

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train.py --batch 1

# Use lite model
python train.py --model pointpillars_lite

# Reduce max voxels
python train.py --max_voxels 10000
```

### Slow Training

```bash
# Ensure CUDA is being used
python -c "import torch; print(torch.cuda.is_available())"

# Increase workers
python train.py --workers 8
```

### Dataset Not Found

```bash
# Verify dataset structure
ls ~/datasets/nuscenes/v1.0-mini/

# Run preparation script
python scripts/prepare_nuscenes.py --data_dir ~/datasets/nuscenes --version v1.0-mini
```

---

## 🔬 Technical Details

### PointPillars Architecture

```
Point Cloud (N, 4)
       ↓
   Voxelization
       ↓
Pillar Feature Net (PFN)
       ↓
  Scatter to BEV
       ↓
  2D CNN Backbone
       ↓
 Detection Head
       ↓
3D Boxes + Classes
```

### Point Cloud Processing

1. **Voxelization**: Points are grouped into vertical pillars
2. **Pillar Features**: PointNet-style feature extraction
3. **Pseudo-Image**: Scatter features to 2D BEV grid
4. **2D Backbone**: Multi-scale feature extraction
5. **Detection Head**: Predict boxes, classes, directions

### Loss Function

- **Classification**: Focal Loss (handles class imbalance)
- **Box Regression**: Smooth L1 Loss
- **Direction**: Binary cross-entropy

---

## 🚗 Integration with Autonomous Driving Stack

This 3D detection module integrates with:

1. **Object Detection (BDD100K)** - 2D image detection
2. **Lane Detection (TuSimple)** - Lane markings
3. **Semantic Segmentation (Cityscapes)** - Scene understanding
4. **Sensor Fusion** - Combine camera + LiDAR detections
5. **Planning Module** - Use 3D boxes for path planning

### Sensor Fusion Example

```python
from utils.dataset import load_points_from_file
from models.pointpillars import get_model

# Load 3D detector
detector_3d = get_model('pointpillars_lite')
detector_3d.load_state_dict(torch.load('best.pth')['model_state_dict'])

# Process LiDAR
points = load_points_from_file('lidar.bin')
detections_3d = detector_3d.predict(points)

# Combine with 2D detections from camera
# ... fusion logic ...
```

---

## 📤 Export for Deployment

```python
import torch
from models.pointpillars import get_model

# Load model
model = get_model('pointpillars_lite', num_classes=10)
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Note: ONNX export for PointPillars requires custom handling
# due to variable-size voxel inputs
```

---

## 🔗 Resources

- [nuScenes Dataset](https://www.nuscenes.org/)
- [nuScenes DevKit](https://github.com/nutonomy/nuscenes-devkit)
- [PointPillars Paper](https://arxiv.org/abs/1812.05784)
- [Open3D](http://www.open3d.org/) - 3D visualization

---

## 📝 Complete Autonomous Driving Stack

You now have all four perception modules:

| Module | Dataset | Purpose |
|--------|---------|---------|
| ✅ Object Detection | BDD100K | 2D object detection |
| ✅ Lane Detection | TuSimple | Lane marking detection |
| ✅ Semantic Segmentation | Cityscapes | Scene understanding |
| ✅ 3D Detection | nuScenes | 3D object localization |

**Next Steps:**
1. Train all models on your RTX 4050
2. Integrate models into unified perception pipeline
3. Test in CARLA simulator
4. Add prediction and planning modules
5. Deploy to edge device or vehicle

Good luck with your autonomous driving project! 🚗

# TensorRT-LightNet: High-Efficiency Real-Time CNN for Edge AI

**trt-lightnet** is a TensorRT-based CNN inference framework optimized for edge AI devices (Jetson Xavier, Orin, etc.). It combines [LightNet](https://github.com/daniel89710/lightNet) — a lightweight neural network architecture for edge devices — with NVIDIA TensorRT to deliver real-time object detection, semantic segmentation, and depth estimation in a single multitask pipeline.

## Table of Contents

- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Local Build](#local-build)
  - [Docker Build](#docker-build)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
- [Command-Line Reference](#command-line-reference)
- [Python API (pylightnet)](#python-api-pylightnet)
- [Advanced Features](#advanced-features)
- [Tips](#tips)
- [Project Structure](#project-structure)
- [References](#references)

---

## Key Features

| Feature | Description |
|---|---|
| **Multitask Inference** | Object detection + semantic segmentation + depth estimation in one pass |
| **Multi-Precision** | FP32 / FP16 / INT8 quantization with per-layer precision control |
| **2:4 Structured Sparsity** | 50% weight reduction with maintained accuracy on Ampere+ GPUs |
| **NVDLA Support** | Offload inference to the Deep Learning Accelerator on Xavier/Orin |
| **Hierarchical Detection** | Two-stage subnet for fine-grained classification (e.g. traffic light color) |
| **Uncertainty Estimation** | Entropy-based confidence maps from softmax outputs |
| **BEV Generation** | Bird's Eye View projection from monocular depth maps |
| **Range Image Segmentation** | LiDAR point cloud → range image → semantic segmentation pipeline |
| **Python Bindings** | ctypes-based `pylightnet` package for scripting and integration |

---

## Requirements

### Local Build

| Dependency | Version |
|---|---|
| CUDA | 11.0 or later |
| TensorRT | 8.5, 8.6, or **10.x** |
| CMake | 3.10+ |
| GCC | ≤ 11.x |

**Tested environments:**

- CUDA 11.7 + TensorRT 8.5.2 on Ubuntu 22.04
- CUDA 12.2 + TensorRT 8.6.0 on Ubuntu 22.04
- CUDA 11.4 + TensorRT 8.6.0 on Jetson JetPack 5.1
- CUDA 11.8 + TensorRT 8.6.1 on Ubuntu 22.04
- CUDA 12.x + TensorRT 10.x on Ubuntu 22.04

### Docker Build

- Docker 24.0+
- NVIDIA Container Toolkit 1.14+

---

## Installation

### Local Build

1. Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:tier4/trt-lightnet.git
cd trt-lightnet
```

2. Install system dependencies:

```bash
sudo apt update
sudo apt install -y \
    libgflags-dev \
    libboost-all-dev \
    libopencv-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    libssl-dev
```

3. Build and install:

```bash
mkdir build && cd build
cmake ../
make -j$(nproc)
sudo make install
```

The `cnpy` library (for NumPy-format tensor debugging) is fetched automatically by CMake if not found.

### Docker Build

```bash
# Clone
git clone --recurse-submodules git@github.com:tier4/trt-lightnet.git
cd trt-lightnet

# Build image (choose your architecture)
docker build -f Dockerfile_x86     -t trt-lightnet:latest .   # x86_64
docker build -f Dockerfile_aarch64 -t trt-lightnet:latest .   # Jetson (aarch64)

# Run container
docker run -it --gpus all          trt-lightnet:latest   # x86_64
docker run -it --runtime=nvidia    trt-lightnet:latest   # Jetson
```

---

## Quick Start

### 1. Build a TensorRT Engine

An ONNX model must be converted to a TensorRT engine before inference. The engine is cached to disk and reused on subsequent runs.

```bash
cd build

# FP32 (highest accuracy, largest model)
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp32

# FP16 (recommended for most edge deployments)
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp16

# INT8 (smallest/fastest; requires calibration image list)
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision int8 --first true

# DLA engine on Xavier / Orin (INT8 only)
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision int8 --first true --dla 0
```

> **Note:** `--first true` skips INT8 quantization on the first layer, which is highly sensitive and can cause accuracy loss if quantized.

### 2. Run Inference

Three mutually exclusive input sources are supported:

| Flag | Type | Input source |
|---|---|---|
| `--d <path>` | string | Directory of image files — iterates in filename order. Press **Space** to advance, **q** to quit. |
| `--v <path>` | string | Video file (MP4, AVI, etc.) — plays frame by frame. |
| `--cam <id>` | int | Live camera — opens the device at `/dev/video<id>` via OpenCV. Use `0` for the default webcam. |

```bash
# From an image directory
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp16 --d /path/to/images

# From a video file
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp16 --v /path/to/video.mp4

# From a live camera (device ID 0)
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp16 --cam 0

# Save detection results to disk
./trt-lightnet --flagfile ../configs/YOUR_CONFIG.txt --precision fp16 \
    --d /path/to/images \
    --save-detections true \
    --save-detections-path /path/to/output
```

---

## Configuration Files

trt-lightnet uses [gflags](https://github.com/gflags/gflags)-style config files (plain text, one flag per line). Command-line flags override values in the config file.

### Minimal example (detection only)

```ini
--onnx=/path/to/model.onnx
--names=/path/to/classes.names
--rgb=/path/to/colormap.colormap
--precision=fp16
--anchors=8,9,36,11,15,28,40,36,29,72,104,24,86,67,58,140,265,70
--num_anchors=3
--c=10
--thresh=0.2
--nms_thresh=0.6
--cuda=true
```

### Multi-task example (detection + segmentation + depth)

```ini
--onnx=/path/to/multitask.onnx
--names=/path/to/classes.names
--rgb=/path/to/detect.colormap
--precision=fp16
--anchors=8,9,36,11,15,28
--num_anchors=3
--c=8
--thresh=0.2
--nms_thresh=0.6
--cuda=true

# Semantic segmentation
--mask=/path/to/segmentation.csv

# Depth + Bird's Eye View
--fx=1000.0
--fy=1000.0
--max_distance=80.0
```

### Two-stage detection (main model + subnet)

```ini
# Main detection model
--onnx=/path/to/main.onnx
--names=/path/to/main.names
--rgb=/path/to/main.colormap
--precision=fp16
--anchors=...
--num_anchors=5
--c=10
--thresh=0.2

# Subnet for fine-grained classification (e.g. traffic lights)
--subnet_onnx=/path/to/subnet.onnx
--subnet_names=/path/to/subnet.names
--subnet_rgb=/path/to/subnet.colormap
--subnet_anchors=7,7,14,14,42,42
--subnet_num_anchors=3
--subnet_c=6
--target_names=/path/to/trigger_classes.names   # classes that activate the subnet
--subnet_thresh=0.2
--batch_size=64
```

### Data file formats

| File | Format | Description |
|---|---|---|
| `.names` | One class name per line | Detection class labels |
| `.colormap` | `R,G,B` per line | Per-class visualization colors |
| `.csv` | `id,name,r,g,b,is_dynamic` | Segmentation class metadata |
| `calibration_images.txt` | One image path per line | Images used for INT8 calibration |

### INT8 calibration

Prepare a text file listing calibration images (50–500 images representative of the deployment data):

```
/data/calib/frame_000.jpg
/data/calib/frame_001.jpg
...
```

Then reference it in the config:

```ini
--calibration_images=/path/to/calibration_images.txt
--calib=Entropy   # Entropy | MinMax | Legacy | Percentile
```

---

## Command-Line Reference

All flags can be set in a config file (`--flagfile`) or passed directly on the command line. The full list is in `src/config_parser.cpp`.

### Core flags

| Flag | Type | Description |
|---|---|---|
| `--flagfile` | string | **(Required)** Path to the config file |
| `--precision` | string | Engine precision: `fp32`, `fp16`, `int8` |
| `--onnx` | string | Path to the ONNX model |
| `--names` | string | Class names file |
| `--rgb` | string | Colormap file for visualization |
| `--c` | int | Number of detection classes |
| `--thresh` | float | Detection confidence threshold |
| `--nms_thresh` | float | NMS IoU threshold |
| `--anchors` | int list | Anchor box dimensions (flat list) |
| `--num_anchors` | int | Number of anchors per scale |

### Engine build flags

| Flag | Type | Description |
|---|---|---|
| `--calib` | string | INT8 calibration algorithm |
| `--calibration_images` | string | Calibration image list file |

### Model optimization flags

| Flag | Type | Description |
|---|---|---|
| `--first` | bool | Skip INT8 quantization on the first layer (recommended: `true`) |
| `--last` | bool | Skip INT8 quantization on the last layer |
| `--sparse` | bool | Enable 2:4 structured sparsity (Ampere+ GPU required) |
| `--dla` | int | DLA core index — `0` or `1`; **Xavier / Orin only** |

> See [Tips](#tips) for details on each optimization technique.

### Inference flags

| Flag | Type | Description |
|---|---|---|
| `--d` | string | Directory of images — iterates in filename order (Space: next, q: quit) |
| `--v` | string | Video file (MP4, AVI, etc.) |
| `--cam` | int | Live camera device ID (e.g. `0` for `/dev/video0`) |
| `--save-detections` | bool | Save detection output images |
| `--save-detections-path` | string | Output directory for saved results |
| `--profile` | bool | Print per-layer latency profile |
| `--debug_tensors` | string | Comma-separated tensor names to dump |

### Segmentation / depth flags

| Flag | Type | Description |
|---|---|---|
| `--mask` | string | Segmentation CSV file |
| `--entropy` | bool | Compute and display entropy maps |
| `--fx`, `--fy` | float | Camera focal lengths (for BEV projection) |
| `--max_distance` | float | Maximum depth range (meters) |
| `--smooth` | bool | Road-guided depth smoothing |

### LiDAR range image flags

| Flag | Type | Description |
|---|---|---|
| `--lidar` | bool | Enable range image processing mode |
| `--camera_name` | string | Camera sensor identifier |

---

## Python API (pylightnet)

`pylightnet` is a ctypes-based Python wrapper around the shared library built by trt-lightnet.

### Installation

```bash
cd python
pip install setuptools==68.2.2
pip install .
```

### Demo scripts

```bash
# Detection + segmentation from a video
python scripts/demo.py \
    --flagfile /path/to/config.txt \
    --video    /path/to/video.mp4

# LiDAR range image segmentation from a T4 dataset
python scripts/range_image_demo.py \
    --t4dataset  /path/to/t4dataset \
    --camera-name CAM_FRONT_WIDE \
    --flagfile   /path/to/config.txt \
    --save-segmentation \
    --save-uncertainty \
    --output-dir ./output
```

### API usage

```python
import pylightnet

# Load config and create engine
config = pylightnet.load_config("/path/to/config.txt")
names   = pylightnet.load_names_from_file(config["names"])
colormap = pylightnet.load_colormap_from_file(config["rgb"])

lightnet = pylightnet.create_lightnet_from_config(config)

# Run inference on a BGR image (numpy array)
lightnet.infer(frame, cuda=True)

# Retrieve detection results
bboxes = lightnet.get_bboxes()
pylightnet.draw_bboxes_on_image(frame, bboxes, colormap, names)

# Retrieve segmentation mask
seg_data = pylightnet.load_segmentation_data(config["mask"])
argmax2bgr = lightnet.segmentation_to_argmax2bgr(seg_data)
lightnet.make_mask(argmax2bgr)
masks = lightnet.get_masks_from_cpp()

# Uncertainty / entropy
lightnet.make_entropy()
entropy_maps = lightnet.get_entropy_maps_from_cpp()

# Cleanup
lightnet.destroy()
```

### Docker test

```bash
# Build and run the pylightnet test inside Docker
make test-pylightnet
```

---

## Advanced Features

### Uncertainty Estimation

When `--entropy` is set, trt-lightnet computes per-class entropy from the softmax segmentation output. High entropy indicates low model confidence and is useful for out-of-distribution detection.

```bash
./trt-lightnet --flagfile config.txt --precision fp16 --d images/ --entropy
```

### Bird's Eye View (BEV) Projection

With camera intrinsics (`--fx`, `--fy`) and a depth output head, trt-lightnet can back-project the depth map into a top-down occupancy grid. Combine with `--smooth` to use road-plane segmentation for improved flatness correction.

### Two-Stage Hierarchical Detection

A primary model detects coarse categories; for any detected instance belonging to a `--target_names` class, a secondary subnet (`--subnet_onnx`) classifies the crop at higher resolution (e.g. red/amber/green/arrow for traffic lights). Batch inference over all crops is controlled by `--batch_size`.

### LiDAR Range Image Segmentation

Point cloud files (binary `.bin` format from Seyond or compatible LiDAR) can be converted on-the-fly to range images and passed through a segmentation network:

```bash
./trt-lightnet --flagfile configs/CoMLOps-Large-RangeImage-Segmenation-Model.txt \
    --precision fp16 --lidar --camera_name CAM_FRONT_WIDE
```

### Per-Layer Profiling

```bash
./trt-lightnet --flagfile config.txt --precision fp16 --d images/ --profile
```

Prints a table of per-layer latency to help identify bottlenecks.

---

## Tips

### `--first` — Protect the First Layer from INT8 Quantization

The first convolutional layer of a CNN processes raw pixel values and is highly sensitive to quantization error. Quantizing it to INT8 often causes a noticeable accuracy drop because the dynamic range of input activations is much wider than that of intermediate layers.

```bash
# Recommended: skip INT8 on the first layer
./trt-lightnet --flagfile config.txt --precision int8 --first true
```

Setting `--first true` keeps the first layer in FP32 (or FP16) while quantizing the rest to INT8, giving the best accuracy–speed tradeoff. `--last true` does the same for the final output layer.

---

### `--sparse` — 2:4 Structured Sparsity

2:4 structured sparsity enforces that **at least 2 out of every 4 consecutive weights are zero**. NVIDIA Ampere (and later) GPUs have dedicated sparse tensor cores that exploit this pattern for a near **2× throughput boost** with no additional latency penalty.

```
Before sparsity:  [0.3, -0.7,  0.1, -0.5]
After  sparsity:  [0.0, -0.7,  0.0, -0.5]  ← 2 zeros forced per 4 values
```

**Requirements:**
- GPU: Ampere (RTX 30xx / A-series) or later
- The model must have been **trained with sparsity-aware training** (e.g. NVIDIA ASP or PyTorch semi-structured sparsity). Enabling `--sparse` on a dense model will degrade accuracy.

```bash
./trt-lightnet --flagfile config.txt --precision fp16 --sparse true
```

---

### `--dla` — NVIDIA Deep Learning Accelerator

DLA is a fixed-function hardware accelerator built into **Jetson Xavier and Orin** SoCs. Offloading layers to DLA frees the GPU for other tasks and significantly reduces power consumption — important for battery-constrained or thermally-limited deployments.

```bash
# Use DLA core 0 (Xavier / Orin only)
./trt-lightnet --flagfile config.txt --precision int8 --first true --dla 0
```

**Key points:**
- DLA only supports **INT8** (and limited FP16) precision.
- Not all layer types are supported by DLA; unsupported layers automatically fall back to GPU.
- Two DLA cores are available (`--dla 0` or `--dla 1`) and can be used simultaneously for different models.
- DLA is **not available** on desktop/server GPUs — use only on Jetson Xavier / Orin.

For a detailed overview of DLA architecture and usage in autonomous driving, see:
> [TIER IV Tech Blog — NVDLA on Jetson (docswell, 2023)](https://www.docswell.com/s/TIER_IV/KGX2L8-2023-07-24-120048#p27)

---

## Project Structure

```
trt-lightnet/
├── CMakeLists.txt              # Build configuration
├── Dockerfile_x86              # Docker image for x86_64
├── Dockerfile_aarch64          # Docker image for Jetson
├── configs/                    # Example config files (.txt) and precision tables (.json)
├── data/                       # Colormaps, class name files, segmentation CSVs
├── include/
│   ├── tensorrt_lightnet/      # Main inference engine headers
│   ├── tensorrt_common/        # TensorRT wrapper
│   ├── cuda_utils/             # CUDA memory helpers
│   ├── sensor/                 # Sensor calibration parsing
│   ├── pcdUtils/               # Point cloud utilities
│   └── fswp/                   # Free-space detection
├── src/
│   ├── lightnet_detector.cpp   # Main executable entry point
│   ├── tensorrt_lightnet/      # Inference engine implementation
│   ├── tensorrt_common/        # TensorRT engine build/load
│   ├── preprocess.cu           # GPU preprocessing CUDA kernels
│   ├── config_parser.cpp       # gflags config parsing
│   ├── sensor/                 # Sensor config parsers
│   ├── pcdUtils/               # LiDAR → range image conversion
│   └── fswp/                   # Free-space polygon extraction
├── python/
│   ├── setup.py                # Python package build
│   ├── _pylightnet.py          # ctypes wrapper internals
│   └── scripts/
│       ├── demo.py             # Camera/video demo
│       ├── range_image_demo.py # LiDAR T4 dataset demo
│       ├── anonymizer.py       # Privacy blurring script
│       └── ensemble_anonymizer.py
└── packages/                   # Auto-fetched external deps (cnpy)
```

---

## Implementation References

- [LightNet](https://github.com/daniel89710/lightNet) — base CNN architecture
- [TensorRT](https://github.com/NVIDIA/TensorRT) — inference optimizer
- [NVIDIA DeepStream SDK](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)
- [yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt) — YOLO-TRT patterns
- [trt-yoloXP](https://github.com/tier4/trt-yoloXP) — TIER IV predecessor

## References

[1] [LightNet](https://github.com/daniel89710/lightNet)  
[2] [TensorRT](https://developer.nvidia.com/tensorrt)  
[3] [Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)  
[4] [NVDLA](http://nvdla.org/)  
[5] [Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)  
[6] [lightNet-TR (original)](https://github.com/daniel89710/trt-lightnet)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

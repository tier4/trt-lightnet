# TensorRT-LightNet: High-Efficiency and Real-Time CNN Implementation on Edge AI

trt-lightNet is a CNN implementation optimized for edge AI devices that combines the advantages of LightNet <sup>[[1]](#references)</sup> and TensorRT <sup>[[2]](#references)</sup>. LightNet is a lightweight and high-performance neural network framework designed for edge devices, while TensorRT is a high-performance deep learning inference engine developed by NVIDIA for optimizing and running deep learning models on GPUs. trt-lightnet uses the Network Definition API provided by TensorRT to integrate LightNet into TensorRT, allowing it to run efficiently and in real-time on edge devices.
This is a reproduction of trt-lightnet <sup>[[6]](#references)</sup>, which generates a TensorRT engine from the ONNX format.

## Key Improvements

### 2:4 Structured Sparsity

trt-lightnet utilizes 2:4 structured sparsity <sup>[[3]](#references)</sup>  to further optimize the network. 2:4 structured sparsity means that two values must be zero in each contiguous block of four values, resulting in a 50% reduction in the number of weights. This technique allows the network to use fewer weights and computations while maintaining accuracy.

![Sparsity](https://developer-blogs.nvidia.com/ja-jp/wp-content/uploads/sites/6/2022/06/2-4-structured-sparse-matrix.png "sparsity")

### NVDLA Execution

trt-lightnet also supports the execution of the neural network on the NVIDIA Deep Learning Accelerator (NVDLA) <sup>[[4]](#references)</sup> , a free and open architecture that provides high performance and low power consumption for deep learning inference on edge devices. By using NVDLA, trt-lightnet can further improve the efficiency and performance of the network on edge devices.

![NVDLA](https://i0.wp.com/techgrabyte.com/wp-content/uploads/2019/09/Nvidia-Open-Source-Its-Deep-Learning-Inference-Compiler-NVDLA-2.png?w=768&ssl=1 "NVDLA")


### Multi-Precision Quantization

In addition to post training quantization <sup>[[5]](#references)</sup>, trt-lightnet also supports multi-precision quantization, which allows the network to use different precision for weights and activations. By using mixed precision, trt-lightnet can further reduce the memory usage and computational requirements of the network while still maintaining accuracy. By writing it in CFG, you can set the precision for each layer of your CNN.

![Quantization](https://developer-blogs.nvidia.com/wp-content/uploads/2021/07/qat-training-precision.png "Quantization")



### Multitask Execution (Detection/Segmentation)

trt-lightnet also supports multitask execution, allowing the network to perform both object detection and segmentation tasks simultaneously. This enables the network to perform multiple tasks efficiently on edge devices, saving computational resources and power.


## Installation

### Requirements

#### For Local Installation

-   CUDA 11.0 or later
-   TensorRT 8.5 or 8.6
-   cnpy for debug of tensors
This repository has been tested with the following environments:

- CUDA 11.7 + TensorRT 8.5.2 on Ubuntu 22.04
- CUDA 12.2 + TensorRT 8.6.0 on Ubuntu 22.04
- CUDA 11.4 + TensorRT 8.6.0 on Jetson JetPack5.1
- CUDA 11.8 + TensorRT 8.6.1 on Ubuntu 22.04
- gcc <= 11.x

#### For Docker Installation

-  Docker
-  NVIDIA Container Toolkit

This repository has been tested with the following environments:

- Docker 24.0.7 + NVIDIA Container Toolkit 1.14.3 on Ubuntu 20.04

### Steps for Local Installation

1.  Clone the repository, and the dependent packages

```shell
$ git clone --recurse-submodules git@github.com:tier4/trt-lightnet.git
$ cd trt-lightnet
```

2.  Install libraries.

```shell
$ sudo apt update
$ sudo apt install libgflags-dev
$ sudo apt install libboost-all-dev
$ sudo apt install libopencv-dev
$ sudo apt install libeigen3-dev
$ sudo apt install nlohmann-json3-dev
```


3.  Compile the TensorRT implementation.

```shell
$ mkdir build && cd build
$ cmake ../
$ make -j
```

### Steps for Docker Installation

1.  Clone the repository.

```shell
$ git clone --recurse-submodules git@github.com:tier4/trt-lightnet.git
$ cd trt-lightnet
```

2.  Build the docker image.

```shell
# For x86
$ docker build -f Dockerfile_x86 -t trt-lightnet:latest .
# For aarch64
$ docker build -f Dockerfile_aarch64 -t trt-lightnet:latest .
```

3. Run the docker container.

```shell
# For x86
$ docker run -it --gpus all trt-lightnet:latest
# For aarch64
$ docker run -it --runtime=nvidia trt-lightnet:latest
```

## Model
 T.B.D

## Usage

### Converting a LightNet model to a TensorRT engine

Build FP32 engine
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision fp32
```

Build FP16(HALF) engine
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision fp16
```

Build INT8 engine  
(You need to prepare a list for calibration in "configs/calibration_images.txt".)
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision int8 --first true
```
First layer is much more sensitive for quantization.
Threfore, the first layer is not quanitzed using "--first true"

Build DLA engine (Supported by only Xavier and Orin)
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision int8 --first true --dla [0/1]
```

### Inference with the TensorRT engine

Inference from images
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision [fp32/fp16/int8] --first true {--dla [0/1]} --d DIRECTORY
```

Inference from video
```shell
$ ./trt-lightnet --flagfile ../configs/CONFIGS.txt --precision [fp32/fp16/int8] --first true {--dla [0/1]} --v VIDEO
```

## Most commonly used arguments and options
Here shows a part of most commonly used options for `trt-lightnet`. For more flags implemented, please refer to `src/config_parser.cpp`

- `--flagfile <path>` (required):
  - The path to the config file, which contains some basic operations (e.g. onnx, thresh)
  - Note that the options in the config file can be overwritten from command line.
  - Example: `../configs/CONFIGS.txt`

- `--precision <level>` (required):
  - Specified the quantization level during building the inference engine. Available options are:
    - `fp32`: Full precision inference engine
    - `fp16`: Half precision inference engine
    - `int8`: int8 precision inference engine
  - Note that, if `int8` is picked, it requires `calibration_images.txt` in `configs/` directory.
  - Example: `int8`

- `--first` (optional):
  - A boolean flag to choose if applying quantization to first layer or not.
    - Example: `true`
  - In general, the first layer is a sensitive layer, where the quantization may leads to precision  drop. So set `--first` as `true` to skip the quantization is recommended.


- `--d <path>` (optional):
  - The path to the directory of images
  - Example: `../sample_data/images`
  - During the inference, user can press `space` to jump to next image to infer.
  
- `--v <path>` (optional):
  - The path to the video file
  - Example: `../sample_data/sample.mp4`

- `--save-detections` (optional):
  - A boolean flag to choose if save the detections result or not.
  - Example: `true`

- `--save-detections-path` (optional):
  - The flag to determinate the output directory if `--save-detections` is set `true`
  - Example: `../workspace/detections_result`


## Implementation

trt-lightnet is built on the LightNet framework and integrates with TensorRT using the Network Definition API. The implementation is based on the following repositories:

-   LightNet: [https://github.com/daniel89710/lightNet](https://github.com/daniel89710/lightNet)
-   TensorRT: [https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
-   NVIDIA DeepStream SDK: [https://github.com/NVIDIA-AI-IOT/deepstream\_reference\_apps/tree/restructure](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/restructure)
-   YOLO-TensorRT: [https://github.com/enazoe/yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt)
-   trt-yoloXP: [https://github.com/tier4/trt-yoloXP]

## Conclusion

trt-lightnet is a powerful and efficient implementation of CNNs using Edge AI. With its advanced features and integration with TensorRT, it is an excellent choice for real-time object detection and semantic segmentation applications on edge devices.

# References
[1]. [LightNet](https://github.com/daniel89710/lightNet)  
[2]. [TensorRT](https://developer.nvidia.com/tensorrt)  
[3]. [Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)  
[4]. [NVDLA](http://nvdla.org/)  
[5]. [Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)  
[6]. [lightNet-TR](https://github.com/daniel89710/trt-lightnet)


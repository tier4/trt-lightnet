FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TENSORRT_VERSION=8.6.1.6-1+cuda11.8
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Tokyo

# Install fundamental packages
RUN apt update -y \
  && apt upgrade -y \
  && apt install -y build-essential software-properties-common git cmake wget libgflags-dev libboost-all-dev libopencv-dev python3.10 python3-pip python3-setuptools libeigen3-dev nlohmann-json3-dev zlib1g-dev libssl-dev

# Install TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update
RUN apt install \
  libnvinfer-headers-dev=${TENSORRT_VERSION} \
  libnvinfer8=${TENSORRT_VERSION} \
  libnvinfer-dev=${TENSORRT_VERSION} \
  libnvonnxparsers8=${TENSORRT_VERSION} \
  libnvonnxparsers-dev=${TENSORRT_VERSION} \
  libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
  libnvinfer-plugin8=${TENSORRT_VERSION} \
  libnvinfer-plugin-dev=${TENSORRT_VERSION} \
  libnvparsers8=${TENSORRT_VERSION} \
  libnvparsers-dev=${TENSORRT_VERSION} \
  && rm -rf /var/lib/apt/lists/*

# Copy source code
RUN mkdir -p /opt/app
COPY . /opt/app
WORKDIR /opt/app

CMD ["/bin/bash"]

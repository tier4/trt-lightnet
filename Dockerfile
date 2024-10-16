FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TENSORRT_VERSION=10.5.0.18-1+cuda12.6
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Tokyo

# Install fundamental packages
RUN apt update -y \
  && apt upgrade -y \
  && apt install -y build-essential software-properties-common git cmake wget libgflags-dev libboost-all-dev libopencv-dev

# Install TensorRT
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/ /"
RUN apt-get update
RUN apt install \
  libnvinfer-headers-dev=${TENSORRT_VERSION} \
  libnvinfer10=${TENSORRT_VERSION} \
  libnvinfer-dev=${TENSORRT_VERSION} \
  libnvonnxparsers10=${TENSORRT_VERSION} \
  libnvonnxparsers-dev=${TENSORRT_VERSION} \
  libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
  libnvinfer-plugin10=${TENSORRT_VERSION} \
  libnvinfer-plugin-dev=${TENSORRT_VERSION} \
  libnvparsers10=${TENSORRT_VERSION} \
  && rm -rf /var/lib/apt/lists/*

# Build app
RUN mkdir -p /opt/app
COPY . /opt/app
WORKDIR /opt/app
RUN mkdir -p build && cd build && cmake .. && make -j

# Default CMD
ENTRYPOINT ["/opt/app/docker-entrypoint.sh"]
CMD ["/bin/bash"]

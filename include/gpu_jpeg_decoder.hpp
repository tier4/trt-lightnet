// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRT_LIGHTNET__GPU_JPEG_DECODER_HPP_
#define TRT_LIGHTNET__GPU_JPEG_DECODER_HPP_

#include <iostream>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <opencv2/core/mat.hpp>

#define CHECK_CUDA(call) trt_lightnet::checkCuda(call, __FILE__, __LINE__)
#define CHECK_NVJPEG(call) trt_lightnet::checkNvJpeg(call, __FILE__, __LINE__)

namespace trt_lightnet {

template <typename T>
void checkCuda (T call, const char* file, int line) {
  cudaError_t _e = (call);
  if (_e != cudaSuccess) {
    std::cerr << "CUDA Runtime failure: '" << cudaGetErrorString(_e) << "' at " << file << ":" << line << std::endl;
    exit(1);
  }
}

template <typename T>
void checkNvJpeg (T call, const char* file, int line) {
  nvjpegStatus_t _e = (call);
  if (_e != NVJPEG_STATUS_SUCCESS) {
    std::cerr << "NVJPEG failure: '" << _e << "' at " << file << ":" << line << std::endl;
    exit(1);
  }
}

class GPUJpegDecoder {
 public:
  explicit GPUJpegDecoder();
  ~GPUJpegDecoder();
  cv::Mat decode(const std::vector<uint8_t>& encoded_data);

 protected:
  nvjpegHandle_t nvjpeg_handler_;
  nvjpegJpegState_t nvjpeg_state_;
  nvjpegImage_t nvjpeg_image_;

};

}  // namespace trt_lightnet

#endif // TRT_LIGHTNET__GPU_JPEG_DECODER_HPP_

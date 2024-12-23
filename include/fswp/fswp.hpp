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

#ifndef TENSORRT_FACESWAPPER__TENSORRT_FACESWAPPER_HPP_
#define TENSORRT_FACESWAPPER__TENSORRT_FACESWAPPER_HPP_

#include <NvInfer.h>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <vector>

namespace fswp {
struct BBox {
  float x1, y1;  // xmin, ymin
  float x2, y2;  // xmax, ymax
};

class FaceSwapper {
 public:
  FaceSwapper(const std::filesystem::path &onnx_path,
              const tensorrt_common::BuildConfig &build_config, const std::size_t batch_size = 1,
              std::string precision = "fp32", const std::size_t max_workspace_size = 1 << 30);
  ~FaceSwapper();
  void allocateMemory();
  cv::Mat inpaint(const cv::Mat &image, const std::vector<BBox> &bboxes);
  void printProfiling(void);
  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;
  std::vector<cuda_utils::CudaUniquePtr<float[]>> input_d_;
  std::vector<cuda_utils::CudaUniquePtr<float[]>> output_d_;
  std::vector<cuda_utils::CudaUniquePtrHost<float[]>> output_h_;
  cuda_utils::StreamUniquePtr stream_{cuda_utils::makeCudaStream()};
  std::size_t batch_size_;
};

}  // namespace fswp

#endif

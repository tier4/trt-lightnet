// Copyright 2023 TIER IV, Inc.
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

#ifndef TENSORRT_COMMON__SIMPLE_PROFILER_HPP_
#define TENSORRT_COMMON__SIMPLE_PROFILER_HPP_

#include <NvInfer.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>


namespace tensorrt_common
{
/**
 * @struct LayerInfo
 * @brief Structural metadata captured for a network layer (mainly convolutions).
 */
struct LayerInfo
{
  int in_c;                  ///< Number of input channels.
  int out_c;                 ///< Number of output channels.
  int w;                     ///< Input width.
  int h;                     ///< Input height.
  int k;                     ///< Kernel size.
  int stride;                ///< Convolution stride.
  int groups;                ///< Number of convolution groups.
  nvinfer1::LayerType type;  ///< TensorRT layer type.
};

/**
 * @class SimpleProfiler
 * @brief Collects per-layer profile information, assuming times are reported in the same order.
 */
class SimpleProfiler : public nvinfer1::IProfiler
{
public:
  /**
   * @struct Record
   * @brief Accumulated timing statistics for a single layer.
   */
  struct Record
  {
    float time{0};         ///< Total accumulated runtime in milliseconds.
    int count{0};          ///< Number of times the layer was invoked.
    float min_time{-1.0};  ///< Minimum observed runtime (-1 until first report).
    int index;             ///< Stable display order index assigned on first report.
  };

  /// Constructs a profiler, optionally merging records from existing profilers.
  SimpleProfiler(
    std::string name,
    const std::vector<SimpleProfiler> & src_profilers = std::vector<SimpleProfiler>());

  /// Reports the runtime of one layer (TensorRT IProfiler callback).
  void reportLayerTime(const char * layerName, float ms) noexcept override;

  /// Records structural metadata for a layer for use in the report.
  void setProfDict(nvinfer1::ILayer * layer) noexcept;

  /// Streams a formatted profile report.
  friend std::ostream & operator<<(std::ostream & out, SimpleProfiler & value);

private:
  std::string m_name;
  std::map<std::string, Record> m_profile;
  int m_index;
  std::map<std::string, LayerInfo> m_layer_dict;
};
}  // namespace tensorrt_common
#endif  // TENSORRT_COMMON__SIMPLE_PROFILER_HPP_

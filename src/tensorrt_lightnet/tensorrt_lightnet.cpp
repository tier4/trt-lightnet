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

/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"
#include <tensorrt_lightnet/colormap.hpp>
#include <tensorrt_lightnet/calibrator.hpp>
#include <tensorrt_lightnet/preprocess.hpp>
#include <tensorrt_lightnet/tensorrt_lightnet.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>
#include <fstream>
#include <iostream>
#include <filesystem> // Use standardized filesystem library
#include <cassert>
#include <nlohmann/json.hpp>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>  
#include <sstream>
#include <iomanip>
#include <omp.h>

using json = nlohmann::json;
extern const unsigned char jet_colormap[256][3];
extern const unsigned char magma_colormap[256][3];


inline float quantize_angle(float angle, int bin)
{
  return (int(angle/bin)) * bin;
}

inline double
calculateAngle(double x1, double y1, double x2, double y2) {
  double deltaY = y2 - y1;
  
  double deltaX = x2 - x1;

  double angleRadians = atan2(deltaY, deltaX);

  double angleDegrees = angleRadians * (180.0 / M_PI);

  return angleDegrees;
}

std::vector<std::vector<cv::Point>> get_polygons( const cv::Mat &mask)
{  
  std::vector<std::vector<cv::Point>> contours;  
  //cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  // 🚀 メモリ効率の最適化
  contours.reserve(1000);  // 適切な予測値に設定 (データ量に応じて調整)
  // 🔄 CHAIN_APPROX_SIMPLE を使用してデータ量削減
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

void write_json_with_order(const json& j, const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Error: Could not create JSON file." << std::endl;
    return;
  }
  file << std::setw(4) << j << std::endl;
  file.close();
}

/**
 * @brief Draws an arrow on the given image.
 *
 * This function draws an arrow from point pt1 to point pt2 with the specified color, thickness, line type, and shift.
 *
 * @param image The image on which the arrow will be drawn.
 * @param pt1 The starting point of the arrow.
 * @param pt2 The ending point of the arrow.
 * @param color The color of the arrow.
 * @param thickness The thickness of the arrow lines. Default value is 1.
 * @param lineType The type of the arrow lines. Default value is 8 (8-connected line).
 * @param shift The number of fractional bits in the point coordinates. Default value is 0.
 */
inline
void arrow(cv::Mat image, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness = 0, int lineType = 8, int shift = 0)
{
  // Calculate the vector components between pt1 and pt2
  float vx = (float)(pt2.x - pt1.x);
  float vy = (float)(pt2.y - pt1.y);
  float v = sqrt(vx * vx + vy * vy);
  float ux = vx / v;
  float uy = vy / v;

  // Arrow parameters
  float w = 8.0f; // Width of the arrowhead
  float h = 8.0f; // Height of the arrowhead
  cv::Point ptl, ptr;

  // Calculate the left side point of the arrowhead
  ptl.x = (int)((float)pt2.x - uy * w - ux * h);
  ptl.y = (int)((float)pt2.y + ux * w - uy * h);

  // Calculate the right side point of the arrowhead
  ptr.x = (int)((float)pt2.x + uy * w - ux * h);
  ptr.y = (int)((float)pt2.y - ux * w - uy * h);

  // Draw the arrow
  cv::line(image, pt1, pt2, color, thickness, lineType, shift); // Main line
  cv::line(image, pt2, ptl, color, thickness, lineType, shift); // Left side of the arrowhead
  cv::line(image, pt2, ptr, color, thickness, lineType, shift); // Right side of the arrowhead
}

namespace
{
  /**
   * Clamps a value between a specified range.
   * 
   * @param value The value to clamp.
   * @param low The lower bound of the range.
   * @param high The upper bound of the range.
   * @return The clamped value.
   */
  template <typename T>
  T CLAMP(const T& value, const T& low, const T& high)
  {
    return value < low ? low : (value > high ? high : value);
  }

  /**
   * Trims leading whitespace from a string.
   * 
   * @param s Reference to the string to trim.
   */
  static void trimLeft(std::string & s)
  {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  }

  /**
   * Trims trailing whitespace from a string.
   * 
   * @param s Reference to the string to trim.
   */
  static void trimRight(std::string & s)
  {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
  }

  /**
   * Trims both leading and trailing whitespace from a string.
   * 
   * @param s Reference to the string to trim.
   * @return The trimmed string.
   */
  std::string trim(std::string & s)
  {
    trimLeft(s);
    trimRight(s);
    return s;
  }

  /**
   * Checks if a file exists.
   * 
   * @param file_name The name of the file to check.
   * @param verbose Whether to print a message if the file does not exist.
   * @return true if the file exists, false otherwise.
   */
  bool fileExists(const std::string & file_name, bool verbose)
  {
    if (!std::filesystem::exists(std::filesystem::path(file_name))) {
      if (verbose) {
	std::cout << "File does not exist: " << file_name << std::endl;
      }
      return false;
    }
    return true;
  }

  /**
   * Loads a list of strings from a text file, one string per line.
   * 
   * @param filename The path to the text file.
   * @return A vector containing the lines read from the file.
   */
  std::vector<std::string> loadListFromTextFile(const std::string & filename)
  {
    assert(fileExists(filename, true));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if (!f) {
      std::cerr << "Failed to open " << filename << std::endl;
      assert(false); // Changed to assert(false) for clarity.
    }

    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) {
	continue;
      } else {
	list.push_back(trim(line));
      }
    }

    return list;
  }

  /**
   * Loads a list of image filenames from a text file and prepends a prefix if the files don't exist.
   * 
   * @param filename The name of the text file containing the list of images.
   * @param prefix The prefix to prepend to each filename if it does not exist.
   * @return A vector of image filenames, potentially with prefixes.
   */
  std::vector<std::string> loadImageList(const std::string & filename, const std::string & prefix)
  {
    std::vector<std::string> fileList = loadListFromTextFile(filename);
    for (auto & file : fileList) {
      if (fileExists(file, false)) {
	continue;
      } else {
	std::string prefixed = prefix + file;
	if (fileExists(prefixed, false))
	  file = prefixed;
	else
	  std::cerr << "WARNING: Couldn't find: " << prefixed << " while loading: " << filename << std::endl;
      }
    }
    return fileList;
  }
}  // anonymous namespace

namespace tensorrt_lightnet
{

  /**
   * Constructs a TrtLightnet object configured for running TensorRT inference.
   *
   * This constructor initializes a TrtLightnet object using provided model and inference configurations.
   * It handles setting up the TensorRT environment, including configuring precision, calibrators, and
   * workspace size based on the given settings. If INT8 precision is specified, it also manages the
   * calibration process using provided calibration images. Exception handling is included to ensure
   * that necessary prerequisites for chosen precision modes are met.
   *
   * @param model_config Configuration struct containing model-specific parameters such as path, class number,
   * score threshold, NMS threshold, and anchor configurations.
   * @param inference_config Configuration struct containing inference-specific settings like precision,
   * calibration images, workspace size, and batch configuration.
   * @param build_config Struct containing build-specific settings for TensorRT including the calibration type and clip value.
   * @throws std::runtime_error If necessary calibration parameters are missing for INT8 precision or if
   * TensorRT engine initialization fails.
   */  
  TrtLightnet::TrtLightnet(ModelConfig &model_config, InferenceConfig &inference_config, tensorrt_common::BuildConfig build_config, const std::string depth_format)
  {
    const std::string& model_path = model_config.model_path;
    const std::string& precision = inference_config.precision;
    const int num_class = model_config.num_class;    
    const float score_threshold = model_config.score_threshold;
    const float nms_threshold = model_config.nms_threshold;
    const std::vector<int> anchors = model_config.anchors;
    int num_anchor = model_config.num_anchors;
    std::string calibration_image_list_path = inference_config.calibration_images;
    const double norm_factor = 1.0;
    const tensorrt_common::BatchConfig& batch_config = inference_config.batch_config;
    const size_t max_workspace_size = inference_config.workspace_size;
    src_width_ = -1;
    src_height_ = -1;
    norm_factor_ = norm_factor;
    batch_size_ = batch_config[2];
    multitask_ = 0;

    
    if (precision == "int8") {
      if (build_config.clip_value <= 0.0 && calibration_image_list_path.empty()) {
	throw std::runtime_error(
				 "For INT8 precision, calibration_image_list_path must be provided if clip_value is not set.");
      }

      int max_batch_size = batch_size_;
      nvinfer1::Dims input_dims = tensorrt_common::get_input_dims(model_path);
      std::vector<std::string> calibration_images;
      if (!calibration_image_list_path.empty()) {
	calibration_images = loadImageList(calibration_image_list_path, "");
      }

      tensorrt_lightnet::ImageStream stream(max_batch_size, input_dims, calibration_images);

      fs::path calibration_table{model_path};
      fs::path histogram_table{model_path};
      std::string table_extension = build_config.calib_type_str == "Entropy" ? "EntropyV2-calibration.table" :
	(build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") ? "Legacy-calibration.table" :
	"MinMax-calibration.table";

      calibration_table.replace_extension(table_extension);
      histogram_table.replace_extension("histogram.table");
      
      // Initialize the correct calibrator based on the calibration type.
      std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator = initializeCalibrator(build_config, stream, calibration_table, histogram_table, norm_factor_);

      //trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      trt_common_ = std::make_shared<tensorrt_common::TrtCommon>(      
								 model_path, precision, std::move(calibrator), batch_config, max_workspace_size, build_config);
    } else {
      trt_common_ = std::make_shared<tensorrt_common::TrtCommon>(            
								 //trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
								 model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
    }

    trt_common_->setup();
    if (!trt_common_->isInitialized()) {
      //throw std::runtime_error("TensorRT engine initialization failed.");
      std::cerr << ("TensorRT engine initialization failed.") << std::endl;
      trt_common_ = nullptr;
      return;
    }

    // Initialize class members
    num_class_ = num_class;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    anchors_ = anchors;
    num_anchor_ = num_anchor;
    max_index_ = -1;
    // Allocate GPU memory for inputs and outputs based on tensor dimensions.
    allocateMemory();

    h_img_ = NULL;
    d_img_ = NULL;
    d_depthmap_ = NULL;
    d_mask_ = NULL;
    d_bevmap_ = NULL;
    d_depth_colormap_ = NULL;
    d_colorMap_ = NULL;
    d_resized_mask_  = NULL;   
    int chan_size = (4 + 1 + num_class_) * num_anchor_;    
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify depth map tensors by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);
	// Check if the tensor name contains "lgx" and the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  int src_size = outputH * outputW * 3;
	  d_depthmap_ = cuda_utils::make_unique<unsigned char[]>(src_size);
	  d_depth_colormap_ = cuda_utils::make_unique<unsigned char[]>(src_size);
	  if (depth_format == "magma") {
	    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_depth_colormap_.get(), magma_colormap, sizeof(unsigned char) * 256 * 3, cudaMemcpyHostToDevice, *stream_));
	  } else {
	    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_depth_colormap_.get(), jet_colormap, sizeof(unsigned char) * 256 * 3, cudaMemcpyHostToDevice, *stream_));
	  }
	} else 	if (contain(name, "softmax")) {
	  int src_size = outputH * outputW * chan;
	  h_entropy_ = cuda_utils::make_unique_host<float[]>(src_size, cudaHostAllocPortable);	  
	  //CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_entropy_, sizeof(float) * src_size));
	  d_entropy_ = cuda_utils::make_unique<float[]>(src_size);
	}
      }
    }
  }

  /**
   * Initializes the INT8 calibrator based on the specified calibration type.
   * 
   * @param build_config Configuration settings for building the TensorRT engine.
   * @param stream An ImageStream object providing calibration images.
   * @param calibration_table_path Path to the calibration table file.
   * @param histogram_table_path Path to the histogram table file, used by certain calibrators.
   * @param norm_factor Normalization factor to be applied to the input images.
   * @return A unique_ptr to the initialized INT8 calibrator.
   */
  std::unique_ptr<nvinfer1::IInt8Calibrator> TrtLightnet::initializeCalibrator(
									       const tensorrt_common::BuildConfig& build_config,
									       tensorrt_lightnet::ImageStream& stream,
									       const fs::path& calibration_table_path,
									       const fs::path& histogram_table_path,
									       double norm_factor)
  {
    // Choose the calibrator type based on the configuration.
    if (build_config.calib_type_str == "Entropy") {
      return std::make_unique<tensorrt_lightnet::Int8EntropyCalibrator>(
									stream, calibration_table_path.string(), norm_factor);
    } else if (build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") {
      // Assuming percentile and legacy use the same calibrator with different settings.
      const double quantile = 0.999999;
      const double cutoff = 0.999999;
      return std::make_unique<tensorrt_lightnet::Int8LegacyCalibrator>(
								       stream, calibration_table_path.string(), histogram_table_path.string(), norm_factor, true, quantile, cutoff);
    } else { // Defaulting to MinMax calibrator if none specified.
      return std::make_unique<tensorrt_lightnet::Int8MinMaxCalibrator>(
								       stream, calibration_table_path.string(), norm_factor);
    }
  }

  /**
   * Allocates memory for the input and output tensors based on the binding dimensions of the network.
   */
  void TrtLightnet::allocateMemory() {
    for (int i = 0; i < trt_common_->getNbBindings(); i++) {
      std::string name = trt_common_->getIOTensorName(i);
      const auto dims = trt_common_->getBindingDimensions(i);
      nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);
      if (trt_common_->bindingIsInput(i)) {
	std::cout << "(Input)  ";
      } else {
	std::cout << "(Output) ";
      }
      std::cout << name << " => " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3] << " (" << trt_common_->dataType2String(dataType)  << ")" << std::endl;

      // Calculate the tensor volume.
      const auto volume = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());

      if (i == 0) { // Input tensor
	input_d_ = cuda_utils::make_unique<float[]>(batch_size_ * volume);
      } else { // Output tensors
	output_d_.push_back(cuda_utils::make_unique<float[]>(batch_size_ * volume));
	output_h_.push_back(cuda_utils::make_unique_host<float[]>(batch_size_ * volume, cudaHostAllocPortable));
      }
    }
  }

  TrtLightnet::~TrtLightnet() {
    // Cleanup if needed.
  }
  
  /**
   * Prints the profiling information related to the inference process.
   * This method delegates to the underlying TRTCommon instance to handle the details.
   */
  void TrtLightnet::printProfiling(void)
  {
    if (trt_common_) { // Ensure trt_common_ is not a nullptr before attempting to call its methods.
      trt_common_->printProfiling();
    } else {
      std::cerr << "Error: TRTCommon instance is not initialized." << std::endl;
    }
  }

  /**
   * Preprocesses a batch of images for inference.
   * This involves normalizing the images, converting them to the NCHW format,
   * and copying the processed data to the GPU.
   *
   * @param images A vector of images (cv::Mat) to be processed.
   */  
  void TrtLightnet::preprocess(const std::vector<cv::Mat> &images) {
    // Ensure there are images to process.
    if (images.empty()) {
      std::cerr << "Preprocess called with an empty image batch." << std::endl;
      return;
    }
    const auto batch_size = images.size();
    auto input_dims = trt_common_->getBindingDimensions(0);
    input_dims.d[0] = batch_size; // Adjust the batch size in dimensions.
    trt_common_->setBindingDimensions(0, input_dims); // Update dimensions with the new batch size.
    const float inputH = static_cast<float>(input_dims.d[2]);
    const float inputW = static_cast<float>(input_dims.d[3]);
    const float inputC = static_cast<float>(input_dims.d[1]);
    std::vector<cv::Mat>  src;
    if (inputC == 1) {
      for (int b = 0; b < (int)batch_size; b++) {
	cv::Mat gray;
	cv::cvtColor(images[b], gray, cv::COLOR_BGR2GRAY);
	src.push_back(gray);
      }
    } else {
      src = images;
    }
    
    // Normalize images and convert to blob directly without additional copying.
    float scale = 1 / 255.0;
    const auto nchw_images = cv::dnn::blobFromImages(src, scale, cv::Size(inputW, inputH), cv::Scalar(0.0, 0.0, 0.0), true);   

    // If the data is continuous, we can use it directly. Otherwise, we need to clone it for contiguous memory.
    input_h_ = nchw_images.isContinuous() ? nchw_images.reshape(1, nchw_images.total()) : nchw_images.reshape(1, nchw_images.total()).clone();
    // Ensure the input device buffer is allocated with the correct size and copy the data.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice, *stream_));    
  }

  /**
   * @brief Preprocesses input images on the GPU for inference.
   *
   * This function prepares the input image data to be compatible with the model's input dimensions.
   * It supports only a single batch of images and resizes the input to match the model's required dimensions.
   *
   * @param images A vector of input images (cv::Mat) to preprocess.
   */
  void TrtLightnet::preprocess_gpu(const std::vector<cv::Mat> &images) {
    // Check if the input image batch is empty
    if (images.empty()) {
      std::cerr << "Preprocess called with an empty image batch." << std::endl;
      return;
    }

    // Get batch size and input dimensions
    const auto batch_size = images.size();
    auto input_dims = trt_common_->getBindingDimensions(0);

    // Adjust the batch size in the input dimensions
    input_dims.d[0] = batch_size;
    trt_common_->setBindingDimensions(0, input_dims); // Update dimensions with the new batch size

    // Extract input dimensions
    const float inputH = static_cast<float>(input_dims.d[2]);
    const float inputW = static_cast<float>(input_dims.d[3]);
    const float inputC = static_cast<float>(input_dims.d[1]);

    // Normalization factor
    float norm = 1.0f / 255.0f;

    // Get the dimensions of the first image (assuming all images in the batch are the same size)
    int w = images[0].cols;
    int h = images[0].rows;
    int src_size = 3 * w * h; // Source size in bytes (3 channels: RGB)

    // Allocate pinned (host) and device memory for image data if not already allocated
    if (!h_img_) {
      CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_img_, sizeof(unsigned char) * src_size));
      CHECK_CUDA_ERROR(cudaMalloc((void **)&d_img_, sizeof(unsigned char) * src_size));
    }

    // Copy image data to pinned memory
    memcpy(h_img_, images[0].data, src_size * sizeof(unsigned char));

    // Copy image data from pinned memory to device memory asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_img_, h_img_, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice, *stream_));

    // Perform GPU-based blob creation (resize and normalize the image)
    blobFromImageGpu(
		     (float*)(input_d_.get()), // Pointer to device input memory
		     d_img_,                  // Pointer to device source image data
		     inputW,                  // Target width
		     inputH,                  // Target height
		     inputC,                  // Number of channels
		     w,                       // Original width
		     h,                       // Original height
		     3,                       // Number of channels (RGB)
		     norm,                    // Normalization factor
		     *stream_                 // CUDA stream
		     );
  }

  /**
   * Prepares the inference by ensuring the network is initialized and setting up the input tensor.
   * @return True if the network is already initialized; false otherwise.
   */
  bool TrtLightnet::doInference(void)
  {
    if (!trt_common_->isInitialized()) {
      return false;
    }
    return infer();
  }

  /**
   * Executes the inference operation by setting up the buffers, launching the network, and transferring the outputs.
   * @return Always returns true to indicate the inference operation was called.
   */
  bool TrtLightnet::infer(void)
  {
    std::vector<void *> buffers = {input_d_.get()};
    for (int i = 0; i < static_cast<int>(output_d_.size()); i++) {
      buffers.push_back(output_d_.at(i).get());
    }

    trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);
    // Retrieve output tensors from device buffers
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const auto output_size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
      CHECK_CUDA_ERROR(cudaMemcpyAsync(output_h_.at(i-1).get(), output_d_.at(i-1).get(), sizeof(float) * output_size, cudaMemcpyDeviceToHost, *stream_));
    }
    cudaStreamSynchronize(*stream_);
    return true;
  }

  /**
   * Prepares the inference by ensuring the network is initialized and setting up the input tensor.
   * @return True if the network is already initialized; false otherwise.
   */
  bool TrtLightnet::doInference(const int batchSize)
  {
    if (!trt_common_->isInitialized()) {
      return false;
    }
    return infer(batchSize);
  }

  /**
   * Executes the inference operation by setting up the buffers, launching the network, and transferring the outputs.
   * @return Always returns true to indicate the inference operation was called.
   */
  bool TrtLightnet::infer(const int batchSize)
  {
    std::vector<void *> buffers = {input_d_.get()};
    for (int i = 0; i < static_cast<int>(output_d_.size()); i++) {
      buffers.push_back(output_d_.at(i).get());
    }

    trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

    // Retrieve output tensors from device buffers
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      auto dims = trt_common_->getBindingDimensions(i);
      dims.d[0] = batchSize;
      auto output_size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
      output_size *= batchSize;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(output_h_.at(i-1).get(), output_d_.at(i-1).get(), sizeof(float) * output_size, cudaMemcpyDeviceToHost, *stream_));
    }
    cudaStreamSynchronize(*stream_);
    return true;
  }
  
  /**
   * Draws bounding boxes on an image based on detection results.
   * @param img The image to draw bounding boxes on.
   * @param bboxes The detected bounding boxes.
   * @param colormap The color map to use for different classes.
   * @param names The names of the detected classes.
   */
  void TrtLightnet::drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, std::vector<std::vector<int>> &colormap, std::vector<std::string> names)
  {
    for (const auto& bbi : bboxes) {
      int id = bbi.classId;
      std::stringstream stream;
      if (!names.empty() && names[id] == "none") continue;
      if (!names.empty()) {
	stream << std::fixed << std::setprecision(2) << names[id] << "  " << bbi.prob;
      } else {
	stream << std::fixed << std::setprecision(2) << "id:" << id << "  score:" << bbi.prob;
      }
      
      cv::Scalar color = colormap.empty() ? cv::Scalar(255, 0, 0) : cv::Scalar(colormap[id][2], colormap[id][1], colormap[id][0]);
      if (bbi.isHierarchical) {
	std::string c_name;
	if (bbi.subClassId == 0) {
	  color = cv::Scalar(0, 255, TLR_GREEN);
	} else if (bbi.subClassId == TLR_YELLOW) {
	  color = cv::Scalar(0, 255, 255);
	} else {
	  color = cv::Scalar(0, 0, 255);
	}
	if (!names.empty() && names[id] == "arrow") {
	  float sin = bbi.sin;
	  float cos = bbi.cos;
	  float deg = atan2(sin, cos) * 180.0 / M_PI;
	  std::stringstream stream_TLR;
	  stream_TLR <<  std::fixed << std::setprecision(2) << int(deg);
	  int xlen = bbi.box.x2 - bbi.box.x1;
	  int ylen = bbi.box.y2 - bbi.box.y1;	  
	  int xcenter = (bbi.box.x2 + bbi.box.x1)/2;
	  int ycenter = (bbi.box.y2 + bbi.box.y1)/2;	  
	  int xr  = xcenter+(int)(sin * ylen/5);
	  int yr  = ycenter-(int)(cos * ylen/5);
	  if (xlen > 12) {
	    if (bbi.subClassId == 0) {
	      int y_offset = 0;
	      arrow(img, cv::Point{xcenter, ycenter + y_offset}, cv::Point{xr, yr + y_offset}, color, xlen/8);
	    }
	  }
	  cv::putText(img, stream_TLR.str(), cv::Point(bbi.box.x1, bbi.box.y2 + 24), 1, 1.0, color, 1);	  
	}
      }
      if (bbi.keypoint.size() == 0) {
	cv::rectangle(img, cv::Point(bbi.box.x1, bbi.box.y1), cv::Point(bbi.box.x2, bbi.box.y2), color, 4);
      }
      cv::putText(img, stream.str(), cv::Point(bbi.box.x1, bbi.box.y1 - 5), 0, 0.5, color, 1);
      if (bbi.keypoint.size() > 0) {
	auto keypoint = bbi.keypoint;
	drawKeypoint(img, keypoint, color);
      }
    }
  }

  /**
   * Extracts bounding box information from the network's output.
   * 
   * @param imageH The height of the input image.
   * @param imageW The width of the input image.
   */
  void TrtLightnet::makeBbox(const int imageH, const int imageW)
  {
    bbox_.clear();
    const auto inputDims = trt_common_->getBindingDimensions(0);
    int inputW = inputDims.d[3];
    int inputH = inputDims.d[2];
    // Channel size formula to identify relevant tensor outputs for bounding boxes.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    int tlr_size = (4 + 1 + num_class_ + 3 + 2) * num_anchor_;

    int detection_count = 0;
    for (int i = 0; i < trt_common_->getNbBindings(); i++) {
      if (trt_common_->bindingIsInput(i)) {
	continue;
      }
      const auto dims = trt_common_->getBindingDimensions(i);
      int gridW = dims.d[3];
      int gridH = dims.d[2];
      int chan = dims.d[1];

      if (chan_size == chan) { // Filtering out the tensors that match the channel size for detections.
	std::vector<BBoxInfo> b = decodeTensor(0, imageH, imageW, inputH, inputW, &(anchors_[num_anchor_ * (detection_count) * 2]), num_anchor_, output_h_.at(i-1).get(), gridW, gridH);
	bbox_.insert(bbox_.end(), b.begin(), b.end());
	detection_count++;
      } else if (tlr_size == chan) {
	//Decode TLR Tensor
	std::vector<BBoxInfo> b = decodeTLRTensor(0, imageH, imageW, inputH, inputW, &(anchors_[num_anchor_ * (detection_count) * 2]), num_anchor_, output_h_.at(i-1).get(), gridW, gridH);
	bbox_.insert(bbox_.end(), b.begin(), b.end());
	detection_count++;
      }
    }
    bbox_ = nonMaximumSuppression(nms_threshold_, bbox_); // Apply NMS and return the filtered bounding boxes.
    //bbox_ = nmsAllClasses(nms_threshold_, bbox_, num_class_); // Apply NMS and return the filtered bounding boxes.
  }

  /**
   * Clears the detected bounding boxes specifically from the subnet.
   */
  void TrtLightnet::clearSubnetBbox()
  {
    subnet_bbox_.clear();
  }

  /**
   * Appends a vector of detected bounding boxes to the existing list of bounding boxes from the subnet.
   * 
   * @param bb A vector of BBoxInfo that contains bounding boxes to be appended.
   */
  void TrtLightnet::appendSubnetBbox(std::vector<BBoxInfo> bb)    
  {
    subnet_bbox_.insert(subnet_bbox_.end(), bb.begin(), bb.end());
  }

  /**
   * Apply NMS for subnet BBox
   */
  void TrtLightnet::doNonMaximumSuppressionForSubnetBbox()
  {
    subnet_bbox_ = nonMaximumSuppression(nms_threshold_, subnet_bbox_);
  }

  /**
   * Returns the list of bounding boxes detected by the subnet.
   * 
   * @return A vector of BBoxInfo containing the bounding boxes detected by the subnet.
   */
  std::vector<BBoxInfo> TrtLightnet::getSubnetBbox()
  {
    return subnet_bbox_;
  }

  /**
   * Returns the list of bounding boxes detected by the engine.
   * 
   * @return A vector of BBoxInfo containing the bounding boxes detected by the engine.
   */
  std::vector<BBoxInfo> TrtLightnet::getBbox()
  {
    return bbox_;
  }

  std::vector<BBoxInfo> TrtLightnet::getBbox(const int imageH, const int imageW, const int batchIndex)
  {
    bbox_.clear();
    const auto inputDims = trt_common_->getBindingDimensions(0);
    int inputW = inputDims.d[3];
    int inputH = inputDims.d[2];
    // Channel size formula to identify relevant tensor outputs for bounding boxes.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    int tlr_size = (4 + 1 + num_class_ + 3 + 2) * num_anchor_;

    int detection_count = 0;
    for (int i = 0; i < trt_common_->getNbBindings(); i++) {
      if (trt_common_->bindingIsInput(i)) {
	continue;
      }
      const auto dims = trt_common_->getBindingDimensions(i);
      int gridW = dims.d[3];
      int gridH = dims.d[2];
      int chan = dims.d[1];
      float *p_output = &((output_h_.at(i-1).get())[batchIndex * (gridW * gridH * chan)]);
      if (chan_size == chan) { // Filtering out the tensors that match the channel size for detections.
	std::vector<BBoxInfo> b = decodeTensor(0, imageH, imageW, inputH, inputW, &(anchors_[num_anchor_ * (detection_count) * 2]), num_anchor_, p_output, gridW, gridH);
	bbox_.insert(bbox_.end(), b.begin(), b.end());
	detection_count++;
      } else if (tlr_size == chan) {
	//Decode TLR Tensor
	std::vector<BBoxInfo> b = decodeTLRTensor(0, imageH, imageW, inputH, inputW, &(anchors_[num_anchor_ * (detection_count) * 2]), num_anchor_, p_output, gridW, gridH);
	bbox_.insert(bbox_.end(), b.begin(), b.end());
	detection_count++;
      }
    }

    return nonMaximumSuppression(nms_threshold_, bbox_); // Apply NMS and return the filtered bounding boxes.
  }    
  
  
  std::vector<std::string> TrtLightnet::getNames()
  {
    return names_;
  }

  int TrtLightnet::getBatchSize()
  {
    return batch_size_;
  }  

  void TrtLightnet::setNames(std::vector<std::string> names)
  {
    names_ = names;
  }

  void TrtLightnet::setDetectionColormap(std::vector<std::vector<int>> &colormap)
  {
    colormap_ = colormap;
  }

  std::vector<std::vector<int>> TrtLightnet::getDetectionColormap(void)
  {
    return colormap_;
  }

  /**
   * Computes the area of a single bounding box.
   *
   * @param bbox The bounding box for which the area is to be calculated.
   * @return The area of the bounding box. Returns 0 if the dimensions are invalid.
   */
  float TrtLightnet::calculateBBoxArea(const BBox& bbox) {
    float width = std::max(0.0f, bbox.x2 - bbox.x1);
    float height = std::max(0.0f, bbox.y2 - bbox.y1);
    return width * height;
  }

  /**
   * Computes the overlapping area between two bounding boxes.
   *
   * @param bbox1 The first bounding box.
   * @param bbox2 The second bounding box.
   * @return The area of the overlapping region between the two bounding boxes. Returns 0 if there is no overlap.
   */
  float TrtLightnet::calculateOverlapArea(const BBox& bbox1, const BBox& bbox2) {
    float overlapX1 = std::max(bbox1.x1, bbox2.x1);
    float overlapY1 = std::max(bbox1.y1, bbox2.y1);
    float overlapX2 = std::min(bbox1.x2, bbox2.x2);
    float overlapY2 = std::min(bbox1.y2, bbox2.y2);

    // Calculate overlap width and height
    float overlapWidth = std::max(0.0f, overlapX2 - overlapX1);
    float overlapHeight = std::max(0.0f, overlapY2 - overlapY1);

    // Return the area of the overlapping region
    return overlapWidth * overlapHeight;
  }

  /**
   * Computes the aspect ratio of a bounding box.
   *
   * @param bbox The bounding box for which the aspect ratio is to be calculated.
   * @return The aspect ratio (width / height). Returns 0 if the height is zero.
   */  
  float TrtLightnet::calculateAspectRatio(const BBox& bbox) {
    float width = std::max(0.0f, bbox.x2 - bbox.x1);
    float height = std::max(0.0f, bbox.y2 - bbox.y1);
    return (height > 0.0f) ? (width / height) : 0.0f; // Avoid division by zero
  }
  
  /**
   * Checks if one bounding box is completely contained within another bounding box.
   *
   * @param bbox1 The bounding box to check for containment.
   * @param bbox2 The bounding box that may contain bbox1.
   * @return True if bbox1 is contained within bbox2, false otherwise.
   */
  bool TrtLightnet::isContained(const BBox& bbox1, const BBox& bbox2)
  {
    return (bbox1.x1 >= bbox2.x1 && bbox1.y1 >= bbox2.y1 &&
	    bbox1.x2 <= bbox2.x2 && bbox1.y2 <= bbox2.y2);
  }

  /**
   * Removes bounding boxes that are contained within others.
   *
   * @param names A list of class names corresponding to bounding box class IDs.
   * @param target A list of target class names to filter for removal based on containment.
   */
  void TrtLightnet::removeContainedBBoxes(std::vector<std::string> &names, std::vector<std::string> &target)
  {
    std::vector<BBoxInfo> result;

    for (size_t i = 0; i < bbox_.size(); ++i) {
      bool isContainedFlag = false;
      bool flg = false;
      for (auto &t : target) {
	if (t == names[bbox_[i].classId]) {
	  flg = true;
	}
      }
      if (!flg) {
	result.push_back(bbox_[i]);
	continue;
      }
      for (size_t j = 0; j < bbox_.size(); ++j) {
	bool flg = false;
	for (auto &t : target) {
	  if (t == names[bbox_[j].classId]) {
	    flg = true;
	  }
	}
	if (!flg) {	  
	  continue;
	}
	if (i != j) {	
	  float overlap =  calculateOverlapArea(bbox_[i].box, bbox_[j].box);
	  float area = calculateBBoxArea(bbox_[i].box);
	  if ((overlap / area) > 0.85) {
	    isContainedFlag = true;
	    break;
	  }
	}
      }

      if (!isContainedFlag) {
	result.push_back(bbox_[i]);
      }
    }
      bbox_ = result;
  }

  /**
   * Removes bounding boxes whose aspect ratio exceeds a specified threshold.
   *
   * @param names A list of class names corresponding to bounding box class IDs.
   * @param target A list of target class names to filter for removal based on aspect ratio.
   * @param targetAspectRatio The maximum allowable aspect ratio. Bounding boxes with aspect ratios exceeding this value are removed.
   */
  void TrtLightnet::removeAspectRatioBoxes(std::vector<std::string> &names, std::vector<std::string> &target, float targetAspectRatio) {
    std::vector<BBoxInfo> result;

    for (const auto& bboxInfo : bbox_) {
      bool flg = false;
      for (auto &t : target) {
	if (t == names[bboxInfo.classId]) {
	  flg = true;
	}
      }
      if (!flg) {
	result.push_back(bboxInfo);	
	continue;
      }      
      float aspectRatio = calculateAspectRatio(bboxInfo.box);
      if (aspectRatio > targetAspectRatio) {
	result.push_back(bboxInfo);
      }
    }

    bbox_ =  result;
  }

  /**
   * @brief Processes a BEV (Bird's Eye View) map to identify and visualize free space.
   *
   * This function processes a BEV map (`bevmap_`) to create a free space map (`freespace_`) by:
   * - Converting specific pixel values in the BEV map to a binary image.
   * - Smoothing the binary image and detecting contours.
   * - Identifying the largest contour and visualizing it along with other contours.
   * - Displaying the resulting free space and binary image.
   *
   * @param road_ids A vector of road IDs (currently not used within the function).
   */
  void TrtLightnet::makeFreespace(std::vector<int> road_ids) {
    // Initialize the free space map and a grayscale version of it.
    freespace_ = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);  // Free space map (RGB).
    cv::Mat grayscale = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC1);  // Grayscale map for processing.

    // Iterate through the BEV map to identify specific pixels representing the road surface.
    for (int x = 0; x < GRID_W; x++) {
      for (int y = GRID_H - 1; y >= 0; y--) {
	cv::Vec3b b = bevmap_.at<cv::Vec3b>(y, x);
	if (b[0] == 128 && b[1] == 64 && b[2] == 128) {  // Road pixel identified.
	  grayscale.at<uchar>(y, x) = 255;  // Mark road pixels in grayscale.
	}
      }
    }

    // Smooth the grayscale image using a Gaussian blur.
    cv::Mat smoothedImage;
    cv::GaussianBlur(grayscale, smoothedImage, cv::Size(7, 7), 0);

    // Convert the smoothed image to a binary image using a threshold.
    cv::Mat binaryImage;
    cv::threshold(smoothedImage, binaryImage, 0, 255, cv::THRESH_BINARY);

    // Display the binary image for debugging purposes.
    cv::imshow("binaryImage", binaryImage);

    // Find contours in the binary image.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Identify the largest contour by area.
    int largestContourIndex = -1;
    double largestArea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
      double area = cv::contourArea(contours[i]);
      if (area > largestArea) {
	largestArea = area;
	largestContourIndex = static_cast<int>(i);
      }
    }

    // Draw all outer contours in green (indicating free space).
    for (size_t i = 0; i < contours.size(); i++) {
      if (hierarchy[i][3] == -1) {  // Only draw outermost contours.
	cv::drawContours(freespace_, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), cv::FILLED);
      }
    }

    // Highlight the largest contour in green if found.
    if (largestContourIndex != -1) {
      cv::drawContours(freespace_, contours, largestContourIndex, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // Draw other contours (not the largest) in red.
    for (size_t i = 0; i < contours.size(); i++) {
      if (i != (size_t)largestContourIndex) {
	cv::drawContours(freespace_, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA, hierarchy);
      }
    }

    // Display the free space map for debugging purposes.
    cv::imshow("freespace", freespace_);
  }
  
  /**
   * @brief Generates and returns the occupancy grid for the BEV (Bird's Eye View) map.
   * 
   * The occupancy grid is used to represent road, obstacles, and other entities
   * in a bird's eye view format. This function initializes and updates the 
   * occupancy grid based on the provided BEV map and bounding box information.
   *
   * @return cv::Mat The occupancy grid as a CV Mat object.
   */
  cv::Mat TrtLightnet::getOccupancyGrid(void) {
    return occupancy_;
  }

  /**
   * @brief Creates the occupancy grid based on road and obstacle data.
   *
   * This function processes the BEV map (`bevmap_`) and bounding box data (`bbox_`) to:
   * - Identify and mark road areas in the occupancy grid.
   * - Map bounding boxes to 3D coordinates and project them onto the BEV map.
   * - Render obstacles and roads in distinct colors based on a provided colormap.
   * 
   * @param road_ids A vector of road IDs (currently unused in the function).
   * @param im_w Image width of the input data.
   * @param im_h Image height of the input data.
   * @param calibdata Camera calibration data used for coordinate transformations.
   * @param colormap A 2D vector specifying the color for each class of objects.
   * @param names A list of class names corresponding to the object classes.
   * @param target A list of target class names to process.
   */
  void TrtLightnet::makeOccupancyGrid(std::vector<int> road_ids, const int im_w, const int im_h, const Calibration calibdata,std::vector<std::vector<int>> &colormap, std::vector<std::string> names, std::vector<std::string> &target)
  {
    int block_size = 4;
    
    occupancy_ = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);  // Initialize the BEV map with zeros.
    for (int x = block_size; x < GRID_W-block_size; x+=block_size) {
      for (int y =  GRID_H - 1 - block_size; y >=block_size; y-=block_size) {
	bool flg_road = false;
	for (int xb = -block_size; xb <= block_size; xb++) {
	  for (int yb = -block_size; yb <= block_size; yb++) {	
	    cv::Vec3b b = bevmap_.at<cv::Vec3b>(y-yb, x-xb);
	    flg_road = (b[0] == 128 && b[1] == 64 && b[2] == 128) ? true : false;	      
	  }
	}
	if (flg_road) {
	  for (int xb = -block_size; xb <= block_size; xb++) {
	    for (int yb = -block_size; yb <= block_size; yb++) {
	      cv::Vec3b &og = occupancy_.at<cv::Vec3b>(y-yb, x-xb);
	      og[0] = 128;
	      og[1] = 64;
	      og[2] = 128;	  	      
	    }
	  }
	}
      }
    }

    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w = static_cast<float>(outputW) / static_cast<float>(im_w);
	  float scale_h = static_cast<float>(outputH) / static_cast<float>(im_h);
	  float *buf = static_cast<float *>(output_h_.at(i - 1).get());
	  float gran_h = static_cast<float>(GRID_H) / calibdata.max_distance;

	  // Process each bounding box.
	  for (const auto &b : bbox_) {
	    bool flg = false;
	    if ((names[b.classId] == "UNKNOWN") ||
		(names[b.classId] == "CAR") ||
		(names[b.classId] == "TRUCK") ||
		(names[b.classId] == "BUS") ||
		(names[b.classId] == "BICYCLE") ||
		(names[b.classId] == "MOTORBIKE") ||
		(names[b.classId] == "PEDESTRIAN")) {		
	      flg = true;
	    }

	    if (!flg) {
	      continue;
	    }

	    // Calculate coordinates for the BEV map.
	    int xlen = (b.box.x2 - b.box.x1) ;
	    int offset = (int)(xlen * 0.25);
	    for (int w = b.box.x1+offset; w <= b.box.x2-offset; w++) {
	      int cx = w;
	      int cy = (b.box.y2+b.box.y1)/2;
	      if (cx < 0 || cx >= im_w) continue;
	      if (cy < 0 || cy >= im_h) continue;	      
	      int stride = outputW * static_cast<int>(cy * scale_h);
	      float distance = buf[stride + static_cast<int>(cx * scale_w)] * calibdata.max_distance;
	      distance = std::min(distance, calibdata.max_distance);

	      float src_x = static_cast<float>(cx);
	      float x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance;

	      if (x3d > 20.0) {
		continue;
	      }

	      x3d = (x3d + 20.0f) * GRID_W / 40.0f;
	      int x_bev = static_cast<int>(x3d);
	      int y_bev = static_cast<int>(GRID_H - static_cast<int>(distance * gran_h));
	      auto color = cv::Scalar(colormap[b.classId][2], colormap[b.classId][1], colormap[b.classId][0]);
	      x_bev = x_bev >= GRID_W ? GRID_W-1  : x_bev;
	      x_bev = x_bev < 0 ? 0  : x_bev;
	      y_bev = y_bev >= GRID_H ? GRID_H-1  : y_bev;
	      y_bev = y_bev < 0 ? 0  : y_bev;	      	      
	      // Plot a circle onto the BEV map.
	      for (int xb = -block_size/4; xb <= block_size/4; xb++) {
		for (int yb = -block_size/4; yb <= block_size/4; yb++) {		  
		  int xx = x_bev - xb;
		  int yy = y_bev - yb;
		  xx = xx >= GRID_W ? GRID_W-1  : xx;
		  xx = xx < 0 ? 0  : xx;
		  yy = yy >= GRID_H ? GRID_H-1  : yy;
		  yy = yy < 0 ? 0  : yy;	      	      		  
		  cv::Vec3b &og = occupancy_.at<cv::Vec3b>(yy, xx);
		  og[0] = color[0];
		  og[1] = color[1];
		  og[2] = color[2];		  
		}
	      }
	      
	    }
	  }
	}
      }
    }

    block_size = 8;
    for (int x = 0; x < GRID_W; x++) {
      for (int y =  GRID_H - 1 - block_size; y >=block_size; y-=block_size) {
	bool flg_road = false;
	cv::Vec3b b = bevmap_.at<cv::Vec3b>(y, x);
	flg_road = (b[0] == 128 && b[1] == 64 && b[2] == 128) ? true : false;	      
	if (flg_road) {
	  flg_road = false;
	  for (int yb = y - block_size; yb < y; yb++) {
	      cv::Vec3b &og = occupancy_.at<cv::Vec3b>(yb, x);
	      if (!flg_road) {
		flg_road = (og[0] == 128 && og[1] == 64 && og[2] == 128) ? true : false;
	      }
	      if (flg_road) {
		if (og[0] == 0 && og[1] == 0 && og[2] == 0) {
		  og[0] = 128;
		  og[1] = 64;
		  og[2] = 128;
		} else if (!((og[0] == 128 && og[1] == 64 && og[2] == 128) || (og[0] == 0 && og[1] == 0 && og[2] == 0))) {
		  flg_road = false;
		}
	      }
	  }
	}
      }
    }    
    cv::imshow("occupancy_grid", occupancy_);    
  }
  
  /**
   * @brief Smoothes the depth map using road segmentation information.
   *
   * This function processes depth map tensors to refine the depth values 
   * by averaging distances over regions identified as roads or similar surfaces 
   * in the segmentation map. The smoothed depth values are then applied back to the depth map.
   */
  void TrtLightnet::smoothDepthmapFromRoadSegmentation(std::vector<int> road_ids) {
    // Formula to identify tensors unrelated to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Iterate over all tensor bindings to locate depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify depth map tensors by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Verify tensor type is FLOAT and name contains "lgx".
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float *buf = static_cast<float *>(output_h_.at(i - 1).get());

	  // Find the segmentation tensor for road segmentation labels.
	  for (int j = 1; j < trt_common_->getNbBindings(); j++) {
	    std::string seg_name = trt_common_->getIOTensorName(j);

	    if (contain(seg_name, "argmax")) {
	      const auto seg_dims = trt_common_->getBindingDimensions(j);
	      const int segWidth = seg_dims.d[3];
	      const int segHeight = seg_dims.d[2];
	      const float scaleW = static_cast<float>(outputW) / segWidth;
	      const float scaleH = static_cast<float>(outputH) / segHeight;
	      const int *argmax = (int *)(output_h_.at(j - 1).get());
	      // Iterate over the height of the depth map.
	      for (int y = 0; y < outputH; y++) {
		float sum_dist = 0.0;
		int count = 0;

		// First pass: Calculate average depth for road segments.
		for (int x = 0; x < outputW; x++) {
		  const int id = argmax[static_cast<int>(x * scaleW) + segWidth * static_cast<int>(y * scaleH)];
		  bool is_road = false;
		  for (const int& value : road_ids) {
		    if (value == id) {
		      is_road = true;
		    }
		  }
		  if (is_road) { // Road-related segmentation IDs		  
		    count++;
		    sum_dist += buf[x + outputW * y];
		  }
		}

		// Avoid division by zero in case no road segments are detected.
		if (count > 0) {
		  sum_dist /= count;
		}

		// Second pass: Apply smoothed distance to road segment pixels.
		for (int x = 0; x < outputW; x++) {
		  const int id = argmax[static_cast<int>(x * scaleW) + segWidth * static_cast<int>(y * scaleH)];
		  bool is_road = false;
		  for (const int& value : road_ids) {
		    if (value == id) {
		      is_road = true;
		    }
		  }		  
		  if (is_road) { // Road-related segmentation IDs		  		  
		    buf[x + outputW * y] = sum_dist;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  /**
   * @brief Smooths a depth map using road segmentation on the GPU.
   *
   * This function identifies relevant tensors for depth maps and segmentation,
   * then applies GPU-based smoothing of the depth map using road segmentation labels.
   *
   * @param road_ids A vector of road segmentation label IDs to identify road regions.
   */
  void TrtLightnet::smoothDepthmapFromRoadSegmentationGpu(std::vector<int> road_ids) {
    // Formula to identify tensors unrelated to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Iterate over all tensor bindings to locate depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify depth map tensors by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Verify tensor type is FLOAT and name contains "lgx".
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {

	  // Find the segmentation tensor for road segmentation labels.
	  for (int j = 1; j < trt_common_->getNbBindings(); j++) {
	    std::string seg_name = trt_common_->getIOTensorName(j);

	    if (contain(seg_name, "argmax")) {
	      const auto seg_dims = trt_common_->getBindingDimensions(j);
	      const int segWidth = seg_dims.d[3];
	      const int segHeight = seg_dims.d[2];
	      int numRoadIds = static_cast<int>(road_ids.size());

	      // Allocate device memory for road IDs if not already allocated.
	      if (!d_road_ids_) {
		d_road_ids_ = cuda_utils::make_unique<int[]>(numRoadIds);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_road_ids_.get(), road_ids.data(), sizeof(int) * numRoadIds, cudaMemcpyHostToDevice, *stream_));
	      }

	      // Perform GPU-based smoothing of the depth map.
	      smoothDepthmapGpu(
				(float *)output_d_.at(i - 1).get(),
				(const int *)output_d_.at(j - 1).get(),
				outputW, outputH,
				segWidth, segHeight,
				(int *)d_road_ids_.get(),
				numRoadIds,
				                            *stream_
				);
	    }
	  }
	}
      }
    }
  }
  
  /**
   * @brief Generates depth maps from the output tensors of the TensorRT model.
   *
   * This function processes the output tensors from the TensorRT model to generate depth maps.
   * The depth maps are stored in the `depthmaps_` vector. The format of the depth maps can either 
   * be grayscale (CV_8UC1) or a 3-channel color map (CV_8UC3) using the "magma" colormap.
   *
   * @param depth_format Specifies the format of the depth map. If "magma" is specified, the depth 
   * maps will be in color; otherwise, they will be in grayscale.
   */
  void TrtLightnet::makeDepthmap(std::string &depth_format)
  {
    depthmaps_.clear();  // Clear any existing depth maps.

    // Formula to identify output tensors not related to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify depth map tensors by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {

	  cv::Mat depthmap;

	  // Initialize depth map with appropriate format (color or grayscale).
	  if (depth_format == "magma" || depth_format == "jet") {
	    depthmap = cv::Mat::zeros(outputH, outputW, CV_8UC3);
	  } else {
	    depthmap = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	  }
	  
	  float *buf = (float *)output_h_.at(i-1).get();

	  // Populate the depth map with the processed depth values.
	  for (int y = 0; y < outputH; y++) {
	    int stride = outputW * y;

	    for (int x = 0; x < outputW; x++) {
	      float rel = 1.0f - buf[stride + x];  // Invert the depth value.
	      int value = static_cast<int>(rel * 255);  // Scale to 8-bit value.

	      if (depth_format == "magma") {
		// Apply the "magma" colormap.
		depthmap.at<cv::Vec3b>(y, x)[0] = magma_colormap[255 - value][2];
		depthmap.at<cv::Vec3b>(y, x)[1] = magma_colormap[255 - value][1];
		depthmap.at<cv::Vec3b>(y, x)[2] = magma_colormap[255 - value][0];
	      } else if (depth_format == "jet") {
		// Apply the "magma" colormap.
		depthmap.at<cv::Vec3b>(y, x)[0] = jet_colormap[255 - value][2];
		depthmap.at<cv::Vec3b>(y, x)[1] = jet_colormap[255 - value][1];
		depthmap.at<cv::Vec3b>(y, x)[2] = jet_colormap[255 - value][0];
	      } else {
		// Use grayscale format.
		depthmap.at<unsigned char>(y, x) = value;
	      }
	    }
	  }
	  // Store the generated depth map.
	  depthmaps_.push_back(depthmap);
	}
      }
    }
  }

  /**
   * @brief Generates depth maps using GPU-based processing.
   *
   * This function iterates through tensor bindings to locate depth map tensors,
   * processes them using a GPU, and stores the resulting depth maps in a vector.
   */
  void TrtLightnet::makeDepthmapGpu(void) {
    depthmaps_.clear();  // Clear any existing depth maps.

    // Formula to identify output tensors not related to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify depth map tensors by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  cv::Mat depthmap;
	  // Initialize depth map with appropriate format (color or grayscale).
	  depthmap = cv::Mat::zeros(outputH, outputW, CV_8UC3);
	  int src_size = outputH * outputW * 3;

	  // Generate depth map on the GPU.
	  generateDepthmapGpu(
			      (unsigned char *)d_depthmap_.get(),
			      (const float *)output_d_.at(i - 1).get(),
			      d_depth_colormap_.get(),
			      outputW, outputH,
			      *stream_
			      );

	  // Copy the generated depth map from device to host memory.
	  CHECK_CUDA_ERROR(cudaMemcpyAsync(
					   depthmap.data,
					   d_depthmap_.get(),
					   sizeof(unsigned char) * src_size,
					   cudaMemcpyDeviceToHost,
					   *stream_
					   ));

	  // Store the depth map in the vector.
	  depthmaps_.push_back(depthmap);
	}
      }
    }
  }
    
  /**
   * @brief Generates a bird's-eye view map (BEV map) by back-projecting a depth map onto a 2D grid.
   *
   * This function creates a back-projected bird's-eye view map using the input depth map data from the 
   * TensorRT model. It processes the output tensors of the model to compute the 3D coordinates of 
   * points and maps them onto a 2D grid. The resulting BEV map is stored in the `bevmap_` member 
   * variable as a CV_8UC3 Mat object.
   *
   * @param im_w Width of the input image used for scaling the back-projected points.
   * @param im_h Height of the input image used for scaling the back-projected points.
   * @param calibdata Calibration data containing camera intrinsic parameters and maximum distance.
   */
  void TrtLightnet::makeBackProjection(const int im_w, const int im_h, const Calibration calibdata)
  {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    bevmap_ = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);  // Initialize the BEV map with zeros.

    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];
      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w = (float)(im_w) / (float)outputW;
	  float *buf = (float *)output_h_.at(i-1).get();
	  float gran_h = (float)GRID_H / calibdata.max_distance;
	  int mask_w = masks_[0].cols;
	  int mask_h = masks_[0].rows;
	  float mask_scale_w = mask_w / (float)outputW;
	  float mask_scale_h = mask_h / (float)outputH;

	  // Process the depth map within a specific vertical range (from 4/5 to 1/3 of the height).
	  //for (int y = outputH * 4/5; y >= outputH * 1/3; y--) {
	  for (int y = outputH-1; y >= 0; y--) {	  
	    int stride = outputW * y;

	    for (int x = 0; x < outputW; x++) {
	      float distance = buf[stride + x] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;

	      // Compute 3D coordinates from the 2D image coordinates.
	      float src_x = x * scale_w;
	      float x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance;


	      // Skip points that are too far in the x-direction or below the ground plane.
	      if (x3d > 20.0) continue;
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      if (x3d > GRID_H || x3d < 0.0) continue;

	      // Transform to BEV coordinates
	      // \( x_{\text{bev}} = \frac{X}{\text{resolution}} + \frac{\text{BEV width}}{2} \)
	      int x_bev = static_cast<int>(static_cast<int>(x3d));
	      // \( y_{\text{bev}} = \frac{\text{BEV height}}{2} - \frac{Y}{\text{resolution}} \)
	      //int y_bev = static_cast<int>(bevCenterY - Y / BEV_RESOLUTION);
	      int y_bev = static_cast<int>(GRID_H - static_cast<int>(distance * gran_h));
	      // Map the 3D point onto the 2D BEV map, applying the mask if available.
	      if (masks_.size() > 0) {
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[0] = masks_[0].at<cv::Vec3b>(y * mask_scale_h, x * mask_scale_w)[0];
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[1] = masks_[0].at<cv::Vec3b>(y * mask_scale_h, x * mask_scale_w)[1];
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[2] = masks_[0].at<cv::Vec3b>(y * mask_scale_h, x * mask_scale_w)[2];
	      } else {
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[0] = 255;
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[1] = 255;
		bevmap_.at<cv::Vec3b>(y_bev, x_bev)[2] = 255;
	      }
	    }
	  }
	}
      }
    }
  }

  /**
   * @brief Creates a bird's-eye view (BEV) map from depth maps using GPU-based back projection.
   *
   * This function processes depth map tensors and applies GPU-based back projection
   * to generate a BEV map using calibration data and masks.
   *
   * @param im_w Image width of the original input.
   * @param im_h Image height of the original input.
   * @param calibdata Calibration data required for back projection.
   */
  void TrtLightnet::makeBackProjectionGpu(const int im_w, const int im_h, const Calibration calibdata, int padding)
  {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    bevmap_ = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);  // Initialize the BEV map with zeros.
    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w = static_cast<float>(im_w) / static_cast<float>(outputW);
	  float scale_h = static_cast<float>(im_h) / static_cast<float>(outputH);
	  int mask_w = masks_[0].cols;
	  int mask_h = masks_[0].rows;
	  int bev_size = GRID_H * GRID_W * 3;
	  int mask_size = mask_h * mask_w * 3;

	  // Allocate device memory for BEV map and mask if not already allocated.
	  if (!d_bevmap_) {
	    d_bevmap_ = cuda_utils::make_unique<unsigned char[]>(bev_size);
	  }
	  if (!d_mask_) {
	    d_mask_ = cuda_utils::make_unique<unsigned char[]>(mask_size);
	  }

	  // Copy mask data to device memory.
	  CHECK_CUDA_ERROR(cudaMemcpyAsync(
					   d_mask_.get(),
					   masks_[0].data,
					   sizeof(unsigned char) * mask_size,
					   cudaMemcpyHostToDevice,
					   *stream_
					   ));

	  // Initialize the BEV map memory on the device.
	  cudaMemset(d_bevmap_.get(), 0, bev_size * sizeof(unsigned char));

	  // Perform GPU-based back projection.
	  getBackProjectionGpu(
			       (const float *)output_d_.at(i - 1).get(),
			       outputW, outputH,
			       scale_w, scale_h,
			       calibdata,
			       mask_w, mask_h,
			       (unsigned char *)d_mask_.get(),
			       masks_[0].step,
			       d_bevmap_.get(),
			       bevmap_.step,
			       padding,
			       *stream_
			       );

	  // Copy the resulting BEV map from device to host memory.
	  CHECK_CUDA_ERROR(cudaMemcpyAsync(
					   bevmap_.data,
					   d_bevmap_.get(),
					   sizeof(unsigned char) * bev_size,
					   cudaMemcpyDeviceToHost,
					   *stream_
					   ));
	}
      }
    }
  }

  /**
   * @brief Performs a GPU-based back projection without densifying the BEV map.
   *
   * This function processes a depth map tensor to create a sparse BEV (Bird's Eye View) map
   * using GPU-based back projection. The output map is visualized as `sparsemap_`.
   *
   * @param im_w The width of the input image.
   * @param im_h The height of the input image.
   * @param calibdata Calibration data containing camera parameters and configurations.
   */
  void TrtLightnet::makeBackProjectionGpuWithoutDensify(const int im_w, const int im_h, const Calibration calibdata)
  {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    sparsemap_ = cv::Mat::zeros(GRID_H, GRID_W, CV_8UC3);  // Initialize the BEV map with zeros.
    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w = static_cast<float>(im_w) / static_cast<float>(outputW);
	  float scale_h = static_cast<float>(im_h) / static_cast<float>(outputH);
	  int mask_w = masks_[0].cols;
	  int mask_h = masks_[0].rows;
	  int bev_size = GRID_H * GRID_W * 3;
	  int mask_size = mask_h * mask_w * 3;

	  // Allocate device memory for BEV map and mask if not already allocated.
	  if (!d_bevmap_) {
	    d_bevmap_ = cuda_utils::make_unique<unsigned char[]>(bev_size);
	  }
	  if (!d_mask_) {
	    d_mask_ = cuda_utils::make_unique<unsigned char[]>(mask_size);
	  }

	  // Copy mask data to device memory.
	  CHECK_CUDA_ERROR(cudaMemcpyAsync(
					   d_mask_.get(),
					   masks_[0].data,
					   sizeof(unsigned char) * mask_size,
					   cudaMemcpyHostToDevice,
					   *stream_
					   ));

	  // Initialize the BEV map memory on the device.
	  cudaMemset(d_bevmap_.get(), 0, bev_size * sizeof(unsigned char));

	  // Perform GPU-based back projection.
	  getBackProjectionGpu(
			       (const float *)output_d_.at(i - 1).get(),
			       outputW, outputH,
			       scale_w, scale_h,
			       calibdata,
			       mask_w, mask_h,
			       (unsigned char *)d_mask_.get(),
			       masks_[0].step,
			       d_bevmap_.get(),
			       sparsemap_.step,
			       0,
			       *stream_
			       );

	  // Copy the resulting BEV map from device to host memory.
	  CHECK_CUDA_ERROR(cudaMemcpyAsync(
					   sparsemap_.data,
					   d_bevmap_.get(),
					   sizeof(unsigned char) * bev_size,
					   cudaMemcpyDeviceToHost,
					   *stream_
					   ));
	}
      }
    }
    cv::imshow("sparse-bevmap", sparsemap_);    
  }  

  /**
   * @brief Creates a height map from the depth map data using 3D geometry transformations.
   *
   * This function generates a height map by converting depth data from a tensor to a color-coded
   * visualization, where the height values are represented in a jet colormap.
   *
   * @param im_w The width of the input image.
   * @param im_h The height of the input image.
   * @param calibdata The calibration data containing intrinsic camera parameters and max distance.
   * @return cv::Mat The generated height map as a color-coded image.
   */  
  cv::Mat TrtLightnet::makeHeightmap(const int im_w, const int im_h, const Calibration calibdata)
  {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    cv::Mat mask;

    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];
      const float max_height = 80.0;
      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  mask = cv::Mat::zeros(outputH, outputW, CV_8UC3);      
	  
	  float scale_w = (float)(im_w) / (float)outputW;
	  float *buf = (float *)output_h_.at(i-1).get();

	  // Process the depth map within a specific vertical range (from 4/5 to 1/3 of the height).
	  for (int y = outputH - 1; y >= 0; y--) {	  
	    int stride = outputW * y;

	    for (int x = 0; x < outputW; x++) {
	      float distance = buf[stride + x] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;

	      // Compute 3D coordinates from the 2D image coordinates.
	      float src_x = x * scale_w;
	      float y3d = ((src_x - calibdata.v0) / calibdata.fy) * distance;	      

	      y3d = y3d < 0.0 ? 0.0 : y3d;
	      y3d = y3d > max_height ? max_height : y3d;
	      int value = y3d / (float)max_height * 255;	      
	      mask.at<cv::Vec3b>(y, x)[0] = jet_colormap[value][0];
	      mask.at<cv::Vec3b>(y, x)[1] = jet_colormap[value][1];
	      mask.at<cv::Vec3b>(y, x)[2] = jet_colormap[value][2];	  
	    }
	  }
	}
      }
    }
    return mask;
  }

  /**
   * @brief Blends segmentation masks with an input image using GPU acceleration.
   *
   * This function applies segmentation masks to an input image using GPU-based 
   * processing. It supports resizing masks and blending them with the image using 
   * specified weights for alpha, beta, and gamma.
   *
   * @param image Input/output image to which the segmentation masks are applied.
   * @param alpha Weight of the original image in the blending process.
   * @param beta Weight of the segmentation mask in the blending process.
   * @param gamma Scalar added to each sum during blending.
   */  
  void TrtLightnet::blendSegmentationGpu(cv::Mat &image, float alpha, float beta, float gamma)
  {
    int mask_size = image.cols * image.rows * 3;
    if (!d_resized_mask_) {
      d_resized_mask_  = cuda_utils::make_unique<unsigned char[]>(mask_size);
    }
    if (!d_mask_) {
      d_mask_ = cuda_utils::make_unique<unsigned char[]>(mask_size);
    }    
    if (!d_img_) {
      CHECK_CUDA_ERROR(cudaMalloc((void **)&d_img_, sizeof(unsigned char) * mask_size));
    }
    memcpy(h_img_, image.data, mask_size * sizeof(unsigned char));
    // Copy image data from pinned memory to device memory asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_img_, h_img_, mask_size * sizeof(unsigned char), cudaMemcpyHostToDevice, *stream_));    
    for (const auto &mask : masks_) {
      if (masks_.size() == 1) {
	resizeNearestNeighborGpu(d_resized_mask_.get(), d_mask_.get(),
				 image.cols, image.rows, 3,
				 masks_[0].cols, masks_[0].rows, 3, *stream_);
      } else {     
	cv::Mat resized;
	cv::resize(mask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
	CHECK_CUDA_ERROR(cudaMemcpyAsync(
					 d_resized_mask_.get(),
					 resized.data,
					 sizeof(unsigned char) * mask_size,
					 cudaMemcpyHostToDevice,
					 *stream_
					 ));
      }      
      addWeightedGpu(d_img_, d_img_, d_resized_mask_.get(), alpha, beta, gamma, image.cols, image.rows, 3, *stream_);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(h_img_, d_img_, sizeof(unsigned char) * mask_size, cudaMemcpyDeviceToHost, *stream_));
      cudaStreamSynchronize(*stream_);
      memcpy(image.data, h_img_, mask_size * sizeof(unsigned char));       
    }
  }
  
  /**
   * Retrieves the generated BEV map.
   * 
   * @return BEV map.
   */    
  cv::Mat TrtLightnet::getBevMap(void)
  {
    return bevmap_;
  }

  /**
   * Retrieves the generated sparse BEV map.
   * 
   * @return sparse bev map.
   */      
  cv::Mat TrtLightnet::getSparseBevMap(void)
  {
    return sparsemap_;
  }  
  
  /**
   * Retrieves the generated depth maps.
   * 
   * @return A vector of cv::Mat, each representing a depth map for an input image.
   */  
  std::vector<cv::Mat> TrtLightnet::getDepthmap(void)
  {
    return depthmaps_;
  }

  /**
   * @brief Calculates the median distance from a region in a buffer.
   *
   * This function extracts distance values from a specified region in the buffer,
   * based on the bounding box coordinates, and computes the median distance.
   *
   * @param buf Pointer to the buffer containing distance data.
   * @param outputW Width of the output buffer (used for indexing).
   * @param scale_w Horizontal scaling factor for the buffer.
   * @param scale_h Vertical scaling factor for the buffer.
   * @param b Bounding box specifying the region of interest.
   * @return Median distance value from the specified region.
   */
  float TrtLightnet::get_meadian_dist(const float *buf, int outputW, float scale_w, float scale_h, const BBox b) {
    std::vector<float> dist;

    // Iterate over the region defined by the bounding box
    for (int y = b.y1; y < b.y2; y++) {
      for (int x = b.x1; x < b.x2; x++) {
	// Calculate the index in the buffer and fetch the distance value
	int stride = outputW * static_cast<int>(y * scale_h);
	dist.push_back(buf[stride + static_cast<int>(x * scale_w)]);
      }
    }

    // Ensure the distance vector is not empty before accessing the median
    if (dist.empty()) {
      return 0.0f; // Return 0 if no values are present to avoid undefined behavior
    }

    // Sort the distances to compute the median
    std::sort(dist.begin(), dist.end());

    // Calculate and return the median value
    return dist[dist.size() / 2];
  }

  /**
   * @brief Plots circles representing detected objects onto the BEV map.
   *
   * This function uses bounding box data and calibration parameters to determine
   * the 3D positions of objects and plots corresponding circles on the BEV map.
   *
   * @param im_w Width of the original input image.
   * @param im_h Height of the original input image.
   * @param calibdata Calibration data required for mapping coordinates.
   * @param names Vector of class names for detected objects.
   * @param target Vector of target object classes to exclude from plotting.
   */
  void TrtLightnet::plotCircleIntoBevmap(const int im_w, const int im_h, const Calibration calibdata, std::vector<std::string> &names, std::vector<std::string> &target) {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Loop through all bindings to find depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w = static_cast<float>(outputW) / static_cast<float>(im_w);
	  float scale_h = static_cast<float>(outputH) / static_cast<float>(im_h);
	  float *buf = static_cast<float *>(output_h_.at(i - 1).get());
	  float gran_h = static_cast<float>(GRID_H) / calibdata.max_distance;

	  // Process each bounding box.
	  for (const auto &b : bbox_) {
	    bool flg = false;

	    // Check if the class of the bounding box matches any target classes.
	    for (const auto &t : target) {
	      if (t == names[b.classId]) {
		flg = true;
		break;
	      }
	    }

	    if (flg) {
	      continue;
	    }

	    // Calculate coordinates for the BEV map.
	    int cx = (b.box.x2 + b.box.x1) / 2 - 1;
	    int cy = b.box.y2 - 1;
	    int stride = outputW * static_cast<int>(cy * scale_h);
	    float distance = buf[stride + static_cast<int>(cx * scale_w)] * calibdata.max_distance;
	    distance = std::min(distance, calibdata.max_distance);

	    float src_x = static_cast<float>(cx);
	    float x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance;

	    if (x3d > 20.0) {
	      continue;
	    }

	    x3d = (x3d + 20.0f) * GRID_W / 40.0f;
	    int x_bev = static_cast<int>(x3d);
	    int y_bev = static_cast<int>(GRID_H - static_cast<int>(distance * gran_h));

	    // Plot a circle onto the BEV map.
	    cv::circle(bevmap_, cv::Point(x_bev, y_bev), 8, cv::Scalar(255, 255, 255), 2);
	  }
	}
      }
    }
  }
  
  /**
   * @brief Adds bounding boxes into a bird's-eye view (BEV) map.
   *
   * This function processes bounding box information to project it into a BEV map. 
   * It calculates the position, dimensions, and angles of objects based on keypoints 
   * and calibration data. The resulting bounding boxes are visualized in the BEV map.
   * (under development ...)
   * @param im_w Width of the input image.
   * @param im_h Height of the input image.
   * @param calibdata Calibration data containing intrinsic and extrinsic camera parameters.
   * @param names Vector of object class names corresponding to their labels.
   */
  void TrtLightnet::addBBoxIntoBevmap(const int im_w, const int im_h, const Calibration calibdata, std::vector<std::string> &names)
  {
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    const float VEHICLE_LEN = 2.2;
    const float VEHICLE_WIDTH = 1.2;
    const float PED_LEN = 0.25;
    const float PED_WIDTH = 0.25;
    const float CYCLE_LEN = 1.0;
    const float CYCLE_WIDTH = 0.5;        
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];
      // Identify the depth map tensor by checking the channel size and tensor name.
      if (chan_size != chan && outputW > 1 && outputH > 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Check if the tensor name contains "lgx" and if the tensor type is 'kFLOAT'.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float scale_w =  (float)outputW / (float)(im_w);
	  float scale_h = (float)outputH / (float)(im_h);
	  float *buf = (float *)output_h_.at(i-1).get();
	  float gran_h = (float)GRID_H / calibdata.max_distance;

	  for (const auto& b : bbox_) {
	    if (b.keypoint.size()) {
	      int offset = 0;
	      int lx0 = b.keypoint[0].lx0 + offset;
	      int ly0 = b.keypoint[0].ly0 + offset;
	      int lx1 = b.keypoint[0].lx1 + offset;
	      int ly1 = b.keypoint[0].ly1 - offset;
	      int rx0 = b.keypoint[0].rx0 - offset;
	      int ry0 = b.keypoint[0].ry0 + offset;
	      int rx1 = b.keypoint[0].rx1 - offset;
	      int ry1 = b.keypoint[0].ry1 - offset;
	      bool is_front = (b.keypoint[0].attr_prob > 0.5);
	      int cx = (b.box.x1 + b.box.x2) / 2;
	      //int cy = b.box.y2;
	      int cy = (ly0+ry0)/2;	      

	      int stride = outputW * int(ly0 * scale_h);
	      float distance = buf[stride + int(lx0 * scale_w)] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;
	      float src_x = (float)lx0;

	      float x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance;

	      // Skip points that are too far in the x-direction or below the ground plane.
	      if (x3d > 20.0) continue;
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      cv::Point p_lx0;
	      p_lx0.x = x3d;
	      p_lx0.y = GRID_H - static_cast<int>(distance * gran_h);
#ifdef DEBUG	      	      
	      cv::circle(bevmap_, p_lx0, 1, cv::Scalar(255, 0, 255), cv::FILLED, 4, 0);
#endif
	      stride = outputW * int(ly1 * scale_h);
	      distance = buf[stride + int(lx1 * scale_w)] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;	      
	      src_x = (float)lx1;

	      x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance * 2.0;

	      // Skip points that are too far in the x-direction or below the ground plane.
	      if (x3d > 20.0) continue;
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      cv::Point p_lx1;
	      p_lx1.x = x3d;
	      p_lx1.y = GRID_H - static_cast<int>(distance * gran_h);
#ifdef DEBUG	      	      
	      cv::circle(bevmap_, p_lx1, 1, cv::Scalar(255, 255, 0), cv::FILLED, 4, 0);
	      cv::line(bevmap_, p_lx0, p_lx1, cv::Scalar(255, 255, 255), 1, 8, 0);
#endif	      
	      //rx0,rx1
	      stride = outputW * int(ry0 * scale_h);
	      distance = buf[stride + int(rx0 * scale_w)] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;
	      src_x = (float)rx0;

	      x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance * 2.0;

	      // Skip points that are too far in the x-direction or below the ground plane.
	      if (x3d > 20.0) continue;
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      cv::Point p_rx0;
	      p_rx0.x = x3d;
	      p_rx0.y = GRID_H - static_cast<int>(distance * gran_h);
#ifdef DEBUG	      	      
	      cv::circle(bevmap_, p_rx0, 1, cv::Scalar(255, 0, 255), cv::FILLED, 4, 0);
#endif
	      stride = outputW * int(ry1 * scale_h);
	      distance = buf[stride + int(rx1 * scale_w)] * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;	      
	      src_x = (float)rx1;

	      x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance * 2.0;

	      // Skip points that are too far in the x-direction or below the ground plane.
	      if (x3d > 20.0) continue;
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      cv::Point p_rx1;
	      p_rx1.x = x3d;
	      p_rx1.y = GRID_H - static_cast<int>(distance * gran_h);
#ifdef DEBUG	      	      
	      cv::circle(bevmap_, p_rx1, 1, cv::Scalar(255, 255, 0), cv::FILLED, 4, 0);
	      cv::line(bevmap_, p_rx0, p_rx1, cv::Scalar(255, 255, 255), 1, 8, 0);
#endif
	      //center
	      stride = outputW * int(cy * scale_h);
	      float rel_dist = buf[stride + int(cx * scale_w)];
	      if (b.keypoint[0].isOccluded) {
		rel_dist = get_meadian_dist(buf, outputW, scale_w, scale_h, b.box);
	      }
	      distance = rel_dist * calibdata.max_distance;
	      distance = distance > calibdata.max_distance ? calibdata.max_distance : distance;
	      if (distance < 2.0) {
		continue;
	      }
	      src_x = (float)cx;

	      x3d = ((src_x - calibdata.u0) / calibdata.fx) * distance * 2.0;

	      // Skip points that are too far in the x-direction or below the ground plane.
	      x3d = (x3d + 20.0) * GRID_W / 40.0;
	      cv::Point p_center;
	      p_center.x = x3d;
	      p_center.y = GRID_H - static_cast<int>(distance * gran_h);
#ifdef DEBUG	      	      
	      cv::circle(bevmap_, p_center, 4, cv::Scalar(0, 0, 255), cv::FILLED, 4, 0);
#endif
	      float angle;

	      if (abs(b.box.x1-lx1) < abs(b.box.x2-rx1)) {	      
		if (abs(p_lx1.x-p_lx0.x) < 1) {
		  angle = 0.0;
		} else {
		  angle =  calculateAngle(p_lx1.x, p_lx1.y, p_lx0.x , p_lx0.y) - 90;		  
		}
	      } else {
		if (abs(p_rx1.x-p_rx0.x) < 1) {
		  angle = 0.0;
		} else {
		  angle =  calculateAngle(p_rx1.x, p_rx1.y, p_rx0.x , p_rx0.y) - 90;		  
		}
	      }
	      //angle = quantize_angle(angle, 96);
	      cv::Point2f center;
	      cv::Size rectSize;
	      int xlen = b.box.x2 - b.box.x1;
	      float cuboid_len;
	      float cuboid_width;	      
	      cuboid_len = VEHICLE_LEN;
	      cuboid_width = VEHICLE_WIDTH;
	      if (names[b.label] == "PEDESTRIAN") {
		cuboid_len = PED_LEN;
		cuboid_width = PED_WIDTH;
	      } else if (names[b.label] == "BICYCLE" || names[b.label] == "MOTORBIKE") {
		cuboid_len = CYCLE_LEN;
		cuboid_width = CYCLE_WIDTH;
	      }
	      if (abs(lx0-b.box.x1) < (xlen * 0.07) && abs(rx0-b.box.x2) < (xlen * 0.07)) {
		angle = -0.0;
		center.x = p_center.x;
		center.y = p_lx0.y - static_cast<int>(cuboid_len * gran_h) / 2;
		rectSize.width = static_cast<int>(cuboid_width * (GRID_W / 40.0));		
		rectSize.height = static_cast<int>(cuboid_len * gran_h);
	      } else if (abs(b.box.x1-lx1) < abs(b.box.x2-rx1)) {	      
		p_lx1.y = p_center.y - static_cast<int>(cuboid_len * gran_h);
		p_lx1.x = p_center.x - static_cast<int>(cuboid_width/2 * (GRID_W / 40.0));
		p_rx0.x = p_center.x + static_cast<int>(cuboid_width/2 * (GRID_W / 40.0));		  
		p_rx0.y = p_center.y;		
		center.x = (p_lx0.x+p_rx0.x) / 2.0;
		center.y = (p_lx1.y+p_rx0.y) / 2.0;
		rectSize.width = p_rx0.x - p_lx1.x;
		rectSize.height = p_rx0.y - p_lx1.y;				
	      } else {
		p_lx0.y = p_center.y;
		p_lx0.x = p_center.x - static_cast<int>(cuboid_width/2 * (GRID_W / 40.0));
		p_rx1.x = p_center.x + static_cast<int>(cuboid_width/2 * (GRID_W / 40.0));		  
		p_rx1.y = p_center.y - static_cast<int>(cuboid_len * gran_h);
		center.x = (p_rx0.x+p_lx0.x) / 2.0;
		center.y = (p_rx1.y+p_lx0.y) / 2.0;
		rectSize.width = p_rx1.x - p_lx0.x;
		rectSize.height = p_lx0.y - p_rx1.y;		
	      }

	      if (b.keypoint[0].isOccluded) {
		angle = 0.0;
	      }		
	      cv::RotatedRect rotatedRect(center, rectSize, angle);

	      cv::Point2f vertices[4];
	      rotatedRect.points(vertices);
		
	      for (int v = 0; v < 4; v++) {
		
		if (abs(lx0-b.box.x1) < (xlen * 0.07) && abs(rx0-b.box.x2) < (xlen * 0.07)) {		  
		  cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 128, 0), 1);
		  //cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 128, 0), 1);		  		  
		} else if (abs(b.box.x1-lx1) < abs(b.box.x2-rx1)) {	      		  
		  //cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(0, 0, 255), 1);
		  cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 128, 0), 1);		  
		} else {
		  //cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(0, 255, 0), 1);
		  cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 128, 0), 1);		  
		}
		if (v == 1 && !is_front) {
		  cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 255, 255), 2);
		} else if (v == 3 && is_front) {
		  cv::line(bevmap_, vertices[v], vertices[(v + 1) % 4], cv::Scalar(255, 255, 255), 2);
		}
		
	      }
#ifdef DEBUG	      	      	      
	      if (b.keypoint[0].isOccluded) {
		cv::putText(bevmap_, "OCC", p_center, 0, 0.5, cv::Scalar(0, 0, 255), 2);	  
	      }
	      cv::putText(bevmap_, std::to_string(int(angle)), p_center, 0, 0.5, cv::Scalar(255, 0, 255), 1);	  	      
#endif	      
	    }

	  }	  
	  
	}
      }
    }
  }
  
  /**
   * @brief Applies the argmax operation to a given buffer and writes the result into a mask.
   * 
   * This function iterates over the height and width of the output, determines the channel with the maximum 
   * value for each position, and assigns the corresponding color from argmax2bgr to the mask.
   * 
   * @param mask The output mask to which the argmax results are written. This is a cv::Mat object with type CV_8UC3.
   * @param buf The input buffer containing the data in NCHW format (channels, height, width).
   * @param chan The number of channels in the input buffer.
   * @param outputH The height of the output.
   * @param outputW The width of the output.
   * @param argmax2bgr A vector mapping channel indices to colors (cv::Vec3b) for visualization.
   */
  void TrtLightnet::applyArgmax(cv::Mat &mask, const float *buf, const int chan, const int outputH, const int outputW, std::vector<cv::Vec3b> &argmax2bgr)
  {
    for (int y = 0; y < outputH; y++) {
      cv::Vec3b *ptr = mask.ptr<cv::Vec3b>(y);
      for (int x = 0; x < outputW; x++) {
	float max = 0.0;
	int index = 0;
	for (int c = 0; c < chan; c++) {
	  //NCHW
	  float value = buf[c * outputH * outputW + y * outputW + x];
	  if (max < value) {
	    max = value;
	    index = c;
	  }
	}
	ptr[x] = argmax2bgr[index];
      }
    }

  }
  
  /**
   * Generates segmentation masks from the network's output using argmax operation results.
   * 
   * @param argmax2bgr A vector mapping class indices to BGR colors for visualization.
   */
  void TrtLightnet::makeMask(std::vector<cv::Vec3b> &argmax2bgr)
  {
    masks_.clear();
    // Formula to identify output tensors not related to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      const int chan = dims.d[1];
      // Identifying tensors by channel size and name for segmentation masks.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "argmax")) { // Check if tensor name contains "argmax".
	  cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC3);

	  
	  std::vector<ucharRGB> colorMap(argmax2bgr.size());
	  for (size_t i = 0; i < argmax2bgr.size(); ++i) {
	    colorMap[i] = {argmax2bgr[i][0], argmax2bgr[i][1], argmax2bgr[i][2]};
	  }
	  if (!d_colorMap_) {
	    d_colorMap_ = cuda_utils::make_unique<ucharRGB[]>(colorMap.size());	    
	  }
	  if (!d_mask_) {
	    d_mask_ = cuda_utils::make_unique<unsigned char[]>(outputW * outputH * 3);
	  }
	  cudaMemcpy(d_colorMap_.get(), colorMap.data(), colorMap.size() * sizeof(ucharRGB), cudaMemcpyHostToDevice);	  
	  mapArgmaxtoColorGpu(d_mask_.get(), (int *)output_d_.at(i-1).get(), outputW, outputH, d_colorMap_.get(), *stream_);
	  cudaMemcpy(mask.data, d_mask_.get(), outputW * outputH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);	  	  
	  cudaStreamSynchronize(*stream_);
	  masks_.push_back(mask);
	} else if (contain(name, "softmax")) { // Check if tensor name contains "softmax".
	  cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC3);
	  const float *buf = (float *)output_h_.at(i-1).get();
	  applyArgmax(mask, buf, chan, outputH, outputW, argmax2bgr);
	  masks_.push_back(mask);
	}
      }
    }
  }


  /**
   * @brief This function calculates the entropy maps from the softmax output of the network.
   * It identifies the tensors that are not related to bounding box detections and processes 
   * the tensors whose names contain "softmax". The function computes the entropy for each 
   * channel and stores the entropy maps.
   */    
  void TrtLightnet::calcEntropyFromSoftmax(void)
  {
    // Formula to identify output tensors not related to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    entropies_.clear();
    ent_maps_.clear();
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      const int chan = dims.d[1];
      // Identifying tensors by channel size and name for segmentation masks.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "softmax")) { // Check if tensor name contains "softmax".
	  cv::Mat vis = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	  cv::Mat ent_map = cv::Mat::zeros(outputH, outputW, CV_8UC3);	  	  
	  std::vector<cv::Mat> entropy_maps(chan);
	  std::vector<float> entropy(chan, 0.0f);
	  for (int c = 0; c < chan; ++c) {
	    entropy_maps[c] = cv::Mat::zeros(outputH, outputW, CV_32FC1);
	  }

	  std::chrono::high_resolution_clock::time_point start, end;
	  start = std::chrono::high_resolution_clock::now();

	  computeEntropyMapGpu((float *)output_d_.at(i-1).get(), d_entropy_.get(),
			       chan, outputH, outputW, *stream_);

	  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_entropy_.get(), d_entropy_.get(), outputW * outputH * chan * sizeof(float), cudaMemcpyDeviceToHost, *stream_));
	  cudaStreamSynchronize(*stream_); 

	  end = std::chrono::high_resolution_clock::now();
	  std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	  std::cout << "##Entropuy: " << duration.count() << " ms " << std::endl;

	  start = std::chrono::high_resolution_clock::now();
	  //#pragma omp parallel for schedule(static, 2a) num_threads(2)    	  
	  for (int y = 0; y < outputH; y++) {
	    for (int x = 0; x < outputW; x++) {
	      for (int c = 0; c < chan; c++) {	  
		//float ent = entropy_maps[c].at<float>(y, x);
		int index = c * outputH * outputW + outputW * y + x;
		float ent = h_entropy_[index] ;				
		entropy[c] += ent;
		//vis.at<unsigned char>(y, x) += (unsigned char)(255 * ent)/chan;		
		vis.at<unsigned char>(y, x) = (vis.at<unsigned char>(y, x) < (unsigned char)(255 * ent)) ? (unsigned char)(255 * ent) : vis.at<unsigned char>(y, x);		
	      }
	    }
	  }
	  end = std::chrono::high_resolution_clock::now();
	  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	  std::cout << "##visalize: " << duration.count() << " ms " << std::endl;
	  
	  for (int c = 0; c < chan; c++) {
	    entropy[c] /= (outputH * outputW);
	  }	  
	  for (int c = 0; c < chan; c++) {	    
	    std::cout << "Entropy (" << c  << ") " << entropy[c] <<std::endl;
	  }
	  
	  entropies_.push_back(entropy);
	  for (int y = 0; y < outputH; y++) {
	    for (int x = 0; x < outputW; x++) {
	      const auto &color = jet_colormap[vis.at<unsigned char>(y, x)];	      
	      ent_map.at<cv::Vec3b>(y, x)[0] = color[0];
	      ent_map.at<cv::Vec3b>(y, x)[1] = color[1];
	      ent_map.at<cv::Vec3b>(y, x)[2] = color[2];
	    }
	  }
	  ent_maps_.push_back(ent_map);
	}
      }
    }
  }

  /**
   * @brief This function returns the calculated entropy maps.
   * 
   * @return A vector of cv::Mat objects representing the entropy maps.
   */  
  std::vector<cv::Mat> TrtLightnet::getEntropymaps() {
    return ent_maps_;
  }

  /**
   * @brief This function returns the calculated entropies for each channel.
   * 
   * @return A vector of vectors, where each inner vector contains the entropies for a particular tensor.
   */
  std::vector<std::vector<float>> TrtLightnet::getEntropies() {
    return entropies_;
  }
  
  /**
   * Calculates cross-task inconsistency between bounding box detections and segmentation maps.
   *
   * @param im_width Width of the input image.
   * @param im_height Height of the input image.
   * @param seg_colormap Vector of Colormap objects containing segmentation information for each class.
   */
  void TrtLightnet::calcCrossTaskInconsistency(int im_width, int im_height, std::vector<Colormap> &seg_colormap) {
    // Expected channel size for bounding box-related tensors
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    inconsistency_.clear();
    inconsistency_map_.clear();

    // Iterate over output tensors, ignoring the first binding as it's typically the input tensor
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      int chan = dims.d[1];

      // Check if tensor is a segmentation mask (not a bounding box) by verifying channel size
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "softmax")) { // Identifies tensors with "softmax" in their name
	  const float *buf = (float *)output_h_.at(i-1).get();
	  cv::Mat vis = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	  cv::Mat argmax = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	  cv::Mat ent_map = cv::Mat::zeros(outputH, outputW, CV_8UC3);
	  std::vector<float> inconsistency(chan, 0.0f);

	  // Populate segmentation mask based on softmax probabilities
	  for (int y = 0; y < outputH; y++) {
	    for (int x = 0; x < outputW; x++) {
	      for (int c = 0; c < chan; c++) {
		if (!seg_colormap[c].is_dynamic) continue;
		float p = buf[c * outputH * outputW + y * outputW + x];
		unsigned char val = static_cast<unsigned char>(255 * p);
		if (val > vis.at<unsigned char>(y, x)) {
		  vis.at<unsigned char>(y, x) = val;
		  argmax.at<unsigned char>(y, x) = c;
		}
	      }
	    }
	  }

	  // Define margin to allow for bounding box overlap
	  float margin = 0.05;

	  // Adjust segmentation values within bounding box areas
	  for (const auto& b : bbox_) {
	    float scale_w = outputW / static_cast<float>(im_width);
	    float scale_h = outputH / static_cast<float>(im_height);
	    int x1 = b.box.x1 * scale_w;
	    int x2 = b.box.x2 * scale_w;
	    int y1 = b.box.y1 * scale_h;
	    int y2 = b.box.y2 * scale_h;
	    int ylen = y2 - y1;
	    int xlen = x2 - x1;
	    x1 = std::max(0, static_cast<int>(x1 - xlen * margin));
	    y1 = std::max(0, static_cast<int>(y1 - ylen * margin));
	    x2 = std::min(outputW - 1, static_cast<int>(x2 + xlen * margin));
	    y2 = std::min(outputH - 1, static_cast<int>(y2 + ylen * margin));
	    unsigned char prob = static_cast<unsigned char>(255 * b.prob);

	    for (int y = y1; y <= y2; y++) {
	      for (int x = x1; x <= x2; x++) {
		vis.at<unsigned char>(y, x) = std::max(0, vis.at<unsigned char>(y, x) - prob);
	      }
	    }
	  }

	  // Map segmentation inconsistency and update color map for visualization
	  for (int y = 0; y < outputH; y++) {
	    for (int x = 0; x < outputW; x++) {
	      int value = vis.at<unsigned char>(y, x);
	      int index = argmax.at<unsigned char>(y, x);
	      const auto &color = jet_colormap[255 - value];
	      ent_map.at<cv::Vec3b>(y, x)[0] = color[0];
	      ent_map.at<cv::Vec3b>(y, x)[1] = color[1];
	      ent_map.at<cv::Vec3b>(y, x)[2] = color[2];
	      inconsistency[index] += (value / 255.0f);
	    }
	  }

	  inconsistency_.push_back(inconsistency);
	  inconsistency_map_.push_back(ent_map);
	}
      }
    }
  }

  /**
   * Returns the inconsistency maps generated for cross-task inconsistency.
   *
   * @return Vector of cv::Mat representing the inconsistency maps.
   */
  std::vector<cv::Mat> TrtLightnet::getCrossTaskInconsistency_map() {
    return inconsistency_map_;
  }

  /**
   * Returns the calculated inconsistency values for each segmentation class.
   *
   * @return Vector of vectors containing inconsistency values for each class.
   */
  std::vector<std::vector<float>> TrtLightnet::getCrossTaskInconsistencies() {
    return inconsistency_;
  }
    
  /**
   * Return mask.
   * 
   * @return A vector of OpenCV Mat objects, each representing a mask image where each pixel's color corresponds to its class's color.
   */
  std::vector<cv::Mat> TrtLightnet::getMask(void)
  {
    return masks_;
  }

  /**
   * Retrieves debugging tensors from the TensorRT bindings that are not inputs.
   *
   * This method iterates over all bindings in a TensorRT common context (`trt_common_`),
   * identifying output tensors whose names match any in a predefined list of debug tensor names.
   * For each matching tensor, this method collects its memory buffer, dimensions, and name,
   * and stores them for external use, typically for debugging or visualization purposes.
   *
   * @param dim_infos Reference to a vector of nvinfer1::Dims to be filled with dimensions of each tensor.
   * @param names Reference to a vector of strings to be filled with names of each tensor.
   * @return std::vector<float*> A vector of pointers to the buffers of debug tensors.
   */
  std::vector<float*> TrtLightnet::getDebugTensors(std::vector<nvinfer1::Dims> &dim_infos, std::vector<std::string> &names)
  {
    std::vector<float*> debug_tensors;
    for (int i = 0; i < trt_common_->getNbBindings(); i++) {
      if (trt_common_->bindingIsInput(i)) {
	continue;
      }
      std::string name = trt_common_->getIOTensorName(i);
      std::vector<std::string> debug_names = trt_common_->getDebugTensorNames();
      for (int j = 0; j < (int)(debug_names.size()); j++) {
	if (debug_names[j] == name) {
	  const auto dims = trt_common_->getBindingDimensions(i);
	  float *buf = (float *)output_h_.at(i-1).get();
	  debug_tensors.emplace_back(buf);
	  dim_infos.emplace_back(dims);
	  names.emplace_back(name);
	}
      }
    }
    return debug_tensors;
  }
    
  /**
   * Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes based on their IoU.
   *
   * @param nmsThresh The IoU threshold used to determine whether boxes overlap too much and should be suppressed.
   * @param binfo Vector of BBoxInfo containing the bounding boxes along with their class labels and probabilities.
   * @return A filtered vector of BBoxInfo, containing only the boxes that survive the NMS.
   */
  std::vector<BBoxInfo> TrtLightnet::nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
  {
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
      if (x1min > x2min) {
	  std::swap(x1min, x2min);
	  std::swap(x1max, x2max);
      }
      return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
      float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
      float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
      float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
      float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
      float overlap2D = overlapX * overlapY;
      float u = area1 + area2 - overlap2D;
      return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
		     [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto& i : binfo) {
      bool keep = true;
      for (auto& j : out) {
	if (keep) {
	  float overlap = computeIoU(i.box, j.box);
	  keep = overlap <= nmsThresh;	  
	} else {
	  break;
	}
      }
      if (keep) out.push_back(i);
    }
    return out;
  }

  /**
   * Applies Non-Maximum Suppression (NMS) separately for each class across all detections.
   *
   * @param nmsThresh The IoU threshold for NMS.
   * @param binfo Vector of BBoxInfo containing the bounding boxes, class labels, and probabilities.
   * @param numClasses The total number of classes.
   * @return A vector of BBoxInfo, containing only the boxes that survive NMS, across all classes.
   */
  std::vector<BBoxInfo> TrtLightnet::nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo, const uint32_t numClasses)
  {
    std::vector<BBoxInfo> result;
    // Split bounding boxes by their class labels.
    std::vector<std::vector<BBoxInfo>> splitBoxes(numClasses);
    for (const auto& box : binfo) {
      splitBoxes.at(box.label).push_back(box);
    }

    // Apply NMS for each class and collect the results.
    for (auto& boxes : splitBoxes) {
      boxes = nonMaximumSuppression(nmsThresh, boxes);
      result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
  }

  /**
   * Converts bounding box dimensions from the feature map scale to the original image scale.
   * 
   * @param bx Center X coordinate of the bounding box on the feature map.
   * @param by Center Y coordinate of the bounding box on the feature map.
   * @param bw Width of the bounding box on the feature map.
   * @param bh Height of the bounding box on the feature map.
   * @param stride_h_ Vertical stride of the feature map.
   * @param stride_w_ Horizontal stride of the feature map.
   * @param netW Width of the network input.
   * @param netH Height of the network input.
   * @return A BBox struct containing the converted bounding box coordinates.
   */
  BBox TrtLightnet::convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh,
				   const uint32_t& stride_h_, const uint32_t& stride_w_,
				   const uint32_t& netW, const uint32_t& netH)
  {
    BBox b;
    // Convert coordinates from feature map scale back to original image scale
    float x = bx * stride_w_;
    float y = by * stride_h_;

    // Calculate top-left and bottom-right coordinates
    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;
    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    // Clamp coordinates to be within the network input dimensions
    b.x1 = CLAMP(b.x1, 0.0f, static_cast<float>(netW));
    b.x2 = CLAMP(b.x2, 0.0f, static_cast<float>(netW));
    b.y1 = CLAMP(b.y1, 0.0f, static_cast<float>(netH));
    b.y2 = CLAMP(b.y2, 0.0f, static_cast<float>(netH));

    return b;
  }
  
  /**
   * Adds a bounding box proposal after converting its dimensions and scaling it relative to the original image size.
   * 
   * @param bx Center X coordinate of the bounding box proposal.
   * @param by Center Y coordinate of the bounding box proposal.
   * @param bw Width of the bounding box proposal.
   * @param bh Height of the bounding box proposal.
   * @param stride_h_ Vertical stride of the feature map.
   * @param stride_w_ Horizontal stride of the feature map.
   * @param maxIndex The class index with the highest probability.
   * @param maxProb The probability of the most likely class.
   * @param image_w Width of the original image.
   * @param image_h Height of the original image.
   * @param input_w Width of the network input.
   * @param input_h Height of the network input.
   * @param binfo Vector to which the new BBoxInfo object will be added.
   */
  void TrtLightnet::addBboxProposal(const float bx, const float by, const float bw, const float bh,
				    const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb,
				    const uint32_t image_w, const uint32_t image_h,
				    const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo)
  {
    BBoxInfo bbi;
    // Convert the bounding box to the original image scale
    bbi.box = convertBboxRes(bx, by, bw, bh, stride_h_, stride_w_, input_w, input_h);

    // Skip invalid boxes
    if (bbi.box.x1 > bbi.box.x2 || bbi.box.y1 > bbi.box.y2) {
      return;
    }

    // Scale box coordinates to match the size of the original image
    bbi.box.x1 = (bbi.box.x1 / input_w) * image_w;
    bbi.box.y1 = (bbi.box.y1 / input_h) * image_h;
    bbi.box.x2 = (bbi.box.x2 / input_w) * image_w;
    bbi.box.y2 = (bbi.box.y2 / input_h) * image_h;

    // Set label and probability
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex; // Note: 'label' and 'classId' are set to the same value. Consider if this redundancy is necessary.
    bbi.isHierarchical = false;
    // Add the box info to the vector
    binfo.push_back(bbi);
  }
  

  /**
   * Decodes the output tensor from a neural network into bounding box information.
   * This implementation specifically handles tensors formatted for Scaled-YOLOv4 detections.
   * 
   * @param imageIdx The index of the image within the batch being processed.
   * @param imageH The height of the input image.
   * @param imageW The width of the input image.
   * @param inputH The height of the network input.
   * @param inputW The width of the network input.
   * @param anchor Pointer to the anchor sizes used by the network.
   * @param anchor_num The number of anchors.
   * @param output Pointer to the output data from the network.
   * @param gridW The width of the grid used by the network for detections.
   * @param gridH The height of the grid used by the network for detections.
   * @return A vector of BBoxInfo containing decoded bounding box information.
   */
  std::vector<BBoxInfo> TrtLightnet::decodeTensor(const int imageIdx, const int imageH, const int imageW,  const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH)
  {
    const int volume = gridW * gridH;
    const float* detections = &output[imageIdx * volume * anchor_num * (5 + num_class_)];

    std::vector<BBoxInfo> binfo;
    const float scale_x_y = 2.0; // Scale factor used for bounding box center adjustments.
    const float offset = 0.5 * (scale_x_y - 1.0); // Offset for scaled centers.

    for (int y = 0; y < gridH; ++y) {
      for (int x = 0; x < gridW; ++x) {
	for (int b = 0; b < anchor_num; ++b) {
	  const int numGridCells = gridH * gridW;
	  const int bbindex = (y * gridW + x) + numGridCells * b * (5 + num_class_);

	  const float objectness = detections[bbindex + 4 * numGridCells]; // Objectness score.
	  if (objectness < score_threshold_) {
	    continue; // Skip detection if below threshold.
	  }

	  // Extract anchor dimensions.
	  const float pw = static_cast<float>(anchor[b * 2]);
	  const float ph = static_cast<float>(anchor[b * 2 + 1]);

	  // Decode bounding box center and dimensions.
	  // Scaled-YOLOv4 format for simple and fast decorder
	  // bx = tx * 2 + cx - 0.5
	  // by = ty * 2 + cy - 0.5
	  // bw = (tw * 2) * (tw * 2) * pw
	  // bh = (th * 2) * (th * 2) * pw
	  // The sigmoid is included in the last layer of the DNN models.
	  // Cite in https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982 (Loss for YOLOv3, YOLOv4 and Scaled-YOLOv4)	  
	  const float bx = x + scale_x_y * detections[bbindex] - offset;
	  const float by = y + scale_x_y * detections[bbindex + numGridCells] - offset;
	  const float bw = pw * std::pow(detections[bbindex + 2 * numGridCells] * 2, 2);
	  const float bh = ph * std::pow(detections[bbindex + 3 * numGridCells] * 2, 2);

	  // Decode class probabilities.
	  float maxProb = 0.0f;
	  int maxIndex = -1;
	  for (int i = 0; i < num_class_; ++i) {
	    float prob = detections[bbindex + (5 + i) * numGridCells];
	    if (prob > maxProb) {
	      maxProb = prob;
	      maxIndex = i;
	    }
	  }
	  maxProb *= objectness; // Adjust probability with objectness score.

	  // Add bounding box proposal if above the threshold.
	  if (maxProb > score_threshold_) {
	    const uint32_t strideH = inputH / gridH;
	    const uint32_t strideW = inputW / gridW;
	    addBboxProposal(bx, by, bw, bh, strideH, strideW, maxIndex, maxProb, imageW, imageH, inputW, inputH, binfo);
	  }
	}
      }
    }
    return binfo;
  }


  /**
   * Adds a bounding box proposal for TLR after converting its dimensions and scaling it relative to the original image size.
   * 
   * @param bx Center X coordinate of the bounding box proposal.
   * @param by Center Y coordinate of the bounding box proposal.
   * @param bw Width of the bounding box proposal.
   * @param bh Height of the bounding box proposal.
   * @param stride_h_ Vertical stride of the feature map.
   * @param stride_w_ Horizontal stride of the feature map.
   * @param maxIndex The class index with the highest probability.
   * @param maxProb The probability of the most likely class.
   * @param maxSubIndex The subclass index with the highest probability.
   * @param cos Cos value for angle (This value in only used for arrow type)
   * @param sin Sin value for angle (This value in only used for arrow type)
   * @param image_w Width of the original image.
   * @param image_h Height of the original image.
   * @param input_w Width of the network input.
   * @param input_h Height of the network input.
   * @param binfo Vector to which the new BBoxInfo object will be added.
   */
  void TrtLightnet::addTLRBboxProposal(const float bx, const float by, const float bw, const float bh,
						const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb, const int maxSubIndex,
						const float cos, const float sin,
				    const uint32_t image_w, const uint32_t image_h,
				    const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo)
  {
    BBoxInfo bbi;
    // Convert the bounding box to the original image scale
    bbi.box = convertBboxRes(bx, by, bw, bh, stride_h_, stride_w_, input_w, input_h);

    // Skip invalid boxes
    if (bbi.box.x1 > bbi.box.x2 || bbi.box.y1 > bbi.box.y2) {
      return;
    }

    // Scale box coordinates to match the size of the original image
    bbi.box.x1 = (bbi.box.x1 / input_w) * image_w;
    bbi.box.y1 = (bbi.box.y1 / input_h) * image_h;
    bbi.box.x2 = (bbi.box.x2 / input_w) * image_w;
    bbi.box.y2 = (bbi.box.y2 / input_h) * image_h;

    // Set label and probability
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = maxIndex; // Note: 'label' and 'classId' are set to the same value. Consider if this redundancy is necessary.

    // Add the box info to the vector
    bbi.isHierarchical = true;
    bbi.subClassId = maxSubIndex;
    bbi.cos = cos;
    bbi.sin = sin;
    binfo.push_back(bbi);
  }


  /**
   * Decodes the output tensor from a neural network into bounding box information.
   * This implementation specifically handles tensors formatted for Scaled-YOLOv4 detections.
   *
   * @param imageIdx The index of the image within the batch being processed.
   * @param imageH The height of the input image.
   * @param imageW The width of the input image.
   * @param inputH The height of the network input.
   * @param inputW The width of the network input.
   * @param anchor Pointer to the anchor sizes used by the network.
   * @param anchor_num The number of anchors.
   * @param output Pointer to the output data from the network.
   * @param gridW The width of the grid used by the network for detections.
   * @param gridH The height of the grid used by the network for detections.
   * @return A vector of BBoxInfo containing decoded bounding box information.
   */
  std::vector<BBoxInfo> TrtLightnet::decodeTLRTensor(const int imageIdx, const int imageH, const int imageW,  const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH)
  {
    //  {bbox}    {obj}  {color}        {type}           {angle}
    // {0 1 2 3}   {4}   {5 6 7}    {8 9 10 11 12 13}    {14 15}
    const int volume = gridW * gridH;
    const float* detections = &output[imageIdx * volume * anchor_num * (5 + 3 + num_class_ + 2)];
    
    std::vector<BBoxInfo> binfo;
    const float scale_x_y = 2.0; // Scale factor used for bounding box center adjustments.
    const float offset = 0.5 * (scale_x_y - 1.0); // Offset for scaled centers.

    for (int y = 0; y < gridH; ++y) {
      for (int x = 0; x < gridW; ++x) {
	for (int b = 0; b < anchor_num; ++b) {
	  const int numGridCells = gridH * gridW;
	  const int bbindex = (y * gridW + x) + numGridCells * b * (5 + 3 + num_class_ + 2);

	  const float objectness = detections[bbindex + 4 * numGridCells]; // Objectness score.
	  if (objectness < score_threshold_) {
	    continue; // Skip detection if below threshold.
	  }

	  // Extract anchor dimensions.
	  const float pw = static_cast<float>(anchor[b * 2]);
	  const float ph = static_cast<float>(anchor[b * 2 + 1]);

	  // Decode bounding box center and dimensions.
	  // Scaled-YOLOv4 format for simple and fast decorder
	  // bx = tx * 2 + cx - 0.5
	  // by = ty * 2 + cy - 0.5
	  // bw = (tw * 2) * (tw * 2) * pw
	  // bh = (th * 2) * (th * 2) * pw
	  // The sigmoid is included in the last layer of the DNN models.
	  // Cite in https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982 (Loss for YOLOv3, YOLOv4 and Scaled-YOLOv4)
	  const float bx = x + scale_x_y * detections[bbindex + X_INDEX * numGridCells] - offset;
	  const float by = y + scale_x_y * detections[bbindex + Y_INDEX * numGridCells] - offset;
	  const float bw = pw * std::pow(detections[bbindex + W_INDEX * numGridCells] * 2, 2);
	  const float bh = ph * std::pow(detections[bbindex + H_INDEX * numGridCells] * 2, 2);

	  // Decode class probabilities.
	  float maxProb = 0.0f;
	  int maxIndex = -1;
	  for (int i = 0; i < num_class_; ++i) {
	    float prob = detections[bbindex + (5 + 3 + i) * numGridCells];
	    if (prob > maxProb) {
	      maxProb = prob;
	      maxIndex = i;
	    }
	  }
	  float maxSubProb = 0.0f;
	  int maxSubIndex = -1;
	  for (int i = 0; i < 3; ++i) {
	    float prob = detections[bbindex + (5+i) * numGridCells];
	    if (prob > maxSubProb) {
	      maxSubProb = prob;
	      maxSubIndex = i;
	    }
	  }
	  maxProb *= objectness; // Adjust probability with objectness score.
	  float cos = detections[bbindex + COS_INDEX * numGridCells];
	  float sin = detections[bbindex + SIN_INDEX * numGridCells];
	  // Add bounding box proposal if above the threshold.
	  if (maxProb > score_threshold_) {
	    const uint32_t strideH = inputH / gridH;
	    const uint32_t strideW = inputW / gridW;
	    addTLRBboxProposal(bx, by, bw, bh, strideH, strideW, maxIndex, maxProb, maxSubIndex, cos, sin, imageW, imageH, inputW, inputH, binfo);
	  }
	}
      }
    }
    return binfo;
  }

  /**
   * @brief Generates keypoints from the output tensors of the neural network.
   * 
   * This function processes the output tensors from a TensorRT model to identify 
   * extracts keypoint information. It clears the existing 
   * keypoints and creates new ones based on the output data, mapping normalized 
   * coordinates to the specified image dimensions.
   * 
   * @param height The height of the image used to scale the keypoint coordinates.
   * @param width The width of the image used to scale the keypoint coordinates.
   */
  void TrtLightnet::makeKeypoint(int height, int width) {
    keypoint_.clear(); // Clear the list of keypoints before processing new data.

    // Calculate the expected channel size for non-bounding box related output tensors.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    // Iterate over all output bindings to identify potential depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Check if the tensor is a depth map based on its dimensions and channel size.
      if (chan_size != chan && outputW == 1 && outputH == 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Verify that the tensor name contains "lgx" and the data type is float.
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) {
	  float *buf = (float *)output_h_.at(i - 1).get(); // Access the tensor data buffer.

	  KeypointInfo key;
	  key.attr_prob = buf[KEY_ATTR]; // Extract the attribute probability.
	  if (static_cast<bool>(buf[KEY_OCC] > 0.9)) {
	    key.isOccluded = true;
	  } else {
	    key.isOccluded = false;
	  }
	  // Map normalized coordinates to the actual image dimensions.
	  key.lx0 = static_cast<int>(buf[KEY_LX0] * (width - 1));
	  key.ly0 = static_cast<int>(buf[KEY_LY0] * (height - 1));
	  key.lx1 = static_cast<int>(buf[KEY_LX1] * (width - 1));
	  key.ly1 = static_cast<int>(buf[KEY_LY1] * (height - 1));
	  key.rx0 = static_cast<int>(buf[KEY_RX0] * (width - 1));
	  key.ry0 = static_cast<int>(buf[KEY_RY0] * (height - 1));
	  key.rx1 = static_cast<int>(buf[KEY_RX1] * (width - 1));
	  key.ry1 = static_cast<int>(buf[KEY_RY1] * (height - 1));

	  key.bot = 0; // Initialize bottom boundary (if needed).
	  key.left = 0; // Initialize left boundary (if needed).

	  keypoint_.push_back(key); // Add the newly created keypoint to the list.
	}
      }
    }
  }
  
  void TrtLightnet::makeTopIndex(void)
  {
    keypoint_.clear(); // Clear the list of keypoints before processing new data.

    // Calculate the expected channel size for non-bounding box related output tensors.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    int index = -1;
    // Iterate over all output bindings to identify potential depth map tensors.
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];

      // Check if the tensor is a depth map based on its dimensions and channel size.
      if (chan_size != chan && outputW == 1 && outputH == 1) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);

	// Verify that the tensor name contains "lgx" and the data type is float.
	if (contain(name, "softmax") && dataType == nvinfer1::DataType::kFLOAT) {
	  float *buf = (float *)output_h_.at(i - 1).get(); // Access the tensor data buffer.
	  float max_value = 0.0;
	  for (int c = 0; c < chan; c++) {
	    if (max_value < buf[c]) {
	      max_value = buf[c];
	      index = c;
	    }	    
	  }
	  max_index_ = index;
	  //std::cout << "(Classification) Max Index :" << max_index_ << "@" << max_value << std::endl;
	}
      }
    }
    return;
  }
  
  /**
   * Returns the list of keypoints detected by the engine.
   * 
   * @return A vector of KeypointInfo containing the keypoints detected by the engine.
   */
  std::vector<KeypointInfo> TrtLightnet::getKeypoints(void)
  {
    return keypoint_;
  }

  int TrtLightnet::getMaxIndex(void)
  {
    return max_index_;
  }
  
  /**
   * @brief Draws keypoints on the given image using predefined colors and shapes.
   * 
   * This function visualizes the keypoints by drawing circles, lines, and rectangles
   * on the provided image, based on the attributes of each keypoint. It adjusts the
   * drawing style depending on the keypoint's attribute probability.
   * 
   * @param img The image on which to draw the keypoints.
   * @param keypoint A vector containing the keypoints to be drawn.
   */
  void TrtLightnet::drawKeypoint(cv::Mat &img, std::vector<KeypointInfo> &keypoint) {
    for (int i = 0; i < (int)keypoint.size(); i++) {
      KeypointInfo key = keypoint[i];
      cv::Point pt_l0, pt_l1;
      cv::Point pt_r0, pt_r1;
      cv::Point pt_b0, pt_b1;
      //cv::Point pt_t0, pt_t1;      
      float attr = key.attr_prob;
      pt_l0.x = key.lx0;
      pt_l0.y = key.ly0;
      pt_l1.x = key.lx1;
      pt_l1.y = key.ly1;
      pt_r0.x = key.rx0;
      pt_r0.y = key.ry0;
      pt_r1.x = key.rx1;
      pt_r1.y = key.ry1;
      //printf("Attr=%f, l0:[%d,%d], l1:[%d,%d], r0:[%d,%d], r1:[%d,%d]\n", attr, pt_l0.x, pt_l0.y, pt_l1.x, pt_l1.y, pt_r0.x, pt_r0.y, pt_r1.x, pt_r1.y);
      pt_b0.x=pt_l0.x;
      pt_b0.y=key.bot;
      pt_b0.x=pt_l0.x;
      pt_b0.y=key.bot;
      

      pt_b1.x=pt_l1.x;
      pt_b1.y=key.bot;
      cv::circle(img, pt_l0, 8, cv::Scalar(255, 0, 255), cv::FILLED, 4, 0);
      cv::circle(img, pt_l1, 4, cv::Scalar(255, 0, 255), cv::FILLED, 4, 0);
      cv::line(img, pt_l0, pt_l1, cv::Scalar(255, 0, 255), 4, 8, 0);
      cv::line(img, pt_r1, pt_l1, cv::Scalar(255, 255, 255), 4, 8, 0);
      cv::circle(img, pt_r0, 8, cv::Scalar(0, 255, 255), cv::FILLED, 4, 0);
      cv::circle(img, pt_r1, 4, cv::Scalar(0, 255, 255), cv::FILLED, 4, 0);
      cv::line(img, pt_r0, pt_r1, cv::Scalar(0, 255, 255), 4, 8, 0);
      cv::line(img, pt_r0, pt_l0, cv::Scalar(0, 255, 0), 4, 8, 0);

      cv::rectangle(img, pt_b1, pt_r1, cv::Scalar(128, 64, 0), 2, 8, 0);      
      cv::rectangle(img, pt_b0, pt_r0, cv::Scalar(0, 128, 255), 4, 8, 0);
      if (attr < 0.5) {
	cv::line(img, pt_b0, pt_r0, cv::Scalar(0, 128, 255), 1, 4, 0);
	pt_b0.x=pt_r0.x;
	pt_b0.y=key.bot;
	cv::line(img, pt_b0, pt_l0, cv::Scalar(0, 128, 255), 1, 4, 0);
      } else {
	cv::line(img, pt_b1, pt_r1, cv::Scalar(128, 64, 0), 1, 4, 0);
	pt_b1.x=pt_r1.x;
	pt_b1.y=key.bot;
	cv::line(img, pt_b1, pt_l1, cv::Scalar(128, 64, 0), 1, 4, 0);
      }
    }
  }
  
  /**
   * @brief Draws keypoints on the given image using the specified color.
   * 
   * This function visualizes the keypoints using lines and rectangles, allowing
   * the user to specify a custom color for all drawn elements.
   * 
   * @param img The image on which to draw the keypoints.
   * @param keypoint A vector containing the keypoints to be drawn.
   * @param color The color used for drawing the keypoints.
   */
  void TrtLightnet::drawKeypoint(cv::Mat &img, std::vector<KeypointInfo> &keypoint, cv::Scalar color) {
    for (int i = 0; i < (int)keypoint.size(); i++) {
      KeypointInfo key = keypoint[i];
      cv::Point pt_l0, pt_l1;
      cv::Point pt_r0, pt_r1;
      cv::Point pt_b0, pt_b1;
      float attr = key.attr_prob;
      pt_l0.x = key.lx0;
      pt_l0.y = key.ly0;
      pt_l1.x = key.lx1;
      pt_l1.y = key.ly1;
      pt_r0.x = key.rx0;
      pt_r0.y = key.ry0;
      pt_r1.x = key.rx1;
      pt_r1.y = key.ry1;
      pt_b0.x=pt_l0.x;
      pt_b0.y=key.bot;

      pt_b1.x=pt_l1.x;
      //      pt_b1.x=pt_l1.x;
      pt_b1.x=pt_r0.x;            
      pt_b1.y=key.bot;
      cv::line(img, pt_l0, pt_l1, color, 4, 8, 0);
      cv::line(img, pt_r1, pt_l1, color, 4, 8, 0);
      cv::line(img, pt_r0, pt_r1, color, 4, 8, 0);
      cv::line(img, pt_r0, pt_l0, color, 4, 8, 0);

      //cv::rectangle(img, pt_b0, pt_r0, color, 4, 8, 0);
      cv::line(img, pt_b0, pt_l0, color, 4, 8, 0);
      cv::line(img, pt_b1, pt_r0, color, 4, 8, 0); 

      cv::line(img, pt_b0, pt_b1, color, 4, 8, 0);
      if (attr < 0.5) {
	cv::line(img, pt_b0, pt_r0, color, 1, 4, 0);
	pt_b0.x=pt_r0.x;
	pt_b0.y=key.bot;
	cv::line(img, pt_b0, pt_l0, color, 1, 4, 0);
      } else {
	cv::line(img, pt_b1, pt_r1, color, 1, 4, 0);
	pt_b1.x=pt_r1.x;
	pt_b1.y=key.bot;
	cv::line(img, pt_b1, pt_l1, color, 1, 4, 0);
      }
      pt_b0.x=pt_l1.x;
      cv::line(img, pt_b0, pt_l1, color, 2, 8, 0);
      pt_b1.x=pt_l0.x;
      pt_b1.y=key.bot;
      cv::line(img, pt_b0, pt_b1, color, 2, 8, 0);
      
      pt_b0.x=pt_r0.x;
      pt_b0.y=key.bot;
      pt_b1.x=pt_r1.x;
      pt_b1.y=key.bot;
      cv::line(img, pt_b0, pt_b1, color, 4, 8, 0);
      cv::line(img, pt_b1, pt_r1, color, 2, 8, 0);

#ifdef DEBUG      
      if (key.isOccluded) {
	cv::putText(img, "occluded", cv::Point((pt_l0.x+pt_r0.x)/2, (pt_l1.y-16)), 0, 0.5, cv::Scalar(0, 0, 255), 8);	  
      } else {
      }
#endif      
    }
  } 

  /**
   * @brief Links a list of keypoints to a bounding box at the specified index.
   * 
   * This function associates the provided keypoints with a bounding box identified 
   * by the given index, allowing the bounding box to store and reference keypoint data.
   * 
   * @param keypoint A vector containing the keypoints to be linked.
   * @param bb_index The index of the bounding box in the bounding box list to which the keypoints should be linked.
   */
  void TrtLightnet::linkKeyPoint(std::vector<KeypointInfo> &keypoint, int bb_index) {
    bbox_[bb_index].keypoint = keypoint;
  }

  /**
   * @brief Clears the list of stored keypoints.
   * 
   * This function empties the keypoint list, removing all previously stored keypoints.
   * It is useful for resetting or reinitializing keypoint data before processing new data.
   */
  void TrtLightnet::clearKeypoint() {
    keypoint_.clear();
  }

  /**
   * @brief Writes segmentation annotations to a JSON file.
   *
   * This function processes segmentation output from a TensorRT model,
   * generates binary masks for each class, extracts polygon contours,
   * and saves the annotations in JSON format.
   *
   * @param json_path Path to the output JSON file.
   * @param image_name Name of the image being processed.
   * @param width Width of the original image.
   * @param height Height of the original image.
   * @param colormap Vector of Colormap objects mapping class indices to names.
   */  
  void TrtLightnet::writeSegmentationAnnotation(const std::string json_path, const std::string image_name, int width, int height, std::vector<Colormap> colormap)
  {
    json::object_t imageAnnotationsOrdered;
    imageAnnotationsOrdered["name"] = image_name;
    imageAnnotationsOrdered["width"] = width;
    imageAnnotationsOrdered["height"] = height;  
    imageAnnotationsOrdered["annotations"] = json::array();    
    int chan_size = (4 + 1 + num_class_) * num_anchor_;
    
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      const int chan = dims.d[1];
      // Identifying tensors by channel size and name for segmentation masks.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "argmax")) { // Check if tensor name contains "argmax".
	  const int *argmax = (int *)(output_h_.at(i - 1).get());
	  
	  for (int c = 1; c < (int)colormap.size(); c++) {
	    cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	    cv::Mat resized;	    
	    for (int x = 0; x < outputW; x++) {
	      for (int y = 0; y < outputH; y++) {
		const int id = argmax[static_cast<int>(x + outputW * y)];
		if (id == c) {
		  mask.at<uchar>(y, x) = 255;
		}
	      }	      
	    }
	    cv::resize(mask, resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
	    std::vector<std::vector<cv::Point>> contours = get_polygons(resized);

	    for (size_t i = 0; i < contours.size(); ++i) {
	      json::object_t annotationOrdered;      
	      annotationOrdered["type"] = "segmentation";
	      annotationOrdered["title"] = colormap[c].name;
	      annotationOrdered["value"] = colormap[c].name;	      
	      json contourJson;
	      json pointsArray = json::array();
	      for (size_t j = 0; j < contours[i].size(); ++j) {
		pointsArray.push_back(contours[i][j].x);
		pointsArray.push_back(contours[i][j].y);
	      }
	      annotationOrdered["points"].push_back({pointsArray});
	      imageAnnotationsOrdered["annotations"].push_back(annotationOrdered);
	    }
	  }
	}
	json jsonData = json::array();  
	jsonData.push_back(imageAnnotationsOrdered);
	write_json_with_order(jsonData, json_path);
      }
    }
  }
  /*
  std::string TrtLightnet::getSegmentationAnnotationStr(const std::string image_name, int width, int height, std::vector<Colormap> colormap)
  {
      
    json::object_t imageAnnotationsOrdered;
    imageAnnotationsOrdered["name"] = image_name;
    imageAnnotationsOrdered["width"] = width;
    imageAnnotationsOrdered["height"] = height;  
    imageAnnotationsOrdered["annotations"] = json::array();    
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    
    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      const int chan = dims.d[1];
      // Identifying tensors by channel size and name for segmentation masks.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "argmax")) { // Check if tensor name contains "argmax".
	  const int *argmax = (int *)(output_h_.at(i - 1).get());
	  
	  for (int c = 1; c < (int)colormap.size(); c++) {
	    cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	    cv::Mat resized;	    
	    for (int x = 0; x < outputW; x++) {
	      for (int y = 0; y < outputH; y++) {
		const int id = argmax[static_cast<int>(x + outputW * y)];
		if (id == c) {
		  mask.at<uchar>(y, x) = 255;
		}
	      }	      
	    }
	    cv::resize(mask, resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
	    std::vector<std::vector<cv::Point>> contours = get_polygons(resized);

	    for (size_t i = 0; i < contours.size(); ++i) {
	      json::object_t annotationOrdered;      
	      annotationOrdered["type"] = "segmentation";
	      annotationOrdered["title"] = colormap[c].name;
	      annotationOrdered["value"] = colormap[c].name;	      
	      json contourJson;
	      json pointsArray = json::array();
	      for (size_t j = 0; j < contours[i].size(); ++j) {
		pointsArray.push_back(contours[i][j].x);
		pointsArray.push_back(contours[i][j].y);
	      }
	      annotationOrdered["points"].push_back({pointsArray});	      
	      imageAnnotationsOrdered["annotations"].push_back(annotationOrdered);	      
	    }
	  }
	}

      }
    }
    std::string json_string = json(imageAnnotationsOrdered).dump();

    return json_string;
  }  
  */  
 
  std::string TrtLightnet::getSegmentationAnnotationStr(
							const std::string image_name, int width, int height, std::vector<Colormap> colormap)
  {
    json::object_t imageAnnotationsOrdered;
    imageAnnotationsOrdered["name"] = image_name;
    imageAnnotationsOrdered["width"] = width;
    imageAnnotationsOrdered["height"] = height;
    imageAnnotationsOrdered["annotations"] = json::array();
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      const int outputW = dims.d[3];
      const int outputH = dims.d[2];
      const int chan = dims.d[1];

      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "argmax")) {
	  const int *argmax = (int *)(output_h_.at(i - 1).get());

	  std::vector<std::vector<json::object_t>> annotations(colormap.size());

	  const double scaleX = static_cast<double>(width) / outputW;
	  const double scaleY = static_cast<double>(height) / outputH;

#pragma omp parallel for num_threads(2)
	  for (int c = 1; c < (int)colormap.size(); c++) {
	    cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC1);

	    for (int y = 0; y < outputH; y++) {
	      uchar* row = mask.ptr<uchar>(y); 
	      for (int x = 0; x < outputW; x++) {
		const int id = argmax[static_cast<int>(x + outputW * y)];
		if (id == c) {
		  row[x] = 255;
		}
	      }
	    }
	    
	    std::vector<std::vector<cv::Point>> contours = get_polygons(mask);

	    std::vector<json::object_t> thread_annotations;

	    for (size_t i = 0; i < contours.size(); ++i) {
	      json::object_t annotationOrdered;
	      annotationOrdered["type"] = "segmentation";
	      annotationOrdered["title"] = colormap[c].name;
	      annotationOrdered["value"] = colormap[c].name;

	      json pointsArray = json::array();
	      for (size_t j = 0; j < contours[i].size(); ++j) {
		int scaled_x = static_cast<int>(contours[i][j].x * scaleX);
		int scaled_y = static_cast<int>(contours[i][j].y * scaleY);
		pointsArray.push_back(scaled_x);
		pointsArray.push_back(scaled_y);
	      }
	      annotationOrdered["points"].push_back({pointsArray});

	      thread_annotations.push_back(annotationOrdered);
	    }

	    annotations[c] = std::move(thread_annotations);
	  }

	  for (const auto &ann_set : annotations) {
	    for (const auto &annotation : ann_set) {
	      if (!annotation.empty()) {
		imageAnnotationsOrdered["annotations"].push_back(annotation);
	      }
	    }
	  }
	}
      }
    }

    std::string json_string = json(imageAnnotationsOrdered).dump();
    return json_string;
  }
  
  
  /**
   * @brief Retrieves the input size of the TensorRT model.
   *
   * This function extracts the input dimensions of the TensorRT model's first binding tensor.
   *
   * @param batch Pointer to store batch size.
   * @param chan Pointer to store the number of channels.
   * @param height Pointer to store the height.
   * @param width Pointer to store the width.
   */  
  void TrtLightnet::getInputSize(int *batch, int *chan, int *height, int *width) {
    if (!trt_common_) {
      std::cerr << "Error: trt_common_ is not initialized." << std::endl;
      return;
    }
    auto input_dims = trt_common_->getBindingDimensions(0);
    *batch = static_cast<float>(input_dims.d[0]);
    *chan = static_cast<float>(input_dims.d[1]);
    *height = static_cast<float>(input_dims.d[2]);
    *width = static_cast<float>(input_dims.d[3]);
  }
  
}  // namespace tensorrt_lightnet


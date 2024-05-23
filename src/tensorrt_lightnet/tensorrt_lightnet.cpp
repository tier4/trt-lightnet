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
  TrtLightnet::TrtLightnet(ModelConfig &model_config, InferenceConfig &inference_config, tensorrt_common::BuildConfig build_config)
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

      trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
								 model_path, precision, std::move(calibrator), batch_config, max_workspace_size, build_config);
    } else {
      trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
								 model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
    }

    trt_common_->setup();
    if (!trt_common_->isInitialized()) {
      throw std::runtime_error("TensorRT engine initialization failed.");
    }

    // Initialize class members
    num_class_ = num_class;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    anchors_ = anchors;
    num_anchor_ = num_anchor;

    // Allocate GPU memory for inputs and outputs based on tensor dimensions.
    allocateMemory();
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

    // Normalize images and convert to blob directly without additional copying.
    float scale = 1 / 255.0;
    const auto nchw_images = cv::dnn::blobFromImages(images, scale, cv::Size(inputW, inputH), cv::Scalar(0.0, 0.0, 0.0), true);   

    // If the data is continuous, we can use it directly. Otherwise, we need to clone it for contiguous memory.
    input_h_ = nchw_images.isContinuous() ? nchw_images.reshape(1, nchw_images.total()) : nchw_images.reshape(1, nchw_images.total()).clone();
    // Ensure the input device buffer is allocated with the correct size and copy the data.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice, *stream_));    
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

      if (!names.empty()) {
	stream << std::fixed << std::setprecision(2) << names[id] << "  " << bbi.prob;
      } else {
	stream << std::fixed << std::setprecision(2) << "id:" << id << "  score:" << bbi.prob;
      }

      cv::Scalar color = colormap.empty() ? cv::Scalar(255, 0, 0) : cv::Scalar(colormap[id][2], colormap[id][1], colormap[id][0]);
      cv::rectangle(img, cv::Point(bbi.box.x1, bbi.box.y1), cv::Point(bbi.box.x2, bbi.box.y2), color, 2);
      cv::putText(img, stream.str(), cv::Point(bbi.box.x1, bbi.box.y1 - 5), 0, 0.5, color, 1);
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
      }
    }

    bbox_ = nonMaximumSuppression(nms_threshold_, bbox_); // Apply NMS and return the filtered bounding boxes.
    //    bbox_ = nmsAllClasses(nms_threshold_, bbox_, num_class_); // Apply NMS and return the filtered bounding boxes.    
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

  /**
   * Generates depth maps from the network's output tensors that are not related to bounding box detections.
   * The method identifies specific tensors for depth map generation based on channel size and name.
   */
  void TrtLightnet::makeDepthmap(void)
  {
    depthmaps_.clear();
    // Formula to identify output tensors not related to bounding box detections.
    int chan_size = (4 + 1 + num_class_) * num_anchor_;

    for (int i = 1; i < trt_common_->getNbBindings(); i++) {
      const auto dims = trt_common_->getBindingDimensions(i);
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];
      // Identifying tensors by channel size and name for depthmap.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	nvinfer1::DataType dataType = trt_common_->getBindingDataType(i);
	if (contain(name, "lgx") && dataType == nvinfer1::DataType::kFLOAT) { // Check if tensor name contains "lgx" and tensor type is 'kFLOAT'.
	  cv::Mat depthmap = cv::Mat::zeros(outputH, outputW, CV_8UC1);
	  float *buf = (float *)output_h_.at(i-1).get();
	  for (int y = 0; y < outputH; y++) {
	    int stride = outputW * y;

	    for (int x = 0; x < outputW; x++) {
	      float rel = 1.0 - buf[stride + x];
	      int value = (int)(rel * 255);
	      depthmap.at<unsigned char>(y, x) = value;
	    }
	  }
	  depthmaps_.push_back(depthmap);
	}
      }
    }
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
      int outputW = dims.d[3];
      int outputH = dims.d[2];
      int chan = dims.d[1];
      // Identifying tensors by channel size and name for segmentation masks.      
      if (chan_size != chan) {
	std::string name = trt_common_->getIOTensorName(i);
	if (contain(name, "argmax")) { // Check if tensor name contains "argmax".
	  cv::Mat mask = cv::Mat::zeros(outputH, outputW, CV_8UC3);
	  int *buf = (int *)output_h_.at(i-1).get();

	  for (int y = 0; y < outputH; y++) {
	    int stride = outputW * y;
	    cv::Vec3b *ptr = mask.ptr<cv::Vec3b>(y);

	    for (int x = 0; x < outputW; x++) {
	      int id = buf[stride + x];
	      ptr[x] = argmax2bgr[id]; // Mapping class index to color.
	    }
	  }
	  masks_.push_back(mask);
	}
      }
    }
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
}  // namespace tensorrt_lightnet

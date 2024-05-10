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

#ifndef TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_
#define TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <tensorrt_common/tensorrt_common.hpp>
#include <tensorrt_lightnet/preprocess.hpp>
#include <tensorrt_lightnet/calibrator.hpp>
#include <memory>
#include <string>
#include <vector>
#include <NvInfer.h>
/**
 * Checks if a given value is contained within a string. This template function is versatile,
 * allowing for different types of values to be checked against the string, assuming the value
 * can be sensibly searched within a string context.
 *
 * @tparam T The type of the value to be checked. Must be compatible with std::string's find method.
 * @param s The string to search within.
 * @param v The value to search for within the string. This value is converted to a suitable format
 *          for comparison with the contents of the string.
 * @return true if the value is found within the string, false otherwise.
 */
template<class T> bool contain(const std::string& s, const T& v) {
  return s.find(v) != std::string::npos;
}

namespace tensorrt_lightnet
{
  /**
   * Aliases for CUDA utility wrappers and functions that simplify handling of CUDA memory and streams.
   */
  using cuda_utils::CudaUniquePtr; ///< Wrapper for device memory allocation with automatic cleanup.
  using cuda_utils::CudaUniquePtrHost; ///< Wrapper for host memory allocation with automatic cleanup.
  using cuda_utils::makeCudaStream; ///< Function to create a CUDA stream with automatic cleanup.
  using cuda_utils::StreamUniquePtr; ///< Wrapper for CUDA stream with automatic cleanup.

  /**
   * Represents a bounding box in a 2D space.
   */
  struct BBox
  {
    float x1, y1; ///< Top-left corner of the bounding box.
    float x2, y2; ///< Bottom-right corner of the bounding box.
  };

  /**
   * Contains information about a detected object, including its bounding box,
   * label, class ID, and the probability of the detection.
   */
  struct BBoxInfo
  {
    BBox box; ///< Bounding box of the detected object.
    int label; ///< Label of the detected object.
    int classId; ///< Class ID of the detected object.
    float prob; ///< Probability of the detection.
  };

  /**
   * Represents a colormap entry, including an ID, a name, and a color.
   * This is used for mapping class IDs to human-readable names and visual representation colors.
   */
  typedef struct Colormap_
  {
    int id; ///< ID of the color map entry.
    std::string name; ///< Human-readable name associated with the ID.
    std::vector<unsigned char> color; ///< Color associated with the ID, typically in RGB format.
  } Colormap;
  
/**
 * @class TrtLightnet
 * @brief TensorRT LIGHTNET for faster inference
 * Entropy calibration.
 */
class TrtLightnet
{
public:

  /**
   * Constructs a TrtLightnet object for performing inference with a TensorRT engine.
   * This constructor initializes the object with various configuration parameters for model execution.
   * 
   * @param model_path The file path to the serialized model.
   * @param precision The precision mode for TensorRT execution (e.g., "FP32", "FP16", "INT8").
   * @param num_class The number of classes that the model predicts (default is 8).
   * @param score_threshold The threshold for filtering out predictions with low confidence scores (default is 0.3).
   * @param nms_threshold The threshold for the Non-Maximum Suppression (NMS) algorithm (default is 0.7).
   * @param anchors A list of anchor sizes for the model (default includes common anchor sizes).
   * @param num_anchor The number of anchors to use (default is 3).
   * @param build_config Configuration options for building the TensorRT engine.
   * @param use_gpu_preprocess A flag indicating whether to use GPU for preprocessing tasks (default is false).
   * @param calibration_image_list_file The file path to a list of images used for INT8 calibration (default is an empty string).
   * @param norm_factor A normalization factor applied to the input images (default is 1.0).
   * @param cache_dir A directory for caching dynamic library files for re-use (default is an empty string). 
   *                  Marked as maybe_unused to indicate it may not be used in all configurations.
   * @param batch_config Configuration for batch size management during inference (default is {1, 1, 1}).
   * @param max_workspace_size The maximum workspace size for TensorRT (default is 1GB).
   */
  TrtLightnet(const std::string &model_path, const std::string &precision, const int num_class = 8,
	      const float score_threshold = 0.3, const float nms_threshold = 0.7,
	      const std::vector<int> anchors = {9, 15,  22, 36,  49, 54,  33,129,  90,101,  83,348, 165,186, 227,463, 703,613}, int num_anchor = 3,
	      const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
	      const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
	      const double norm_factor = 1.0, [[maybe_unused]] const std::string &cache_dir = "",
	      const tensorrt_common::BatchConfig &batch_config = {1, 1, 1},
	      const size_t max_workspace_size = (1 << 30));

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
  std::unique_ptr<nvinfer1::IInt8Calibrator> initializeCalibrator(
								  const tensorrt_common::BuildConfig& build_config,
								  ImageStream& stream,
								  const fs::path& calibration_table_path,
								  const fs::path& histogram_table_path,
								  double norm_factor);

  /**
   * Allocates memory for the input and output tensors based on the binding dimensions of the network.
   */  
  void allocateMemory();
  
  /**
   * Destructor for the TrtLightnet object.
   * Cleans up resources allocated during the lifecycle of the TrtLightnet object,
   * ensuring proper release of memory and device resources.
   */
  ~TrtLightnet();
  
  /**
   * Performs inference on the current data.
   * @return true if inference was successful, false otherwise.
   */
  bool doInference(void);

  /**
   * Alias for doInference, performs inference on the current data.
   * @return true if inference was successful, false otherwise.
   */
  bool infer(void);

  /**
   * Draws bounding boxes on an image.
   * @param img the image on which to draw bounding boxes.
   * @param bboxes the bounding boxes to draw.
   * @param colormap the colors for each class.
   * @param names the names of the classes.
   */
  void drawBbox(cv::Mat &img, std::vector<BBoxInfo> bboxes, std::vector<std::vector<int>> &colormap, std::vector<std::string> names);

  /**
   * Decodes the output tensor into bounding box information.
   * @param imageIdx index of the image being processed.
   * @param imageH height of the image.
   * @param imageW width of the image.
   * @param inputH height of the input tensor.
   * @param inputW width of the input tensor.
   * @param anchor array of anchor sizes.
   * @param anchor_num number of anchors.
   * @param output output tensor from the network.
   * @param gridW width of the grid.
   * @param gridH height of the grid.
   * @return vector of bounding box information.
   */
  std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW, const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH);

  /**
   * Applies non-maximum suppression to filter overlapping bounding boxes.
   * @param nmsThresh threshold for suppression.
   * @param binfo vector of bounding box information before suppression.
   * @return vector of bounding box information after suppression.
   */
  std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);

  /**
   * Applies NMS on all classes.
   * @param nmsThresh threshold for suppression.
   * @param binfo vector of bounding box information before suppression.
   * @param numClasses number of classes.
   * @return vector of bounding box information after suppression.
   */
  std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo, const uint32_t numClasses);

  /**
   * Converts bounding box results from the network's output format.
   * @param bx x-coordinate of the center of the box.
   * @param by y-coordinate of the center of the box.
   * @param bw width of the box.
   * @param bh height of the box.
   * @param stride_h_ stride height.
   * @param stride_w_ stride width.
   * @param netW network width.
   * @param netH network height.
   * @return converted bounding box.
   */
  BBox convertBboxRes(const float& bx, const float& by, const float& bw, const float& bh, const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH);

  /**
   * Adds a bounding box proposal.
   * @param bx x-coordinate of the center of the box.
   * @param by y-coordinate of the center of the box.
   * @param bw width of the box.
   * @param bh height of the box.
   * @param stride_h_ stride height.
   * @param stride_w_ stride width.
   * @param maxIndex class with the highest probability.
   * @param maxProb maximum probability.
   * @param image_w image width.
   * @param image_h image height.
   * @param input_w input width.
   * @param input_h input height.
   * @param binfo vector to add bounding box information to.
   */
  void addBboxProposal(const float bx, const float by, const float bw, const float bh, const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb, const uint32_t image_w, const uint32_t image_h, const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo);

  /**
   * Preprocesses the input images before feeding them to the network.
   * @param images vector of images to preprocess.
   */
  void preprocess(const std::vector<cv::Mat> & images);

  /**
   * Retrieves bounding boxes for a given image size.
   * @param imageH height of the image.
   * @param imageW width of the image.
   */
  void makeBbox(const int imageH, const int imageW);

  /**
   * Return BBox.
   * 
   * @return A vector of BBoxInfo containing the detected bounding boxes.
   */
  std::vector<BBoxInfo> getBbox();  

  /**
   * Generates depth maps from the network's output tensors that are not related to bounding box detections.
   * The method identifies specific tensors for depth map generation based on channel size and name.
   */
  void makeDepthmap(void);

  /**
   * Retrieves the generated depth maps.
   * 
   * @return A vector of cv::Mat, each representing a depth map for an input image.
   */    
  std::vector<cv::Mat> getDepthmap(void);
  
  /**
   * Converts the segmentation output to mask images.
   * Each pixel in the mask image is colored based on the class it belongs to,
   * using the provided mapping from classes to colors.
   * 
   * @param argmax2bgr A vector containing the mapping from segmentation classes to their corresponding colors in BGR format.
   */
  void makeMask(std::vector<cv::Vec3b> &argmax2bgr);

  /**
   * Return mask.
   * 
   * @return A vector of OpenCV Mat objects, each representing a mask image where each pixel's color corresponds to its class's color.
   */
  std::vector<cv::Mat> getMask(void);

  /**
   * Clears the detected bounding boxes specifically from the subnet.
   */
  void clearSubnetBbox();

  /**
   * Appends a vector of detected bounding boxes to the existing list of bounding boxes from the subnet.
   * 
   * @param bb A vector of BBoxInfo that contains bounding boxes to be appended.
   */  
  void appendSubnetBbox(std::vector<BBoxInfo> bb);

  /**
   * Returns the list of bounding boxes detected by the subnet.
   * 
   * @return A vector of BBoxInfo containing the bounding boxes detected by the subnet.
   */
  std::vector<BBoxInfo> getSubnetBbox();
  
  /**
   * Prints the profiling information of the inference process, detailing the performance across each layer of the neural network.
   * This method provides insights into the time spent on each layer during inference, 
   * allowing for a granular analysis of performance bottlenecks and efficiency.
   * Useful for developers and researchers looking to optimize neural network architectures
   * and inference pipelines for speed and resource usage.
   */
  void printProfiling(void);

  /**
   * Unique pointer to a TensorRT common utility class, encapsulating common TensorRT operations.
   */
  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

  /**
   * Host-side input buffer for the model, typically used for pre-processing input data before inference.
   */
  std::vector<float> input_h_;

  /**
   * Device-side input buffer for the model, used to store the input data on the GPU.
   */
  CudaUniquePtr<float[]> input_d_;

  /**
   * Device-side output buffers for the model, each corresponding to an output tensor of the model.
   */
  std::vector<CudaUniquePtr<float[]>> output_d_;

  /**
   * Host-side output buffers for the model, used to store the results after inference for further processing or analysis.
   */
  std::vector<CudaUniquePtrHost<float[]>> output_h_;

  /**
   * CUDA stream for asynchronous execution of pre-processing, inference, and post-processing, improving throughput.
   */
  StreamUniquePtr stream_{makeCudaStream()};

  /**
   * Number of classes that the model is trained to predict.
   */
  int num_class_;

  /**
   * Threshold for filtering out predictions with a confidence score lower than this value.
   */
  float score_threshold_;

  /**
   * Threshold used by the Non-Maximum Suppression (NMS) algorithm to filter out overlapping bounding boxes.
   */
  float nms_threshold_;

  /**
   * List of anchor dimensions used by the model for detecting objects. Anchors are pre-defined sizes and ratios that the model uses as reference points for object detection.
   */
  std::vector<int> anchors_;

  /**
   * Number of anchors used by the model. This typically corresponds to the size of the `anchors_` vector.
   */
  int num_anchor_;

  /**
   * The size of batches processed by the model during inference. A larger batch size can improve throughput but requires more memory.
   */
  int batch_size_;

  /**
   * Normalization factor applied to the input images before passing them to the model. This is used to scale the pixel values to a range the model expects.
   */
  double norm_factor_;

  /**
   * Width of the source images before any preprocessing. This is used to revert any scaling or transformations for visualization or further processing.
   */
  int src_width_;

  /**
   * Height of the source images before any preprocessing. Similar to `src_width_`, used for reverting transformations.
   */
  int src_height_;

  /**
   * Flag indicating whether the model performs multiple tasks beyond object detection, such as segmentation or classification.
   */
  int multitask_;

  /**
   * Stores bounding boxes detected by the primary network.
   */  
  std::vector<BBoxInfo> bbox_;

  /**
   * Stores mask images for each detected object, typically used in segmentation tasks.
   */
  std::vector<cv::Mat> masks_;

  /**
   * Stores depth maps generated from the network's output, providing depth information for each pixel.
   */
  std::vector<cv::Mat> depthmaps_;

  /**
   * Stores bounding boxes detected by the subnet, allowing for specialized processing on selected detections.
   */
  std::vector<BBoxInfo> subnet_bbox_;
};

}  // namespace tensorrt_lightnet

#endif  // TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_

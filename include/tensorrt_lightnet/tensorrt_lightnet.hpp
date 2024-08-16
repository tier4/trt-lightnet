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


enum TLR_Color {
  TLR_GREEN = 0,
  TLR_YELLOW = 1,
  TLR_RED = 2,
};

/**
 *  Tensor Index for TLR
 */
enum TLR_Index {
  X_INDEX = 0, //bbox
  Y_INDEX = 1, //bbox
  W_INDEX = 2, //bbox
  H_INDEX = 3, //bbox
  OBJ_INDEX = 4, //confidence
  R_INDEX = 5, //color
  G_INDEX = 6, //color
  B_INDEX = 7, //color
  CIRCLE_INDEX = 8, //typ
  ARROW_INDEX = 9, //type
  UTURN_INDEX = 10, //typ
  PED_INDEX = 11, //type
  NUM_INDEX = 12, //type
  CROSS_INDEX = 13, //type
  COS_INDEX = 14, //angle
  SIN_INDEX = 15, //angle
};


/**
 * Configuration settings related to the model being used for inference.
 * Includes paths, classification thresholds, and anchor configurations.
 */
struct ModelConfig {
  std::string model_path; ///< Path to the serialized model file.
  int num_class; ///< Number of classes the model can identify.
  float score_threshold; ///< Threshold for classification scores to consider a detection valid.
  std::vector<int> anchors; ///< Anchor sizes for the model.
  int num_anchors; ///< Number of anchors.
  float nms_threshold; ///< Threshold for Non-Maximum Suppression (NMS).  
};

/**
 * Configuration settings for performing inference, including precision and
 * hardware-specific options.
 */
struct InferenceConfig {
  std::string precision; ///< Precision mode for inference (e.g., FP32, FP16).
  bool profile; ///< Flag to enable profiling to measure inference performance.
  bool sparse; ///< Flag to enable sparsity in the model, if supported.
  int dla_core_id; ///< ID of the DLA core to use for inference, if applicable.
  bool use_first_layer; ///< Flag to use the first layer in calculations, typically for INT8 calibration.
  bool use_last_layer; ///< Flag to use the last layer in calculations, affecting performance and accuracy.
  int batch_size; ///< Number of images processed in one inference batch.
  double scale; ///< Scale factor for input image normalization.
  std::string calibration_images; ///< Path to calibration images for INT8 precision mode.
  std::string calibration_type; ///< Type of calibration to use (e.g., entropy).
  tensorrt_common::BatchConfig batch_config; ///< Batch configuration for inference.
  size_t workspace_size; ///< Maximum workspace size for TensorRT.
};

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
    bool isHierarchical;
    int subClassId;
    float sin;
    float cos;
  };

  template <typename ... Args>
  static std::string format(const std::string& fmt, Args ... args )
  {
    size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
    return std::string(&buf[0], &buf[0] + len);
  }
  
  inline std::string getTLRStringFromBBox(BBoxInfo bbi, std::vector<std::string> &names)
  {
    std::string c_str = "";
    float deg = -360.0;
    if (bbi.subClassId == TLR_GREEN) {
      c_str = "green";
    } else if (bbi.subClassId == TLR_YELLOW) {
      c_str = "yellow";
    } else {
      c_str = "red";
    }
    if (names[bbi.classId] == "arrow") {
      float sin = bbi.sin;
      float cos = bbi.cos;
      deg = atan2(sin, cos) * 180.0 / M_PI;
    }
    std::string str = format("%s %f %d %d %d %d %s %f", c_str.c_str(), (float)bbi.prob, (int)bbi.box.x1, (int)bbi.box.y1, (int)bbi.box.x2, (int)bbi.box.y2, names[bbi.classId].c_str(), deg);
    return str;
  }

  
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
  TrtLightnet(ModelConfig &model_config, InferenceConfig &inference_config, tensorrt_common::BuildConfig build_config);

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
  void addTLRBboxProposal(const float bx, const float by, const float bw, const float bh,
						const uint32_t stride_h_, const uint32_t stride_w_, const int maxIndex, const float maxProb, const int maxSubIndex,
				   const float cos, const float sin,
				    const uint32_t image_w, const uint32_t image_h,
				   const uint32_t input_w, const uint32_t input_h, std::vector<BBoxInfo>& binfo);
  
  std::vector<BBoxInfo> decodeTLRTensor(const int imageIdx, const int imageH, const int imageW,  const int inputH, const int inputW, const int *anchor, const int anchor_num, const float *output, const int gridW, const int gridH);
  
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
  void applyArgmax(cv::Mat &mask, const float *buf, const int chan, const int outputH, const int outputW, std::vector<cv::Vec3b> &argmax2bgr);


  /**
   * @brief This function calculates the entropy maps from the softmax output of the network.
   * It identifies the tensors that are not related to bounding box detections and processes 
   * the tensors whose names contain "softmax". The function computes the entropy for each 
   * channel and stores the entropy maps.
   */
  void calcEntropyFromSoftmax(void);

  /**
   * @brief This function returns the calculated entropy maps.
   * 
   * @return A vector of cv::Mat objects representing the entropy maps.
   */
  std::vector<cv::Mat> getEntropymaps(void);

  /**
   * @brief This function returns the calculated entropies for each channel.
   * 
   * @return A vector of vectors, where each inner vector contains the entropies for a particular tensor.
   */
  std::vector<std::vector<float>> getEntropies(void);
  
  /**
   * Return mask.
   * 
   * @return A vector of OpenCV Mat objects, each representing a mask image where each pixel's color corresponds to its class's color.
   */
  std::vector<cv::Mat> getMask(void);

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
  std::vector<float*> getDebugTensors(std::vector<nvinfer1::Dims> &dim_infos, std::vector<std::string> &debug_names);
  
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
   * Apply NMS for subnet BBox
   */
  void doNonMaximumSuppressionForSubnetBbox();
  
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


  /**
   * Stores entropies, allowing for uncertainty estimation using a single DNN.
   */  
  std::vector<std::vector<float>> entropies_;

  /**
   * Stores entropy maps for visualization
   */    
  std::vector<cv::Mat> ent_maps_;  
};

}  // namespace tensorrt_lightnet

#endif  // TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_

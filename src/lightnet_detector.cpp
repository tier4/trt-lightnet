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

#include <memory>
#include <string>
#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <config_parser.h>

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
 * Configuration for input and output paths used during inference.
 * This includes directories for images, videos, and where to save output.
 */
struct PathConfig {
  std::string directory; ///< Directory containing images for inference.
  std::string video_path; ///< Path to a video file for inference.
  int camera_id; ///< Camera device ID for live inference.
  std::string dump_path; ///< Path for dumping intermediate data or debug info.
  std::string output_path; ///< Path to save inference output.
  bool flg_save; ///< Flag indicating whether to save inference output.
  std::string save_path; ///< Directory path where output should be saved.
};

/**
 * Configuration for visualization settings, including whether to show output
 * and how to colorize different aspects of the output.
 */
struct VisualizationConfig {
  bool dont_show; ///< Flag indicating whether to suppress displaying the output window.
  std::vector<std::vector<int>> colormap; ///< Color mapping for classes in bounding boxes.
  std::vector<std::string> names; ///< Names of the classes for display.
  std::vector<cv::Vec3b> argmax2bgr; ///< Mapping from class indices to BGR colors for segmentation masks.
};

/**
 * Replaces the first occurrence of a substring within a string with another substring.
 * If the substring to replace is not found, the original string is returned unchanged.
 * 
 * @param replacedStr The string to search and replace in.
 * @param from The substring to search for.
 * @param to The substring to replace with.
 * @return The modified string if the substring was found and replaced; otherwise, the original string.
 */
std::string replaceOtherStr(std::string &replacedStr, const std::string &from, const std::string &to) {
  size_t start_pos = replacedStr.find(from);
  if(start_pos == std::string::npos)
    return replacedStr; // No match found
  replacedStr.replace(start_pos, from.length(), to);
  return replacedStr;
}

/**
 * Saves an image to a specified directory with a given filename.
 * The path is constructed by combining the directory and filename.
 * A message is printed to the console indicating the save location.
 * 
 * @param img The image (cv::Mat) to save.
 * @param dir The directory where the image should be saved.
 * @param name The filename to use when saving the image.
 */
void save_image(cv::Mat &img, const std::string &dir, const std::string &name)
{
  fs::path p = fs::path(dir) / name; // Use / operator for path concatenation
  std::cout << "## Save " << p << std::endl;
  cv::imwrite(p.string(), img);
}

/**
 * Converts a vector of colormap entries to a vector of BGR color values.
 * This function is typically used for mapping class indices to specific colors
 * for visualization purposes.
 * 
 * @param colormap A vector of colormap entries (each with an ID, name, and color).
 * @return A vector of OpenCV BGR color values corresponding to the colormap entries.
 */
std::vector<cv::Vec3b> getArgmaxToBgr(const std::vector<tensorrt_lightnet::Colormap> colormap)
{
  std::vector<cv::Vec3b> argmax2bgr;
  for (const auto &map : colormap) {
    argmax2bgr.emplace_back(cv::Vec3b(map.color[2], map.color[1], map.color[0]));
  }
  return argmax2bgr;
}  

/**
 * Performs inference on an image using a TensorRT LightNet model, processes the output to draw bounding boxes
 * and applies segmentation masks. This function encapsulates the preprocessing, inference, and postprocessing steps.
 * 
 * @param trt_lightnet A shared pointer to an initialized TensorRT LightNet model.
 * @param image The image to process.
 * @param colormap The color mapping for class labels used in bounding box visualization.
 * @param names The names of the classes corresponding to the detection outputs.
 * @param argmax2bgr A vector of BGR colors used for drawing segmentation masks.
 */
void inferLightnet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<std::vector<int>> &colormap, std::vector<std::string> &names, std::vector<cv::Vec3b> &argmax2bgr)
{
  trt_lightnet->preprocess({image});
  trt_lightnet->doInference();
  auto bbox = trt_lightnet->getBbox(image.rows, image.cols);
  trt_lightnet->drawBbox(image, bbox, colormap, names);
  auto masks = trt_lightnet->getMask(argmax2bgr);
  for (const auto &mask : masks) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
    cv::addWeighted(image, 1.0, resized, 0.5, 0.0, image);
  }
}

int
main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ModelConfig model_config = {
    .model_path = get_onnx_path(),
    .num_class = get_classes(),
    .score_threshold = static_cast<float>(get_score_thresh()),
    .anchors = get_anchors(),
    .num_anchors = get_num_anchors(),
    .nms_threshold = 0.45f // Assuming this value is fixed or retrieved similarly.
  };

  InferenceConfig inference_config = {
    .precision = get_precision(),    
    .profile = get_prof_flg(),
    .sparse = get_sparse_flg(),
    .dla_core_id = get_dla_id(),
    .use_first_layer = get_first_flg(),
    .use_last_layer = get_last_flg(),
    .batch_size = get_batch_size(),
    .scale = 1.0, // Assuming a fixed value or obtained similarly.
    .calibration_images = get_calibration_images(),
    .calibration_type = get_calib_type(),    
    .batch_config = {1, get_batch_size()/2, get_batch_size()},
    .workspace_size = (1 << 30)
  };
  
  PathConfig path_config = {
    .directory = get_directory_path(),
    .video_path = get_video_path(),
    .camera_id = get_camera_id(),
    .dump_path = get_dump_path(),
    .output_path = get_output_path(),
    .flg_save = getSaveDetections(),
    .save_path = getSaveDetectionsPath()
  };

  VisualizationConfig visualization_config = {
    .dont_show = is_dont_show(),
    .colormap = get_colormap(),
    .names = get_names(),
    .argmax2bgr = getArgmaxToBgr(get_seg_colormap())
  };
  
  
  int count = 0;
  // File saving flag and path from PathConfig.
  bool flg_save = path_config.flg_save;
  std::string save_path = path_config.save_path;

  if (flg_save) {
    fs::create_directories(fs::path(save_path) / "detections");
    fs::create_directories(fs::path(save_path) / "segmentations");
  }

  tensorrt_common::BuildConfig build_config(
					    inference_config.calibration_type,
					    inference_config.dla_core_id,
					    inference_config.use_first_layer,
					    inference_config.use_last_layer,
					    inference_config.profile,
					    0.0, // Assuming a fixed value for missing float parameter, might need adjustment.
					        inference_config.sparse
					    );

  // Initialize TensorRT LightNet using parameters from ModelConfig and InferenceConfig.
  auto trt_lightnet = std::make_shared<tensorrt_lightnet::TrtLightnet>(
								       model_config.model_path,
								       inference_config.precision,
								       model_config.num_class,
								       model_config.score_threshold,
								       model_config.nms_threshold,
								       model_config.anchors,
								       model_config.num_anchors,
								       build_config,
								       false, // Assuming this parameter is fixed or needs to be obtained similarly.
								       inference_config.calibration_images,
								       1.0, // Assuming scale factor is fixed or needs to be obtained from InferenceConfig.
								       "", // Assuming this parameter is fixed or needs clarification.
								       inference_config.batch_config,
								           inference_config.workspace_size
								       );
  
  if (!path_config.directory.empty()) {
    for (const auto& file : fs::directory_iterator(path_config.directory)) {
      std::cout << "Inference from " << file.path() << std::endl;
      cv::Mat image = cv::imread(file.path(), cv::IMREAD_UNCHANGED);
      inferLightnet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, visualization_config.argmax2bgr);
      if (!visualization_config.dont_show) {
	cv::imshow("inference", image);
	if (cv::waitKey(0) == 'q') break;
      }
      if (path_config.flg_save) {
	fs::path p = fs::path(path_config.save_path) / "detections";
	save_image(image, p.string(), file.path().filename());
      }
      count++;
    }
  } else if (!path_config.video_path.empty() || path_config.camera_id != -1) {
    cv::VideoCapture video;
    if (path_config.camera_id != -1) {
      video.open(path_config.camera_id);
    } else {
      video.open(path_config.video_path);
    }
    while (1) {
      cv::Mat image;
      video >> image;
      if (image.empty()) break;
      inferLightnet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, visualization_config.argmax2bgr);
      
      if (!visualization_config.dont_show) {
	cv::imshow("inference", image);
	if (cv::waitKey(1) == 'q') break;
      }
      if (flg_save) {
	std::ostringstream sout;
	sout << std::setfill('0') << std::setw(6) << count;	  
	std::string name = "frame_" + sout.str() + ".jpg";
	fs::path p = fs::path(save_path) / "detections";	
	save_image(image, p.string(), name);
      }
      count++;      
    }
  }  
  
  if (inference_config.profile) {
    trt_lightnet->printProfiling();
  }

  return 0;
}

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
#include <omp.h>
#include <config_parser.h>
#include <cnpy.h>
//#include "cnpy.h"
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
  bool flg_save_debug_tensors;
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

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

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
void saveImage(cv::Mat &img, const std::string &dir, const std::string &name)
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
 * Performs inference on an image using a TensorRT LightNet model, processes the output to get bounding boxes
 * segmentation and depth. This function encapsulates the preprocessing, inference, and postprocessing steps.
 * 
 * @param trt_lightnet A shared pointer to an initialized TensorRT LightNet model.
 * @param image The image to process.
 * @param colormap The color mapping for class labels used in bounding box visualization.
 * @param names The names of the classes corresponding to the detection outputs.
 * @param argmax2bgr A vector of BGR colors used for drawing segmentation masks.
 */
void inferLightnet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<cv::Vec3b> &argmax2bgr)
{
  //preprocess
  trt_lightnet->preprocess({image});
  //inference
  trt_lightnet->doInference();
  //postprocess
#pragma omp parallel sections
  {
#pragma omp section
    {  
      trt_lightnet->makeBbox(image.rows, image.cols);
    }
#pragma omp section
    {  	
      trt_lightnet->makeMask(argmax2bgr);
    }
#pragma omp section
    {	  
      trt_lightnet->makeDepthmap();
    }
  }
}

/**
 * Infers a subnet using a given Lightnet model and refines detections based on target classes.
 * 
 * @param trt_lightnet A shared pointer to the primary TensorRT Lightnet model.
 * @param subnet_trt_lightnet A shared pointer to the secondary TensorRT Lightnet model for processing subnets.
 * @param image The image in which detection is performed.
 * @param names A vector of class names corresponding to class IDs.
 * @param target A vector of target class names to filter the detections.
 */
void inferSubnetLightnets(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> subnet_trt_lightnets, cv::Mat &image, std::vector<std::string> &names, std::vector<std::string> &target, const int numWorks)
{
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();
  trt_lightnet->clearSubnetBbox();
  int num = (int)bbox.size();
  std::vector<std::vector<tensorrt_lightnet::BBoxInfo>> tmpBbox;
  tmpBbox.resize(numWorks);
#pragma omp parallel for  
  for (int p = 0; p < numWorks; p++) {
    std::vector<tensorrt_lightnet::BBoxInfo> subnetBbox;
    for (int i = (p) * num / numWorks; i < (p+1) * num / numWorks; i++) {
      auto b = bbox[i];
      int flg = false;
      for (auto &t : target) {
	if (t == names[b.classId]) {
	  flg = true;
	}
      }
      if (!flg) {
	continue;
      }
      cv::Rect roi(b.box.x1, b.box.y1, b.box.x2-b.box.x1, b.box.y2-b.box.y1);
      cv::Mat cropped = (image)(roi);
      subnet_trt_lightnets[p]->preprocess({cropped});
      subnet_trt_lightnets[p]->doInference();
      subnet_trt_lightnets[p]->makeBbox(cropped.rows, cropped.cols);    
      auto bb = subnet_trt_lightnets[p]->getBbox();
      for (int j = 0; j < (int)bb.size(); j++) {
	bb[j].box.x1 += b.box.x1;
	bb[j].box.y1 += b.box.y1;
	bb[j].box.x2 += b.box.x1;
	bb[j].box.y2 += b.box.y1;
      }      
      subnetBbox.insert(subnetBbox.end(), bb.begin(), bb.end());
    }
    tmpBbox[p] = subnetBbox;
  }
  
  for (int p = 0; p < numWorks; p++) {
    trt_lightnet->appendSubnetBbox(tmpBbox[p]);
  }  
}

/**
 * Draws detected objects and their associated masks and depth maps on the image.
 * 
 * @param trt_lightnet A shared pointer to the TensorRT Lightnet model.
 * @param image The image on which detections, masks, and depth maps will be overlaid.
 * @param colormap A vector of vectors containing RGB values for coloring each class.
 * @param names A vector of class names used for labeling the detections.
 */
void drawLightNet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<std::vector<int>> &colormap, std::vector<std::string> &names)
{
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();  
  std::vector<cv::Mat> masks = trt_lightnet->getMask();
  std::vector<cv::Mat> depthmaps = trt_lightnet->getDepthmap();
  
  for (const auto &mask : masks) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
    cv::addWeighted(image, 1.0, resized, 0.45, 0.0, image);
    cv::imshow("mask", mask);
  }
  for (const auto &depth : depthmaps) {
    cv::imshow("depth", depth);
  }
  trt_lightnet->drawBbox(image, bbox, colormap, names);
}

/**
 * Draws detected objects using subnet model data on the image.
 * 
 * @param trt_lightnet A shared pointer to the TensorRT Lightnet model.
 * @param image The image on which subnet detections will be drawn.
 * @param colormap A vector of vectors containing RGB values used for drawing each class.
 * @param subnet_names A vector of subnet class names used for labeling the detections.
 */
void drawSubnetLightNet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<std::vector<int>> &colormap, std::vector<std::string> &subnet_names)
{
  std::vector<tensorrt_lightnet::BBoxInfo> subnet_bbox = trt_lightnet->getSubnetBbox();  
  trt_lightnet->drawBbox(image, subnet_bbox, colormap, subnet_names);		
}

/**
 * Applies a blurring effect on objects detected by the subnet within the given image.
 * 
 * @param trt_lightnet A shared pointer to the TensorRT Lightnet model.
 * @param image The image on which the blurring effect will be applied to the detected objects.
 */
void blurObjectFromSubnetBbox(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image)
{
  cv::Mat cropped;
  int kernel = 15;  
  std::vector<tensorrt_lightnet::BBoxInfo> subnet_bbox = trt_lightnet->getSubnetBbox();
  for (auto &b : subnet_bbox) {
    if ((b.box.x2-b.box.x1) > kernel && (b.box.y2-b.box.y1) > kernel) {
      int width = b.box.x2-b.box.x1;
      int height = b.box.y2-b.box.y1;
      int w_offset = width  * 0.0;
      int h_offset = height * 0.0;      
      cv::Rect roi(b.box.x1+w_offset, b.box.y1+h_offset, width-w_offset*2, height-h_offset*2);
      cropped = (image)(roi);
      if (width > 320 && height > 320) {
	cv::blur(cropped, cropped, cv::Size(kernel*4, kernel*4));
      } else {
	cv::blur(cropped, cropped, cv::Size(kernel, kernel));
      }
    }
  }
}

/**
 * Applies a Pixelation on objects detected by the subnet within the given image.
 * 
 * @param trt_lightnet A shared pointer to the TensorRT Lightnet model.
 * @param image The image on which the blurring effect will be applied to the detected objects.
 */
void applyPixelationObjectFromSubnetBbox(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image)
{
  cv::Mat resized;
  cv::Mat cropped;
  int kernel = 15;  
  std::vector<tensorrt_lightnet::BBoxInfo> subnet_bbox = trt_lightnet->getSubnetBbox();
  for (auto &b : subnet_bbox) {
    if ((b.box.x2-b.box.x1) > kernel && (b.box.y2-b.box.y1) > kernel) {
      int width = b.box.x2-b.box.x1;
      int height = b.box.y2-b.box.y1;
      cv::Rect roi(b.box.x1, b.box.y1, width, height);
      cropped = (image)(roi);
      cv::resize(cropped, resized, cv::Size(width/kernel, height/kernel), 0, 0, cv::INTER_NEAREST);
      cv::resize(resized, cropped, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    }
  }
}

/**
 * Writes detection predictions to a text file within the specified directory.
 *
 * @param save_path The path to the directory where the text file will be saved.
 * @param filename The original filename of the image; used to construct the output text filename.
 * @param names A vector of strings representing class names corresponding to class IDs.
 * @param bbox A vector of bounding box information, each containing class IDs, probability scores, and coordinates.
 */
void
writePredictions(std::string save_path, std::string filename, std::vector<std::string> names, std::vector<tensorrt_lightnet::BBoxInfo> bbox)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = save_path;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);
  for (const auto& bbi : bbox) {
    int id = bbi.classId;
    std::string writing_text = format("%s %f %d %d %d %d", names[id].c_str(), (float)bbi.prob, (int)bbi.box.x1, (int)bbi.box.y1, (int)bbi.box.x2, (int)bbi.box.y2);
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}

/**
 * Processes and saves various outputs (detection images, prediction results, segmentation masks, depth maps) using
 * data from a TrtLightnet object.
 *
 * @param trt_lightnet A shared pointer to the TrtLightnet object containing detection and segmentation information.
 * @param image The image on which detections were performed.
 * @param colormap A vector of vectors, each inner vector representing a color map for the segmentation masks.
 * @param names A vector of class names corresponding to detection class IDs.
 * @param save_path The base path where all outputs will be saved.
 * @param filename The filename of the original image; used to determine output filenames and formats.
 */
void
saveLightNet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<std::vector<int>> &colormap, std::vector<std::string> &names, std::string save_path, std::string filename)
{
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();  
  std::vector<cv::Mat> masks = trt_lightnet->getMask();
  std::vector<cv::Mat> depthmaps = trt_lightnet->getDepthmap();
  
  std::string png_name = filename;
  if (png_name.find(".jpg") != std::string::npos) {
    replaceOtherStr(png_name, ".jpg", ".png");
  }
  fs::create_directories(fs::path(save_path));
#pragma omp parallel sections
  {
#pragma omp section
    {      
      fs::path p = fs::path(save_path) / "detections";
      fs::create_directories(p);
      saveImage(image, p.string(), filename);
    }
#pragma omp section
    {        
      fs::path p = fs::path(save_path) / "predictions";
      writePredictions(p.string(), filename, names, bbox);
      std::vector<tensorrt_lightnet::BBoxInfo> subnet_bbox = trt_lightnet->getSubnetBbox();
      if (subnet_bbox.size()) {
	fs::path p = fs::path(save_path) / "subnet_predictions";
	auto subnet_names = get_subnet_names();
	writePredictions(p.string(), filename, subnet_names, subnet_bbox);
      }
    }
#pragma omp section
    {
      if (masks.size()) {
	fs::create_directories(fs::path(save_path) / "segmentations");
      }
      for (int i = 0; i < (int)masks.size(); i++) {
	fs::path p = fs::path(save_path) / "segmentations" / std::to_string(i);
	fs::create_directory(p);
	cv::Mat resized;
	cv::resize(masks[i], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);    
	saveImage(resized, p.string(), png_name);
      }
    }
#pragma omp section
    {
      if (depthmaps.size()) {
	fs::create_directories(fs::path(save_path) / "depthmaps");
      }
      for (int i = 0; i < (int)depthmaps.size(); i++) {
	fs::path p = fs::path(save_path) / "depthmaps" / std::to_string(i);
	fs::create_directory(p);    
	cv::Mat resized;
	cv::resize(depthmaps[i], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
	saveImage(resized, p.string(), png_name);
      }
    }
  }
}

/**
 * Visualizes a feature map by normalizing and scaling float data into an image.
 *
 * This function converts a single-channel feature map, represented as a float array,
 * into an 8-bit single-channel image. The function normalizes the values of the feature
 * map based on its minimum and maximum values, and scales it to the range of 0 to 255.
 *
 * @param data Pointer to the float array representing the feature map data.
 * @param w Width of the feature map.
 * @param h Height of the feature map.
 * @return cv::Mat The resulting visualization of the feature map as an 8-bit grayscale image.
 */
cv::Mat
visualizeFeaturemap(const float* data, int w, int h)
{
  float min = 0.0;
  float max = 0.0;  
  cv::Mat featuremap = cv::Mat::zeros(h, w, CV_8UC1);
  for (int i = 0; i < w*h; i++) {
    max = std::max(max, data[i]);
    min = std::min(min, data[i]);	
  }
  float range = max - min;
  for (int i = 0; i < w*h; i++) {
    float tmp = (data[i] - min) / range * 255;
    int x = i % h;
    int y = i / h;
    featuremap.at<unsigned char>(y, x) = (unsigned int)(tmp);
  }
  return featuremap;
}

/**
 * Saves debugging tensor data into a .npz file for further analysis.
 *
 * This function retrieves debugging tensors from a given TrtLightnet instance and saves them
 * into a specified directory as a .npz file. The tensors are expected to be in NCHW format.
 *
 * @param trt_lightnet Shared pointer to the TrtLightnet instance from which to retrieve tensors.
 * @param save_path Directory path where the .npz file will be saved.
 * @param filename Name of the file from which to derive the .npz file name.
 */
void
saveDebugTensors(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, std::string save_path, std::string filename)
{
  std::vector<nvinfer1::Dims> dim_infos;
  std::vector<std::string> names;  
  std::vector<float*> debug_tensors = trt_lightnet->getDebugTensors(dim_infos, names);
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".npz";
  std::ofstream writing_file;
  fs::path p = save_path;
  fs::create_directory(p);  
  p.append(dstName);  
  for (int d = 0; d < (int)dim_infos.size(); d++) {
    //Supports only single batch
    //NCHW
    int w = dim_infos[d].d[3];
    int h = dim_infos[d].d[2];
    /*
    for (int c = 0; c < (int)dim_infos[d].d[1]; c++) {
      cv::Mat debug = visualizeFeaturemap(&((debug_tensors[d])[h * w * c]), w, h);
      cv::imshow("debug_"+std::to_string(d)+"_"+std::to_string(c), debug);
    }
    */
    cnpy::npz_save(p.string(), names[d], debug_tensors[d], {(long unsigned int)dim_infos[d].d[1], (long unsigned int)h, (long unsigned int)w}, "a");
  }
}

/**
 * Saves prediction results, including bounding boxes, to a specified directory.
 *
 * This function takes predictions from a TrtLightnet instance, specifically bounding boxes,
 * and writes them to a specified path. The function supports specifying an output file format
 * and directory.
 *
 * @param trt_lightnet Shared pointer to the TrtLightnet instance from which to retrieve predictions.
 * @param names Vector of strings representing the names associated with the predictions.
 * @param save_path Directory path where the prediction results will be saved.
 * @param filename Base name of the file used for saving predictions.
 * @param dst Additional path component or file destination specifier.
 */
void
savePrediction(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, std::vector<std::string> &names, std::string save_path, std::string filename, std::string dst)
{
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();  
  
  std::string png_name = filename;
  fs::path p = fs::path(save_path) / dst;
  writePredictions(p.string(), filename, names, bbox);
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
    .nms_threshold = 0.45f, // Assuming this value is fixed or retrieved similarly.
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
    .save_path = getSaveDetectionsPath(),
    .flg_save_debug_tensors = get_save_debug_tensors()
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
  tensorrt_common::BuildConfig build_config(
					    inference_config.calibration_type,
					    inference_config.dla_core_id,
					    inference_config.use_first_layer,
					    inference_config.use_last_layer,
					    inference_config.profile,
					    0.0, // Assuming a fixed value for missing float parameter, might need adjustment.
					    inference_config.sparse,
					    get_debug_tensors()
					    );

  auto trt_lightnet =
    std::make_shared<tensorrt_lightnet::TrtLightnet>(model_config, inference_config, build_config);
  //Subnet configuration
  std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> subnet_trt_lightnets;
  VisualizationConfig subnet_visualization_config;
  std::vector<std::string> target;
  std::vector<std::string> bluron = get_bluron_names();
  int numWorks = omp_get_max_threads();
  if (get_subnet_onnx_path() != "not-specified") {
    ModelConfig subnet_model_config = {
      .model_path = get_subnet_onnx_path(),
      .num_class = get_subnet_classes(),
      .score_threshold = static_cast<float>(get_score_thresh()),
      .anchors = get_subnet_anchors(),
      .num_anchors = get_subnet_num_anchors(),
      .nms_threshold = 0.25f // Assuming this value is fixed or retrieved similarly.
    };
    subnet_visualization_config = {
      .dont_show = is_dont_show(),
      .colormap = get_subnet_colormap(),
      .names = get_subnet_names(),
      .argmax2bgr = getArgmaxToBgr(get_seg_colormap())
    };
    target = get_target_names();
    for (int w = 0; w < numWorks; w++) {
      if (build_config.dla_core_id >= 2) {
	//use multiple dlas [DLA0 and DLA1]
	build_config.dla_core_id = (int)w/2;
      }
      subnet_trt_lightnets.push_back(
				     std::make_shared<tensorrt_lightnet::TrtLightnet>(subnet_model_config, inference_config, build_config));
    }
  }
  
  
  if (!path_config.directory.empty()) {
    for (const auto& file : fs::directory_iterator(path_config.directory)) {
      std::cout << "Inference from " << file.path() << std::endl;
      cv::Mat image = cv::imread(file.path(), cv::IMREAD_UNCHANGED);
      inferLightnet(trt_lightnet, image, visualization_config.argmax2bgr);

      if (get_subnet_onnx_path() != "not-specified" && subnet_trt_lightnets.size()) {
	inferSubnetLightnets(trt_lightnet, subnet_trt_lightnets, image, visualization_config.names, target, numWorks);
	if (bluron.size()) {
	  blurObjectFromSubnetBbox(trt_lightnet, image);
	}
      }
      
      if (!visualization_config.dont_show) {
	drawLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names);
	if (get_subnet_onnx_path() != "not-specified" && subnet_trt_lightnets.size()) {
	  drawSubnetLightNet(trt_lightnet, image, subnet_visualization_config.colormap, subnet_visualization_config.names);
	}
      }
      if (path_config.flg_save) {
	saveLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, path_config.save_path, file.path().filename());
	if (get_subnet_onnx_path() != "not-specified" && subnet_trt_lightnets.size()) {
	  //	  savePrediction(subnet_trt_lightnet, subnet_visualization_config.names, path_config.save_path, file.path().filename(), "subnet_predictions");
	}
      }
      if (path_config.flg_save_debug_tensors && path_config.save_path != "") {
	fs::path p = fs::path(path_config.save_path) / "debug_tensors";
	saveDebugTensors(trt_lightnet, p.string(), file.path().filename());
      }      
      if (!visualization_config.dont_show) {
	if (image.rows > 1280 && image.cols > 1920) {
	  cv::resize(image, image, cv::Size(1920, 1280), 0, 0, cv::INTER_LINEAR);
	}
	cv::imshow("inference", image);	
	cv::waitKey(0);
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
      inferLightnet(trt_lightnet, image, visualization_config.argmax2bgr);

      if (get_subnet_onnx_path() != "not-specified" && subnet_trt_lightnets.size()) {
	inferSubnetLightnets(trt_lightnet, subnet_trt_lightnets, image, visualization_config.names, target, numWorks);
	if (bluron.size()) {
	  blurObjectFromSubnetBbox(trt_lightnet, image);
	  //applyPixelationObjectFromSubnetBbox(trt_lightnet, image);
	}
      }
      
      if (!visualization_config.dont_show) {
	drawLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names);
	if (get_subnet_onnx_path() != "not-specified" && subnet_trt_lightnets.size()) {
	  drawSubnetLightNet(trt_lightnet, image, subnet_visualization_config.colormap, subnet_visualization_config.names);
	}
      }
      if (flg_save) {
	std::ostringstream sout;
	sout << std::setfill('0') << std::setw(6) << count;	  
	std::string name = "frame_" + sout.str() + ".jpg";
	saveLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, path_config.save_path, name);
      }
      if (path_config.flg_save_debug_tensors && path_config.save_path != "") {
	std::ostringstream sout;
	sout << std::setfill('0') << std::setw(6) << count;	  
	std::string name = "frame_" + sout.str() + ".jpg";
	fs::path p = fs::path(path_config.save_path) / "debug_tensors";
	saveDebugTensors(trt_lightnet, p.string(), name);
      }	
      
      if (!visualization_config.dont_show) {
	if (image.rows > 1280 && image.cols > 1920) {
	  cv::resize(image, image, cv::Size(1920, 1280), 0, 0, cv::INTER_LINEAR);
	}
	cv::imshow("inference", image);	
	if (cv::waitKey(1) == 'q') {
	  break;
	}
      }
      count++;      
    }
  }  
  
  if (inference_config.profile) {
    trt_lightnet->printProfiling();
  }

  return 0;
}

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
#include <pcd2image.hpp>
#include <CalibratedSensorParser.h>
#include <fswp/fswp.hpp>
#include <omp.h>

/**
 * Configuration for input and output paths used during inference.
 * This includes directories for images, videos, and where to save output.
 */
struct PathConfig {
  std::string t4dataset_directory; ///< T4 Dataset Directory containing images for inference.  
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
  std::vector<tensorrt_lightnet::Colormap> seg_colormap;
  std::vector<int> road_ids;
};

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

cv::Mat concatHorizontal(const cv::Mat& mat1, const cv::Mat& mat2) {
  if (mat1.empty() || mat2.empty()) {
    std::cerr << "Input matrices should not be empty!" << std::endl;
    return cv::Mat();
  }

  cv::Mat resizedMat2;
  double aspectRatio = static_cast<double>(mat2.cols) / mat2.rows;
  int newWidth = static_cast<int>(aspectRatio * mat1.rows); 
  cv::resize(mat2, resizedMat2, cv::Size(newWidth, mat1.rows));

  cv::Mat concatenated;
  cv::hconcat(mat1, resizedMat2, concatenated);

  return concatenated;
}

cv::Mat concatHorizontalWithPadding(const cv::Mat& mat1, const cv::Mat& mat2) {
  if (mat1.empty() || mat2.empty()) {
    std::cerr << "Input matrices should not be empty!" << std::endl;
    return cv::Mat();
  }

  int maxHeight = std::max(mat1.rows, mat2.rows);

  cv::Mat paddedMat1 = cv::Mat::zeros(maxHeight, mat1.cols, mat1.type());
  mat1.copyTo(paddedMat1(cv::Rect(0, (maxHeight - mat1.rows) / 2, mat1.cols, mat1.rows)));

  cv::Mat paddedMat2 = cv::Mat::zeros(maxHeight, mat2.cols, mat2.type());
  mat2.copyTo(paddedMat2(cv::Rect(0, (maxHeight - mat2.rows) / 2, mat2.cols, mat2.rows)));

  cv::Mat concatenated;
  cv::hconcat(paddedMat1, paddedMat2, concatenated);

  return concatenated;
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
 * Performs fast face swapping on the input image using TensorRT LightNet and FaceSwapper models.
 *
 * @param trt_lightnet A shared pointer to a `tensorrt_lightnet::TrtLightnet` object used for detecting bounding boxes.
 * @param fswp_model A shared pointer to a `fswp::FaceSwapper` object used for swapping faces and inpainting the image.
 * @param image A reference to the `cv::Mat` object containing the input image. This image is modified in-place with the swapped faces.
 * @param names A vector of strings representing class names corresponding to the detected bounding boxes.
 * @param target A vector of strings representing the target class names to swap faces for. Only bounding boxes with these class names are processed.
 * @param numWorks The number of parallel workers (not used in the current implementation).
 */
void inferFastFaceSwapper(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, std::shared_ptr<fswp::FaceSwapper> fswp_model, cv::Mat &image, std::vector<std::string> &names, std::vector<std::string> &target, const int numWorks)
{
  trt_lightnet->removeAspectRatioBoxes(names, target, 1/4.0);  
  trt_lightnet->removeContainedBBoxes(names, target);
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();
  int num = (int)bbox.size();

  std::vector<tensorrt_lightnet::BBoxInfo> fdet_bboxes;
  for (int i = 0; i < num; i++) {
      auto b = bbox[i];
      if ((b.box.x2 - b.box.x1) < 32) {
	continue;
      }
      if ((b.box.y2 - b.box.y1) < 32) {
	continue;
      }      
      for (auto &t : target) {
	if (t == names[b.classId]) {
	  fdet_bboxes.push_back(b);
	}
      }
  }
  std::vector<fswp::BBox> fswp_bboxes(fdet_bboxes.size());
  std::transform(fdet_bboxes.begin(), fdet_bboxes.end(), fswp_bboxes.begin(),
		 [](const auto &boxinfo) {
		   const auto &box = boxinfo.box;
		   return fswp::BBox{box.x1, box.y1, box.x2, box.y2};
		 });
  fswp_model->inpaint(image, fswp_bboxes);  
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
void inferLightnet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, VisualizationConfig visualization_config, std::shared_ptr<fswp::FaceSwapper> fswp_model, std::vector<std::string> &target)
{
  //preprocess
  if (get_cuda_flg()) {
    trt_lightnet->preprocess_gpu({image});
  } else {
    trt_lightnet->preprocess({image});
  }
  //inference

  trt_lightnet->doInference();

  //postprocess
  trt_lightnet->makeBbox(image.rows, image.cols);

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (get_smooth_depthmap_using_semseg()) {
	if (get_cuda_flg()) {
	  trt_lightnet->smoothDepthmapFromRoadSegmentationGpu(visualization_config.road_ids);
	} else {
	  trt_lightnet->smoothDepthmapFromRoadSegmentation(visualization_config.road_ids);
	}
	
      }
      
      trt_lightnet->makeMask(visualization_config.argmax2bgr);
      if (get_cuda_flg()) {
	trt_lightnet->makeDepthmapGpu();
      } else {
	std::string depth_format = get_depth_format();      
	trt_lightnet->makeDepthmap(depth_format);
      }
      trt_lightnet->makeKeypoint(image.rows, image.cols);

      Calibration calibdata = {
	.u0 = (float)(image.cols/2.0),
	.v0 = (float)(image.rows/2.0),
	.fx = get_fx(),
	.fy = get_fy(),
	.max_distance = get_max_distance(),
      };
      if (get_cuda_flg()) {
	if (get_sparse_depth_flg()) {
	  trt_lightnet->makeBackProjectionGpuWithoutDensify(image.cols, image.rows, calibdata);
	}
	trt_lightnet->makeBackProjectionGpu(image.cols, image.rows, calibdata, 2);
      } else {
	trt_lightnet->makeBackProjection(image.cols, image.rows, calibdata);
      }
    }
#pragma omp section
    {
      if (get_fswp_onnx_path() != "not-specified") {
	std::chrono::high_resolution_clock::time_point start, end;
	if (profile_verbose()) {
	  start = std::chrono::high_resolution_clock::now();
	}
	
	inferFastFaceSwapper(trt_lightnet, fswp_model, image, visualization_config.names, target, 1);
	if (profile_verbose()) {      
	  end = std::chrono::high_resolution_clock::now();
	  std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	  std::cout << "##inferFastFaceSwapper: " << duration.count() << " ms " << std::endl;
	}        
      }
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

  trt_lightnet->doNonMaximumSuppressionForSubnetBbox();
}


/**
 * @brief Performs keypoint inference using multiple subnetworks on a given image.
 * 
 * This function takes an input image and performs keypoint inference using multiple 
 * TensorRT Lightnet subnetworks in parallel. The detected keypoints are then linked 
 * back to the main network. It processes each bounding box region by cropping the 
 * corresponding part of the image, running inference on the subnetwork, and adjusting 
 * the keypoint positions to the original image coordinates.
 * 
 * @param trt_lightnet A shared pointer to the main TensorRT Lightnet network.
 * @param subnet_trt_lightnets A vector of shared pointers to the subnetwork TensorRT Lightnets.
 * @param image The input image on which keypoint inference is performed.
 * @param names A vector of class names corresponding to detected objects.
 * @param target A vector of target class names for which keypoint inference should be performed.
 * @param numWorks The number of parallel workers to use for processing.
 */
void inferKeypointLightnets(
			    std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet,
			    std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> subnet_trt_lightnets,
			    cv::Mat &image,
			    std::vector<std::string> &names,
			    std::vector<std::string> &target,
			    const int numWorks)
{
  // Get the bounding boxes from the main network and clear any existing keypoints.
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();
  trt_lightnet->clearKeypoint();
  int num = static_cast<int>(bbox.size());

  // Parallel processing of bounding boxes using OpenMP.
  #pragma omp parallel for
  for (int p = 0; p < numWorks; p++) {
    std::vector<tensorrt_lightnet::KeypointInfo> subnetKeypoint;

    // Divide the work across parallel workers.
    for (int i = p * num / numWorks; i < (p + 1) * num / numWorks; i++) {
      auto b = bbox[i];
      bool flg = false;

      // Check if the bounding box class is in the target list.
      for (const auto &t : target) {
	if (t == names[b.classId]) {
	  flg = true;
	  break;
	}
      }

      if (!flg) {
	continue; // Skip if the class is not in the target list.
      }
      if ((b.box.x2 - b.box.x1) < 32) {
	continue;
      }
      // Define the region of interest (ROI) based on the bounding box.
      //int ylen = (b.box.y2 + 32) < image.rows ? (b.box.y2 - b.box.y1 + 32) : (b.box.y2 - b.box.y1);
      //cv::Rect roi(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, ylen);
      cv::Rect roi(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1);
      cv::Mat cropped = image(roi);
      
      // Preprocess and run inference on the subnetwork.
      subnet_trt_lightnets[p]->preprocess({cropped});
      subnet_trt_lightnets[p]->doInference();
      subnet_trt_lightnets[p]->makeKeypoint(cropped.rows, cropped.cols);

      // Get the inferred keypoints and adjust them to the original image coordinates.
      auto keypoint = subnet_trt_lightnets[p]->getKeypoints();
      for (auto &k : keypoint) {
	k.lx0 += b.box.x1;
	k.ly0 += b.box.y1;
	k.lx1 += b.box.x1;
	k.ly1 += b.box.y1;
	k.rx0 += b.box.x1;
	k.ry0 += b.box.y1;
	k.rx1 += b.box.x1;
	k.ry1 += b.box.y1;
	k.bot = b.box.y1;
	k.left = b.box.x1;
      }

      // Link the adjusted keypoints back to the main network.
      trt_lightnet->linkKeyPoint(keypoint, i);
      subnetKeypoint.insert(subnetKeypoint.end(), keypoint.begin(), keypoint.end());
    }
  }
  Calibration calibdata = {
    .u0 = (float)(image.cols/2.0),
    .v0 = (float)(image.rows/2.0),
    .fx = get_fx(),
    .fy = get_fy(),
    .max_distance = get_max_distance(),
  };
  trt_lightnet->addBBoxIntoBevmap(image.cols, image.rows, calibdata, names);    
}

/**
 * Draws detected objects and their associated masks and depth maps on the image.
 * 
 * @param trt_lightnet A shared pointer to the TensorRT Lightnet model.
 * @param image The image on which detections, masks, and depth maps will be overlaid.
 * @param colormap A vector of vectors containing RGB values for coloring each class.
 * @param names A vector of class names used for labeling the detections.
 */
void drawLightNet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, cv::Mat &image, std::vector<std::vector<int>> &colormap, std::vector<std::string> &names, std::vector<std::string> &target)
{
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet->getBbox();  
  std::vector<cv::Mat> masks = trt_lightnet->getMask();
  std::vector<cv::Mat> depthmaps = trt_lightnet->getDepthmap();
  std::vector<tensorrt_lightnet::KeypointInfo> keypoint = trt_lightnet->getKeypoints();

  for (const auto &mask : masks) {
    if (get_cuda_flg()) {
      trt_lightnet->blendSegmentationGpu(image, 1.0, get_blending(), 0.0);
    } else {
      cv::Mat resized;
      cv::resize(mask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
      cv::addWeighted(image, 1.0, resized, get_blending(), 0.0, image);
    }
    cv::imshow("mask", mask);
  }

  for (const auto &depth : depthmaps) {
    cv::imshow("depth", depth);
  }

  trt_lightnet->drawBbox(image, bbox, colormap, names);

  if (get_calc_entropy_flg()) {
    std::vector<cv::Mat> ent_maps = trt_lightnet->getEntropymaps();
    for (const auto &ent_map : ent_maps) {
      cv::imshow("entropy", ent_map);
    }
  }

  if (get_calc_cross_task_inconsistency_flg()) {
    std::vector<cv::Mat> inconsitency_maps = trt_lightnet->getCrossTaskInconsistency_map();
    for (const auto &inconsitency_map : inconsitency_maps) {
      cv::imshow("inconsistency", inconsitency_map);
    }    
  }

  if (target.size() > 0 ) {
    if (get_plot_circle_flg()) {
      Calibration calibdata = {
	.u0 = (float)(image.cols/2.0),
	.v0 = (float)(image.rows/2.0),
	.fx = get_fx(),
	.fy = get_fy(),
	.max_distance = get_max_distance(),
      };
      trt_lightnet->plotCircleIntoBevmap(image.cols, image.rows, calibdata, names, target);
    }
  }  

  if (keypoint.size() > 0) {
    trt_lightnet->drawKeypoint(image, keypoint);
  }
    
  if (masks.size() > 0 && depthmaps.size() > 0) {
    cv::Mat bevmap = trt_lightnet->getBevMap();
    cv::imshow("bevmap", bevmap);
    if (get_cuda_flg()) {
      if (get_sparse_depth_flg()) {
	cv::Mat sparsemap = trt_lightnet->getSparseBevMap();
	cv::imshow("sparse-bevmap", sparsemap);
      }
    }
  }
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
	cv::blur(cropped, cropped, cv::Size(kernel*16, kernel*16));
      } else if (width > 160 && height > 160) {
	cv::blur(cropped, cropped, cv::Size(kernel*8, kernel*8));
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
    std::string writing_text;
    if (!bbi.isHierarchical) {
      writing_text = format("%s %f %d %d %d %d", names[id].c_str(), (float)bbi.prob, (int)bbi.box.x1, (int)bbi.box.y1, (int)bbi.box.x2, (int)bbi.box.y2);
    } else {
      //For TLR
      writing_text = tensorrt_lightnet::getTLRStringFromBBox(bbi, names);
    }
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}


/**
 * Writes vectors to a text file within the specified directory.
 *
 * @param save_path The path to the directory where the text file will be saved.
 * @param filename The original filename of the image; used to construct the output text filename.
 * @param names A vector of strings representing class names corresponding to class IDs.
 * @param values A vector of floating point values.
 */
void
writeValue(std::string save_path, std::string filename, std::vector<float> &values)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = save_path;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);
  std::stringstream stream;
  for (int i = 0; i < (int)values.size(); i++) {
    if (i == (int)values.size()-1) {
      stream << std::setprecision(2) << values[i] << "\n";
    } else {
      stream << std::setprecision(2) << values[i] << " ";
    }
  }
  writing_file << stream.str();
  writing_file.close();
}

/**
 * @brief Saves entropy values to files.
 * 
 * This function saves entropy values to individual directories under a specified path.
 * Each entropy vector is saved in its own directory.
 *
 * @param trt_lightnet Shared pointer to a TensorRT Lightnet instance
 * @param save_path Base directory path where entropy values will be saved
 * @param filename The name of the file to save the entropy values
 */
void writeEntropy(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, const std::string& save_path, const std::string& filename)
{
  // Retrieve the entropy vectors
  std::vector<std::vector<float>> entropies = trt_lightnet->getEntropies();

  // Create the main "entropy" directory if entropies are available
  if (!entropies.empty()) {
    fs::create_directories(fs::path(save_path) / "entropy");
  }

  // Save each entropy vector in its own directory
  for (size_t i = 0; i < entropies.size(); i++) {
    // Generate the path for each entropy directory
    fs::path p = fs::path(save_path) / "entropy" / std::to_string(i);
    fs::create_directory(p);

    // Write the entropy values to the file
    writeValue(p.string(), filename, entropies[i]);
  }
}

/**
 * @brief Saves CrossTaskInconsistency values to files.
 * 
 * This function saves CrossTaskInconsistency values to individual directories under a specified path.
 * Each CrossTaskInconsistency vector is saved in its own directory.
 *
 * @param trt_lightnet Shared pointer to a TensorRT Lightnet instance
 * @param save_path Base directory path where CrossTaskInconsistency values will be saved
 * @param filename The name of the file to save the CrossTaskInconsistency values
 */
void writeCrossTaskInconsistency(std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet, const std::string& save_path, const std::string& filename)
{
  // Retrieve the CrossTaskInconsistency vectors
  std::vector<std::vector<float>> inconsitencies = trt_lightnet->getCrossTaskInconsistencies();

  // Create the main "CrossTaskInconsistency" directory if inconsitencies are available
  if (!inconsitencies.empty()) {
    fs::create_directories(fs::path(save_path) / "inconsistency");
  }

  // Save each CrossTaskInconsistency vector in its own directory
  for (size_t i = 0; i < inconsitencies.size(); i++) {
    // Generate the path for each CrossTaskInconsistency directory
    fs::path p = fs::path(save_path) / "inconsistency" / std::to_string(i);
    fs::create_directory(p);

    // Write the CrossTaskInconsistency values to the file
    writeValue(p.string(), filename, inconsitencies[i]);
  }
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
  if (filename.find(".pcd.bin") != std::string::npos) {  
    replaceOtherStr(filename, ".pcd.bin", ".png");
    png_name = filename;
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
  if (get_calc_entropy_flg()) {
    std::vector<cv::Mat> ent_maps = trt_lightnet->getEntropymaps();
    if (ent_maps.size()) {
      fs::create_directories(fs::path(save_path) / "entropyVisualization");
    }
    for (int i = 0; i < (int)ent_maps.size(); i++) {
      fs::path p = fs::path(save_path) / "entropyVisualization" / std::to_string(i);
      fs::create_directory(p);    
      cv::Mat resized;
      cv::resize(ent_maps[i], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
      saveImage(resized, p.string(), png_name);
    }
  }
  if (get_calc_cross_task_inconsistency_flg()) {
    std::vector<cv::Mat> inconsistency_maps = trt_lightnet->getCrossTaskInconsistency_map();
    if (inconsistency_maps.size()) {
      fs::create_directories(fs::path(save_path) / "inconsistencyVisualization");
    }
    for (int i = 0; i < (int)inconsistency_maps.size(); i++) {
      fs::path p = fs::path(save_path) / "inconsistencyVisualization" / std::to_string(i);
      fs::create_directory(p);    
      cv::Mat resized;
      cv::resize(inconsistency_maps[i], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
      saveImage(resized, p.string(), png_name);
    }
  }
  if (masks.size() && depthmaps.size()) {
    fs::path p = fs::path(save_path) / "bevmap";
    fs::create_directory(p);
    cv::Mat bevmap = trt_lightnet->getBevMap();
    saveImage(bevmap, p.string(), png_name);
    p = fs::path(save_path) / "occupancyGrid";
    fs::create_directory(p);
    cv::Mat occupancy = trt_lightnet->getOccupancyGrid();
    saveImage(occupancy, p.string(), png_name);
    if (get_cuda_flg()) {
      if (get_sparse_depth_flg()) {
	cv::Mat sparsemap = trt_lightnet->getSparseBevMap();
	p = fs::path(save_path) / "sparse_bevmap";
	fs::create_directory(p);
	saveImage(sparsemap, p.string(), png_name);	
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
    if (!is_dont_show()) {
      for (int c = 0; c < (int)dim_infos[d].d[1]; c++) {
	cv::Mat debug = visualizeFeaturemap(&((debug_tensors[d])[h * w * c]), w, h);
	cv::imshow("debug_"+std::to_string(d)+"_"+std::to_string(c), debug);
      }
    }

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

/**
 * @brief Runs the inference pipeline for LightNet, including optional subnet and keypoint processing, 
 *        entropy calculations, cross-task inconsistency checks, and visualization.
 *
 * @param trt_lightnet Pointer to the main TensorRT LightNet model.
 * @param subnet_trt_lightnets Vector of pointers to subnet TensorRT LightNet models.
 * @param keypoint_trt_lightnets Vector of pointers to keypoint TensorRT LightNet models.
 * @param image Input image for processing.
 * @param visualization_config Configuration for visualization of the main LightNet results.
 * @param subnet_visualization_config Configuration for visualization of subnet LightNet results.
 * @param target List of target class names for subnet processing.
 * @param keypoint_target List of target class names for keypoint processing.
 * @param bluron List of targets to apply blur effects.
 * @param path_config Configuration for saving outputs.
 * @param filename Name of the file to save the output.
 * @param sensor_name Sensor name to include in the output path.
 * @param cam_name Camera name to include in the output path.
 */
void inferLightNetPipeline(
			   std::shared_ptr<tensorrt_lightnet::TrtLightnet> trt_lightnet,
			   std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> &subnet_trt_lightnets,
			   std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> &keypoint_trt_lightnets,
			   std::shared_ptr<fswp::FaceSwapper> fswp_model,
			   cv::Mat &image,
			   VisualizationConfig visualization_config,
			   VisualizationConfig subnet_visualization_config,
			   std::vector<std::string> &target,
			   std::vector<std::string> &keypoint_target,
			   std::vector<std::string> &bluron,
			   PathConfig path_config,
			   std::string filename,
			   std::string sensor_name,
			   std::string cam_name
			   ) {
  int numWorks = get_workers();
  std::chrono::high_resolution_clock::time_point start, end;
  if (profile_verbose()) {
    start = std::chrono::high_resolution_clock::now();
  }  
  // Perform inference with the main LightNet model
  inferLightnet(trt_lightnet, image, visualization_config, fswp_model, target);
  if (profile_verbose()) {      
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "##inferLightnet: " << duration.count() << " ms " << std::endl;
  }

  
  // Check and process subnet LightNets if applicable
  if (get_subnet_onnx_path() != "not-specified" && !subnet_trt_lightnets.empty()) {
    if (profile_verbose()) {
      start = std::chrono::high_resolution_clock::now();
    }      
    inferSubnetLightnets(trt_lightnet, subnet_trt_lightnets, image, visualization_config.names, target, numWorks);
    if (!bluron.empty()) {
      blurObjectFromSubnetBbox(trt_lightnet, image);
    }
    if (profile_verbose()) {      
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "##inferSubnetLightnets: " << duration.count() << " ms " << std::endl;
    }    
  }

  // Check and process keypoint LightNets if applicable
  if (get_keypoint_onnx_path() != "not-specified" && !keypoint_trt_lightnets.empty()) {
    if (profile_verbose()) {
      start = std::chrono::high_resolution_clock::now();
    }      
    inferKeypointLightnets(trt_lightnet, keypoint_trt_lightnets, image, visualization_config.names, keypoint_target, numWorks);
    if (profile_verbose()) {      
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "##inferKeypointLightnets: " << duration.count() << " ms " << std::endl;
    }        
  }

  // Calculate entropy and save results if required
  if (get_calc_entropy_flg()) {
    trt_lightnet->calcEntropyFromSoftmax();
    if (path_config.save_path != "not-specified") {
      fs::path dstPath(path_config.save_path);
      if (!sensor_name.empty()) {
	dstPath /= sensor_name;
      }
      if (!cam_name.empty()) {
	dstPath /= cam_name;
      }
      writeEntropy(trt_lightnet, dstPath.string(), filename);
    }
  }

  // Calculate cross-task inconsistency and save results if required
  if (get_calc_cross_task_inconsistency_flg()) {
    trt_lightnet->calcCrossTaskInconsistency(image.cols, image.rows, visualization_config.seg_colormap);
    if (path_config.save_path != "not-specified") {
      fs::path dstPath(path_config.save_path);
      if (!sensor_name.empty()) {
	dstPath /= sensor_name;
      }
      if (!cam_name.empty()) {
	dstPath /= cam_name;
      }
      writeCrossTaskInconsistency(trt_lightnet, dstPath.string(), filename);
    }
  }
 
  // Draw visualizations if not suppressed
  if (1) {
    //if (!visualization_config.dont_show) {
    if (profile_verbose()) {
      start = std::chrono::high_resolution_clock::now();
    }          
    drawLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, target);
    if (get_subnet_onnx_path() != "not-specified" && !subnet_trt_lightnets.empty()) {
      drawSubnetLightNet(trt_lightnet, image, subnet_visualization_config.colormap, subnet_visualization_config.names);
    }
    if (profile_verbose()) {      
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "#Visualization: " << duration.count() << " ms " << std::endl;
    }          
  }

  // Save results if the save flag is enabled
  if (path_config.flg_save) {
    fs::path dstPath(path_config.save_path);
    if (!sensor_name.empty()) {
      dstPath /= sensor_name;
    }
    if (!cam_name.empty()) {
      dstPath /= cam_name;
    }
    saveLightNet(trt_lightnet, image, visualization_config.colormap, visualization_config.names, dstPath.string(), filename);
  }

  // Save debug tensors if the debug save flag is enabled
  if (path_config.flg_save_debug_tensors && !path_config.save_path.empty()) {
    fs::path debugPath = fs::path(path_config.save_path) / "debug_tensors";
    saveDebugTensors(trt_lightnet, debugPath.string(), filename);
  }
}


/**
 * @brief Retrieves the calibrated sensor information for a specified camera from the given calibration data path.
 *
 * @param caliibrationInfoPath Path to the directory containing calibration data files.
 * @param camera_name Name of the camera whose calibrated information is to be retrieved.
 * @return CalibratedSensorInfo The calibrated sensor information for the specified camera.
 */
CalibratedSensorInfo getTargetCalibratedInfo(std::string caliibrationInfoPath, std::string camera_name) {
  std::string sensorFileName;
  std::string calibratedSensorFileName;
  std::vector<Sensor> sensors;
  std::vector<CalibratedSensorInfo> calibratedSensors;
  CalibratedSensorInfo targetCalibratedInfo;

  // Construct file paths for sensor.json and calibrated_sensor.json
  sensorFileName = (std::filesystem::path(caliibrationInfoPath) / "sensor.json").string();
  calibratedSensorFileName = (std::filesystem::path(caliibrationInfoPath) / "calibrated_sensor.json").string();

  try {
    // Parse the sensor data from sensor.json
    SensorParser::parse(sensorFileName, sensors);

    // Parse the calibrated sensor data from calibrated_sensor.json
    CalibratedSensorParser::parse(calibratedSensorFileName, calibratedSensors);

    // Iterate through each calibrated sensor and match it with its corresponding sensor information
    for (auto& calibratedSensor : calibratedSensors) {
      // Retrieve and set the sensor name and modality based on the sensor token
      calibratedSensor.name = SensorParser::getSensorNameFromToken(sensors, calibratedSensor.sensor_token);
      calibratedSensor.modality = SensorParser::getSensorModalityFromToken(sensors, calibratedSensor.sensor_token);

      // Set default resolution for the sensor
      calibratedSensor.width = 1920;
      calibratedSensor.height = 1280;

      // Override resolution for specific camera types
      if (calibratedSensor.name == "CAM_FRONT_NARROW" || calibratedSensor.name == "CAM_FRONT_WIDE") {
	calibratedSensor.width = 2880;
	calibratedSensor.height = 1860;
      }
      if (get_width() && get_height()) {
	calibratedSensor.width = get_width();
	calibratedSensor.height = get_height();
      }
      // If the current sensor matches the target camera name, store its information
      if (calibratedSensor.name == camera_name) {
	targetCalibratedInfo = calibratedSensor;
      }
    }
  } catch (const std::exception& e) {
    // Handle any errors that occur during parsing or processing
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // Return the calibrated sensor information for the specified camera
  return targetCalibratedInfo;
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
    .nms_threshold = static_cast<float>(get_nms_thresh()),
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
    .t4dataset_directory = get_t4_dataset_directory_path(),
    .directory = get_directory_path(),
    .video_path = get_video_path(),
    .camera_id = get_camera_id(),
    .dump_path = get_dump_path(),
    .output_path = get_output_path(),
    .flg_save = getSaveDetections(),
    .save_path = getSaveDetectionsPath(),
    .flg_save_debug_tensors = get_save_debug_tensors()
  };

  std::vector<tensorrt_lightnet::Colormap> seg_colormap = get_seg_colormap();  
  VisualizationConfig visualization_config = {
    .dont_show = is_dont_show(),
    .colormap = get_colormap(),
    .names = get_names(),
    .argmax2bgr = getArgmaxToBgr(get_seg_colormap()),
    .seg_colormap = get_seg_colormap(),
    .road_ids = get_road_ids()
  };
  
  
  int count = 0;
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
    std::make_shared<tensorrt_lightnet::TrtLightnet>(model_config, inference_config, build_config, get_depth_format());
  //Subnet configuration
  std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> subnet_trt_lightnets;
  std::vector<std::shared_ptr<tensorrt_lightnet::TrtLightnet>> keypoint_trt_lightnets;
  std::shared_ptr<fswp::FaceSwapper> fswp_model;
  
  VisualizationConfig subnet_visualization_config;
  std::vector<std::string> target, keypoint_target;
  std::vector<std::string> bluron = get_bluron_names();
  int numWorks = get_workers();//omp_get_max_threads();

  if (get_subnet_onnx_path() != "not-specified") {
    ModelConfig subnet_model_config = {
      .model_path = get_subnet_onnx_path(),
      .num_class = get_subnet_classes(),
      .score_threshold = static_cast<float>(get_subnet_score_thresh()),
      .anchors = get_subnet_anchors(),
      .num_anchors = get_subnet_num_anchors(),
      .nms_threshold = 0.25f // Assuming this value is fixed or retrieved similarly.
    };
    subnet_visualization_config = {
      .dont_show = is_dont_show(),
      .colormap = get_subnet_colormap(),
      .names = get_subnet_names(),
      .argmax2bgr = getArgmaxToBgr(seg_colormap)
    };
    target = get_target_names();
    for (int w = 0; w < numWorks; w++) {
      if (build_config.dla_core_id >= 2) {
	//use multiple dlas [DLA0 and DLA1]
	build_config.dla_core_id = (int)w/2;
      }
      subnet_trt_lightnets.push_back(
				     std::make_shared<tensorrt_lightnet::TrtLightnet>(subnet_model_config, inference_config, build_config, get_depth_format()));
    }
  }

  if (get_keypoint_onnx_path() != "not-specified") {
    ModelConfig keypoint_model_config = {
      .model_path = get_keypoint_onnx_path(),
      .num_class = get_classes(),
      .score_threshold = static_cast<float>(get_score_thresh()),
      .anchors = get_anchors(),
      .num_anchors = get_num_anchors(),
      .nms_threshold = static_cast<float>(get_nms_thresh()),      
    };
    keypoint_target = get_keypoint_names();
    for (int w = 0; w < numWorks; w++) {
      if (build_config.dla_core_id >= 2) {
	//use multiple dlas [DLA0 and DLA1]
	build_config.dla_core_id = (int)w/2;
      }
      keypoint_trt_lightnets.push_back(
				       std::make_shared<tensorrt_lightnet::TrtLightnet>(keypoint_model_config, inference_config, build_config, get_depth_format()));
    }
  }  

  if (get_fswp_onnx_path() != "not-specified") {
    int bs = 8;
    std::filesystem::path fswp_path(get_fswp_onnx_path());
    fswp_model = std::make_shared<fswp::FaceSwapper>(fswp_path, build_config, bs, "fp16");
    target = get_target_names();
  }
  auto p2i = pcd2image::Pcd2image();
  CalibratedSensorInfo targetCalibratedInfo;
  if (get_lidar_range_image_flg()) {
    std::string caliibrationInfoPath = get_sensor_config();
    if (!path_config.t4dataset_directory.empty()) {
      fs::path t4_dataset(path_config.t4dataset_directory);
      fs::path t4_annotation = t4_dataset / "annotation";
      caliibrationInfoPath = t4_annotation.string();
    }
    targetCalibratedInfo = getTargetCalibratedInfo(caliibrationInfoPath, get_camera_name());
  }
  std::chrono::high_resolution_clock::time_point start, end;
  
  if (!path_config.t4dataset_directory.empty()) {
    std::cout << path_config.t4dataset_directory << std::endl;
    fs::path t4_dataset(path_config.t4dataset_directory);
    fs::path t4_data = t4_dataset / "data";
    for (const auto& dir : fs::directory_iterator(t4_data.string())) {
      fs::path cam_data(dir);
      std::string sensor_name = cam_data.filename();
      std::string cam_name = "";
      std::cout << " |_" << dir.path() << " - " << sensor_name <<std::endl;
      if ((!get_lidar_range_image_flg() && sensor_name.find("CAM") != std::string::npos) ||
	  (get_lidar_range_image_flg() && sensor_name.find("LIDAR_CONCAT") != std::string::npos)) {
	for (const auto& file : fs::directory_iterator(cam_data.string())) {
	  if (!get_lidar_range_image_flg() && file.path().extension() != ".jpg") continue;
	  if (get_lidar_range_image_flg() && file.path().extension() != ".bin") continue;	  
	  std::cout << "Inference from " << file.path() << std::endl;
	  cv::Mat image;
	  if (get_lidar_range_image_flg()) {
	    //get range image from lidar pcd
	    image = p2i.makeRangeImageFromCalibration(file.path(), targetCalibratedInfo, 120.0);
	    cam_name = targetCalibratedInfo.name;
	  } else {
	    image = cv::imread(file.path(), cv::IMREAD_COLOR);
	  }
	  inferLightNetPipeline(trt_lightnet, subnet_trt_lightnets, keypoint_trt_lightnets, fswp_model, image, visualization_config, subnet_visualization_config, target, keypoint_target, bluron, path_config, file.path().filename(), sensor_name, cam_name);
	  
	  if (!visualization_config.dont_show) {
	    if (image.rows > 1280 && image.cols > 1920) {
	      cv::resize(image, image, cv::Size(1920, 1280), 0, 0, cv::INTER_LINEAR);
	    }
	    cv::imshow("inference", image);	
	    cv::waitKey(0);
	  }
	  count++;
	}
      }
    } 
  } else if (!path_config.directory.empty()) {
    for (const auto& file : fs::directory_iterator(path_config.directory)) {
      if (profile_verbose()) {
	start = std::chrono::high_resolution_clock::now();
      }      
      std::cout << "Inference from " << file.path() << std::endl;
      cv::Mat image = cv::imread(file.path(), cv::IMREAD_COLOR);
      inferLightNetPipeline(trt_lightnet, subnet_trt_lightnets, keypoint_trt_lightnets, fswp_model, image, visualization_config, subnet_visualization_config, target, keypoint_target, bluron, path_config, file.path().filename(), "", "");
      if (!visualization_config.dont_show) {
	if (image.rows > 1280 && image.cols > 1920) {
	  cv::resize(image, image, cv::Size(1920, 1280), 0, 0, cv::INTER_LINEAR);
	}
	cv::imshow("inference", image);	
	cv::waitKey(0);
      }
      if (profile_verbose()) {      
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "#Duration: " << duration.count() << " ms " << " FPS: " << 1000/duration.count()  << std::endl;
      }      
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
      if (profile_verbose()) {
	start = std::chrono::high_resolution_clock::now();
      }
      video >> image;
      if (image.empty()) break;     
      std::ostringstream sout;
      sout << std::setfill('0') << std::setw(6) << count;	  
      std::string name = "frame_" + sout.str() + ".jpg";	              
      inferLightNetPipeline(trt_lightnet, subnet_trt_lightnets, keypoint_trt_lightnets, fswp_model, image, visualization_config, subnet_visualization_config, target, keypoint_target, bluron, path_config, name, "", "");
      
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
      if (profile_verbose()) {      
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<long long, std::milli> duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "#Duration: " << duration.count() << " ms " << " FPS: " << 1000/duration.count()  << std::endl;
      }
    }
  }  
  
  if (inference_config.profile) {
    trt_lightnet->printProfiling();
  }

  return 0;
}

// Copyright 2025 TIER IV, Inc.
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

#include <nlohmann/json.hpp>
#include <tensorrt_lightnet/tensorrt_lightnet.hpp>
#include <cstdint>

constexpr int MAX_STRING_SIZE = 256;
constexpr int MAX_ANCHORS = 40;
constexpr int MAX_CALIBRATION_PATH = 512;
constexpr int MAX_PRECISION_SIZE = 64;

/**
 * @brief C-compatible structure for storing keypoint information.
 */
struct KeypointInfoC {
  float id_prob;      ///< Probability of keypoint presence.
  bool isOccluded;    ///< Whether the keypoint is occluded.
  float attr_prob;    ///< Attribute probability.
  int lx0, ly0, lx1, ly1; ///< Left keypoint coordinates.
  int rx0, ry0, rx1, ry1; ///< Right keypoint coordinates.
  int bot, left;      ///< Additional keypoint attributes.
};

/**
 * @brief C-compatible structure for storing bounding box coordinates.
 */
struct BBoxC {
  float x1, y1, x2, y2; ///< Bounding box coordinates.
};

/**
 * @brief C-compatible structure for storing bounding box information.
 */
struct BBoxInfoC {
  BBoxC box;         ///< Bounding box coordinates.
  int label;         ///< Label ID.
  int classId;       ///< Class ID.
  float prob;        ///< Probability of detection.
  bool isHierarchical; ///< Whether the detection is hierarchical.
  int subClassId;    ///< Subclass ID if applicable.
  float sin;         ///< Sine value for angle-based detections.
  float cos;         ///< Cosine value for angle-based detections.
  KeypointInfoC* keypoints; ///< Pointer to keypoints data.
  int num_keypoints; ///< Number of keypoints.
  char attribute[MAX_STRING_SIZE]; ///< Attribute information.
  float attribute_prob; ///< Probability of the attribute.
  int batch_index;  
};

/**
 * @brief C-compatible structure for model configuration.
 */
typedef struct ModelConfigC {
  char model_path[MAX_STRING_SIZE]; ///< Path to the model file.
  int num_class;          ///< Number of classes.
  float score_threshold;  ///< Threshold for classification scores.
  int anchors[MAX_ANCHORS]; ///< Fixed-size array of anchor sizes.
  int anchor_elements;    ///< Number of elements in the anchor array.
  int num_anchors;        ///< Number of anchors.
  float nms_threshold;    ///< Non-Maximum Suppression (NMS) threshold.
  const char** names;     ///< Pointer to an array of class names.
  int num_names;          ///< Number of names.
  const int* detection_colormap; ///< Pointer to a flattened array of colormap values.
  int detection_colormap_size; ///< Total number of elements in colormap (num_classes * 3).
} ModelConfigC;

/**
 * @brief C-compatible structure for inference configuration.
 */
typedef struct InferenceConfigC {
  char precision[MAX_PRECISION_SIZE]; ///< Precision mode (e.g., FP16, INT8).
  bool profile;         ///< Enable profiling.
  bool sparse;          ///< Use sparse computation.
  int dla_core_id;      ///< DLA core ID (if applicable).
  bool use_first_layer; ///< Enable first-layer optimizations.
  bool use_last_layer;  ///< Enable last-layer optimizations.
  int batch_size;       ///< Inference batch size.
  double scale;         ///< Input scaling factor.
  char calibration_images[MAX_CALIBRATION_PATH]; ///< Path to calibration images.
  char calibration_type[MAX_PRECISION_SIZE]; ///< Calibration type (e.g., KL, entropy).
  int max_batch_size;   ///< Maximum batch size.
  int min_batch_size;   ///< Minimum batch size.
  int optimal_batch_size; ///< Optimal batch size.
  size_t workspace_size; ///< Workspace size for TensorRT execution.
} InferenceConfigC;

typedef struct ColormapC_
{
  int id;
  char name[50];
  unsigned char color[3];
  bool is_dynamic;
} ColormapC;


// Expose functions to Python via extern "C"
extern "C" {
  std::vector<BBoxInfoC> bbox_c_array;
  std::string annotation_str = "";
  /**
   * @brief Initializes a ModelConfigC structure with given parameters.
   * 
   * @param config Pointer to ModelConfigC structure to initialize.
   * @param model_path Path to the model file.
   * @param num_class Number of classes.
   * @param score_threshold Classification score threshold.
   * @param anchors Array of anchor values.
   * @param num_anchors Number of anchors.
   * @param nms_threshold Non-Maximum Suppression (NMS) threshold.
   */
  void initialize_model_config(ModelConfigC* config, const char* model_path, int num_class,
			       float score_threshold, const int* anchors, int num_anchors,
			       float nms_threshold) {
    if (!config || !model_path || !anchors) return;

    strncpy(config->model_path, model_path, sizeof(config->model_path) - 1);
    config->model_path[sizeof(config->model_path) - 1] = '\0';

    config->num_class = num_class;
    config->score_threshold = score_threshold;
    config->num_anchors = num_anchors;
    config->nms_threshold = nms_threshold;

    for (int i = 0; i < num_anchors && i < MAX_ANCHORS; ++i) {
      config->anchors[i] = anchors[i];
    }
  }

  /**
   * @brief Prints the contents of a ModelConfigC structure for debugging.
   * 
   * @param config Pointer to ModelConfigC structure to print.
   */
  void print_model_config(const ModelConfigC* config) {
    if (!config) return;

    std::cout << "Model Path: " << config->model_path << "\n";
    std::cout << "Num Class: " << config->num_class << "\n";
    std::cout << "Score Threshold: " << config->score_threshold << "\n";
    std::cout << "Num Anchor Values: " << config->num_anchors << "\n";
    std::cout << "NMS Threshold: " << config->nms_threshold << "\n";
    std::cout << "Anchors: ";
    for (int i = 0; i < config->num_anchors; ++i) {
      std::cout << config->anchors[i] << " ";
    }
    std::cout << std::endl;
  }

  /**
   * @brief Converts a ModelConfig structure to a ModelConfigC structure.
   * 
   * @param src Pointer to source ModelConfig structure.
   * @param dest Pointer to destination ModelConfigC structure.
   */
  void to_c_model_config(const ModelConfig* src, ModelConfigC* dest) {
    if (!src || !dest) return;

    strncpy(dest->model_path, src->model_path.c_str(), sizeof(dest->model_path) - 1);
    dest->model_path[sizeof(dest->model_path) - 1] = '\0';

    dest->num_class = src->num_class;
    dest->score_threshold = src->score_threshold;
    dest->nms_threshold = src->nms_threshold;

    dest->num_anchors = std::min(static_cast<int>(src->anchors.size()), MAX_ANCHORS);
    for (int i = 0; i < dest->num_anchors; ++i) {
      dest->anchors[i] = src->anchors[i];
    }
  }

  /**
   * @brief Converts a ModelConfigC structure to a ModelConfig structure.
   * 
   * @param src Pointer to source ModelConfigC structure.
   * @param dest Pointer to destination ModelConfig structure.
   */
  void to_cpp_model_config(const ModelConfigC* src, ModelConfig* dest) {
    if (!src || !dest) return;

    dest->model_path = std::string(src->model_path);
    dest->num_class = src->num_class;
    dest->score_threshold = src->score_threshold;
    dest->nms_threshold = src->nms_threshold;

    dest->anchors.clear();
    for (int i = 0; i < src->anchor_elements; ++i) {
      dest->anchors.push_back(src->anchors[i]);
    }
    dest->num_anchors = src->num_anchors;
  }

  /**
   * @brief Prints the contents of a ModelConfig structure for debugging.
   * 
   * @param config Pointer to ModelConfig structure to print.
   */
  void print_cpp_model_config(const ModelConfig* config) {
    if (!config) return;

    std::cout << "Model Path: " << config->model_path << "\n";
    std::cout << "Num Class: " << config->num_class << "\n";
    std::cout << "Score Threshold: " << config->score_threshold << "\n";
    std::cout << "Num Anchors: " << config->anchors.size() << "\n";
    std::cout << "NMS Threshold: " << config->nms_threshold << "\n";
    std::cout << "Anchors: ";
    for (const auto& anchor : config->anchors) {
      std::cout << anchor << " ";
    }
    std::cout << std::endl;
  }

  /**
   * @brief Converts an InferenceConfig structure to a C-compatible InferenceConfigC structure.
   * 
   * @param src Pointer to the source InferenceConfig structure.
   * @param dest Pointer to the destination InferenceConfigC structure.
   */
  extern void to_c_inference_config(const InferenceConfig* src, InferenceConfigC* dest) {
    if (!src || !dest) return;

    strncpy(dest->precision, src->precision.c_str(), sizeof(dest->precision) - 1);
    dest->precision[sizeof(dest->precision) - 1] = '\0';

    dest->profile = src->profile;
    dest->sparse = src->sparse;
    dest->dla_core_id = src->dla_core_id;
    dest->use_first_layer = src->use_first_layer;
    dest->use_last_layer = src->use_last_layer;
    dest->batch_size = src->batch_size;
    dest->scale = src->scale;

    strncpy(dest->calibration_images, src->calibration_images.c_str(), sizeof(dest->calibration_images) - 1);
    dest->calibration_images[sizeof(dest->calibration_images) - 1] = '\0';

    strncpy(dest->calibration_type, src->calibration_type.c_str(), sizeof(dest->calibration_type) - 1);
    dest->calibration_type[sizeof(dest->calibration_type) - 1] = '\0';

    dest->max_batch_size = src->batch_config[2];
    dest->min_batch_size = src->batch_config[0];
    dest->optimal_batch_size = src->batch_config[1];

    dest->workspace_size = src->workspace_size;
  }

  /**
   * @brief Converts an InferenceConfigC structure to an InferenceConfig structure.
   * 
   * @param src Pointer to the source InferenceConfigC structure.
   * @param dest Pointer to the destination InferenceConfig structure.
   */
  extern void to_cpp_inference_config(const InferenceConfigC* src, InferenceConfig* dest) {
    if (!src || !dest) {
      std::cerr << "Error: Null pointer provided." << std::endl;
      return;
    }

    try {
      dest->precision = std::string(src->precision);
      dest->profile = src->profile;
      dest->sparse = src->sparse;
      dest->dla_core_id = src->dla_core_id;
      dest->use_first_layer = src->use_first_layer;
      dest->use_last_layer = src->use_last_layer;
      dest->batch_size = src->batch_size;
      dest->scale = src->scale;

      dest->calibration_images = src->calibration_images[0] != '\0' ? std::string(src->calibration_images) : "";
      dest->calibration_type = src->calibration_type[0] != '\0' ? std::string(src->calibration_type) : "";

      if (dest->batch_config.size() >= 3) {
	dest->batch_config[2] = src->max_batch_size;
	dest->batch_config[0] = src->min_batch_size;
	dest->batch_config[1] = src->optimal_batch_size;
      } else {
	std::cerr << "Error: BatchConfig size is insufficient." << std::endl;
      }

      dest->workspace_size = src->workspace_size;
    } catch (const std::exception& e) {
      std::cerr << "Exception occurred: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "Unknown exception occurred." << std::endl;
    }
  }

  /**
   * @brief Prints the contents of an InferenceConfig structure for debugging.
   * 
   * @param config Pointer to the InferenceConfig structure.
   */
  extern void print_inference_config(const InferenceConfig* config) {
    if (!config) return;

    std::cout << "Precision: " << config->precision << "\n";
    std::cout << "Profile: " << config->profile << "\n";
    std::cout << "Sparse: " << config->sparse << "\n";
    std::cout << "DLA Core ID: " << config->dla_core_id << "\n";
    std::cout << "Use First Layer: " << config->use_first_layer << "\n";
    std::cout << "Use Last Layer: " << config->use_last_layer << "\n";
    std::cout << "Batch Size: " << config->batch_size << "\n";
    std::cout << "Scale: " << config->scale << "\n";
    std::cout << "Calibration Type: " << config->calibration_type << "\n";
  }

  /**
   * @brief Prints the contents of an InferenceConfigC structure for debugging.
   * 
   * @param config Pointer to the InferenceConfigC structure.
   */
  void print_inference_config_c(const InferenceConfigC* config) {
    if (!config) return;

    std::cout << "Precision: " << config->precision << "\n";
    std::cout << "Profile: " << config->profile << "\n";
    std::cout << "Sparse: " << config->sparse << "\n";
    std::cout << "DLA Core ID: " << config->dla_core_id << "\n";
    std::cout << "Use First Layer: " << config->use_first_layer << "\n";
    std::cout << "Use Last Layer: " << config->use_last_layer << "\n";
    std::cout << "Batch Size: " << config->batch_size << "\n";
    std::cout << "Scale: " << config->scale << "\n";
    std::cout << "Calibration Images: " << config->calibration_images << "\n";
    std::cout << "Calibration Type: " << config->calibration_type << "\n";
    std::cout << "Batch Config (max/min/optimal): " << config->max_batch_size << "/"
	      << config->min_batch_size << "/" << config->optimal_batch_size << "\n";
    std::cout << "Workspace Size: " << config->workspace_size << " bytes\n";
  }

/**
 * @brief Create a TrtLightnet instance.
 * 
 * @param modelConfigC Pointer to the C-style model configuration.
 * @param inferenceConfigC Pointer to the C-style inference configuration.
 * @param buildConfigC Pointer to the C-style build configuration.
 * @return void* Pointer to the created TrtLightnet instance.
 */
void* create_trt_lightnet(const ModelConfigC *modelConfigC, const InferenceConfigC *inferenceConfigC,
                          const BuildConfigC *buildConfigC) {
    ModelConfig modelConfig;
    to_cpp_model_config(modelConfigC, &modelConfig);

    InferenceConfig inferenceConfig;
    to_cpp_inference_config(inferenceConfigC, &inferenceConfig);

    tensorrt_common::BuildConfig buildConfig;
    copy_to_cpp_build_config(buildConfigC, &buildConfig);
    
    auto instance = new std::shared_ptr<tensorrt_lightnet::TrtLightnet>(
        std::make_shared<tensorrt_lightnet::TrtLightnet>(modelConfig, inferenceConfig, buildConfig, "magma"));
    
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);

    if (lightnet->trt_common_ == nullptr) {
      std::cout << "Fail lightnet->trt_common_  " << lightnet->trt_common_ <<  std::endl;
      return nullptr;
    }
    
    // Set model names
    std::vector<std::string> names;
    for (int i = 0; i < modelConfigC->num_names; ++i) {
        if (modelConfigC->names[i]) {
            names.emplace_back(modelConfigC->names[i]);
        }
    }
    lightnet->setNames(names);

    // Convert detection_colormap from flattened array to std::vector<std::vector<int>>
    std::vector<std::vector<int>> detection_colormap;
    for (int i = 0; i < modelConfigC->detection_colormap_size; i += 3) {
        detection_colormap.push_back({modelConfigC->detection_colormap[i],
                                      modelConfigC->detection_colormap[i + 1],
                                      modelConfigC->detection_colormap[i + 2]});
    }
    lightnet->setDetectionColormap(detection_colormap);

    return static_cast<void*>(instance);
}

  /**
   * @brief Destroy a TrtLightnet instance.
   * 
   * @param instance Pointer to the instance to be destroyed.
   */
  void destroy_trt_lightnet(void* instance) {
    if (instance) {
      delete static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    }
  }

  /**
   * @brief Get the input size of the TrtLightnet model.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param batch Pointer to store the batch size.
   * @param chan Pointer to store the number of channels.
   * @param height Pointer to store the height.
   * @param width Pointer to store the width.
   */
  void trt_lightnet_get_input_size(void* instance, int* batch, int* chan, int* height, int* width) {
    if (!instance || !batch || !chan || !height || !width) {
      std::cerr << "Error: Null pointer in trt_lightnet_get_input_size." << std::endl;
      return;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    lightnet->getInputSize(batch, chan, height, width);
  }

  /**
   * @brief Display an image using OpenCV.
   * 
   * @param img_data Pointer to the image data.
   * @param width Width of the image.
   * @param height Height of the image.
   * @param channels Number of channels in the image.
   */
  void display_image(unsigned char* img_data, int width, int height, int channels) {
    cv::Mat image(height, width, CV_8UC3, img_data);
    cv::imshow("Image from Python", image);
    cv::waitKey(0);
  }

  /**
   * @brief Resize an image to a new size.
   * 
   * @param img_data Pointer to the original image data.
   * @param width Original image width.
   * @param height Original image height.
   * @param channels Number of channels in the image.
   * @param new_width Desired new width.
   * @param new_height Desired new height.
   * @param resized_img_data Pointer to store the resized image data.
   */
  void resize_image(unsigned char* img_data, int width, int height, int channels, int new_width, int new_height, unsigned char* resized_img_data) {
    cv::Mat image(height, width, CV_8UC3, img_data);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    std::memcpy(resized_img_data, resized_image.data, new_width * new_height * channels);
  }

  /**
   * @brief Perform inference with the TrtLightnet model.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param img_data Pointer to the image data.
   * @param width Image width.
   * @param height Image height.
   * @param cuda If true, use GPU for preprocessing; otherwise, use CPU.
   */
  void infer_lightnet_wrapper(void* instance, unsigned char* img_data, int width, int height, bool cuda) {
    if (!instance || !img_data) {
      std::cerr << "Error: Null pointer in infer_lightnet_wrapper." << std::endl;
      return;
    }

    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    cv::Mat image(height, width, CV_8UC3, img_data);
    // Preprocessing
    if (cuda) {
      lightnet->preprocess_gpu({image});
    } else {
      lightnet->preprocess({image});
    }

    // Inference
    lightnet->doInference();

    // Postprocessing
    lightnet->makeBbox(image.rows, image.cols);
    lightnet->makeTopIndex();    
  }

  /**
   * @brief Convert std::vector<BBoxInfo> to a C-compatible array.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param size Pointer to store the size of the resulting array.
   * @return BBoxInfoC* Pointer to the converted array.
   */
  BBoxInfoC* get_bbox_array(void* instance, int* size) {
    if (!instance) {
      std::cerr << "Error: Null pointer in get_bbox_array." << std::endl;
      return nullptr;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    std::vector<tensorrt_lightnet::BBoxInfo> bbox_ = lightnet->getBbox();

    bbox_c_array.clear();
    for (const auto& bbox : bbox_) {
      BBoxInfoC bbox_c;
      bbox_c.box = {bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2};
      bbox_c.label = bbox.label;
      bbox_c.classId = bbox.classId;
      bbox_c.prob = bbox.prob;
      bbox_c.isHierarchical = bbox.isHierarchical;
      bbox_c.subClassId = bbox.subClassId;
      bbox_c.sin = bbox.sin;
      bbox_c.cos = bbox.cos;

      // Convert keypoints
      bbox_c.num_keypoints = bbox.keypoint.size();
      static std::vector<KeypointInfoC> keypoints_c;
      keypoints_c.clear();
      for (const auto& kp : bbox.keypoint) {
	keypoints_c.push_back({kp.id_prob, kp.isOccluded, kp.attr_prob,
	    kp.lx0, kp.ly0, kp.lx1, kp.ly1, kp.rx0, kp.ry0, kp.rx1, kp.ry1, kp.bot, kp.left});
      }
      bbox_c.keypoints = keypoints_c.data();

      bbox_c_array.push_back(bbox_c);
    }
    *size = bbox_c_array.size();
    return bbox_c_array.data();
  }

  int get_top_index(void* instance)
  {
    if (!instance) {
      std::cerr << "Error: Null pointer in get_bbox_array." << std::endl;
      return -1;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    int top_index = lightnet->getMaxIndex();
    return top_index;
  }
  
  /**
   * @brief Convert an array of RGB values to a vector of cv::Vec3b.
   * 
   * @param rgb_values Pointer to the array of RGB values.
   * @param length Length of the array.
   * @return std::vector<cv::Vec3b>* Pointer to the allocated vector.
   */
  std::vector<cv::Vec3b>* convert_to_vec3b(uint8_t* rgb_values, size_t length) {
    auto* argmax2bgr = new std::vector<cv::Vec3b>;

    for (size_t i = 0; i < length; i += 3) {
      cv::Vec3b color(rgb_values[i], rgb_values[i + 1], rgb_values[i + 2]);
      argmax2bgr->push_back(color);
    }
    
    return argmax2bgr;
  }

  /**
   * @brief Free memory allocated for a vector of cv::Vec3b.
   * 
   * @param argmax2bgr Pointer to the vector to be freed.
   */
  void free_vec3b(std::vector<cv::Vec3b>* argmax2bgr) {
    delete argmax2bgr;
  }

  /**
   * @brief Invoke the makeMask function on a TrtLightnet instance.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param argmax2bgr Pointer to the color map vector.
   */
  void makeMask(void* instance, std::vector<cv::Vec3b>* argmax2bgr) {
    if (!argmax2bgr) return;
    if (!instance) {
      std::cerr << "Error: Null pointer in makeMask." << std::endl;
      return;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    lightnet->makeMask(*argmax2bgr);
  }

  /**
   * @brief Retrieve masks from a TrtLightnet instance.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @return std::vector<cv::Mat>* Pointer to the allocated vector of masks.
   */
  std::vector<cv::Mat>* get_masks(void* instance) {
    if (!instance) {
      std::cerr << "Error: Null pointer in get_masks." << std::endl;
      return nullptr;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    return new std::vector<cv::Mat>(lightnet->getMask());
  }

  /**
   * @brief Get the number of masks in the vector.
   * 
   * @param masks Pointer to the mask vector.
   * @return size_t The number of masks.
   */
  size_t get_mask_count(std::vector<cv::Mat>* masks) {
    return masks ? masks->size() : 0;
  }

  /**
   * @brief Get a pointer to the data of a mask.
   * 
   * @param masks Pointer to the mask vector.
   * @param index Index of the mask.
   * @return uint8_t* Pointer to the mask data or nullptr if invalid.
   */
  uint8_t* get_mask_data(std::vector<cv::Mat>* masks, size_t index) {
    if (!masks || index >= masks->size()) return nullptr;
    return masks->at(index).data;
  }

  /**
   * @brief Get the shape of a mask.
   * 
   * @param masks Pointer to the mask vector.
   * @param index Index of the mask.
   * @param rows Pointer to store the number of rows.
   * @param cols Pointer to store the number of columns.
   * @param channels Pointer to store the number of channels.
   */
  void get_mask_shape(std::vector<cv::Mat>* masks, size_t index, int* rows, int* cols, int* channels) {
    if (!masks || index >= masks->size()) return;
    *rows = masks->at(index).rows;
    *cols = masks->at(index).cols;
    *channels = masks->at(index).channels();
  }

  /**
   * @brief Free memory allocated for the mask vector.
   * 
   * @param masks Pointer to the mask vector to be freed.
   */
  void free_masks(std::vector<cv::Mat>* masks) {
    delete masks;
  }

  /**
   * @brief Get a polygon annotation string for segmentation.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param width Image width.
   * @param height Image height.
   * @param colormap_array Pointer to the color map array.
   * @param length Length of the color map array.
   * @param image_name Name of the image.
   * @return const char* C-string containing the annotation.
   */
  const char* get_polygon_str(void* instance, int width, int height, ColormapC* colormap_array, size_t length, const char *image_name) {
    if (!instance) return nullptr;
    std::vector<tensorrt_lightnet::Colormap> colormaps;

    for (size_t i = 0; i < length; i++) {
      tensorrt_lightnet::Colormap colormap = {
	colormap_array[i].id,
	std::string(colormap_array[i].name),
	{(unsigned char)colormap_array[i].color[0], (unsigned char)colormap_array[i].color[1], (unsigned char)colormap_array[i].color[2]},
	colormap_array[i].is_dynamic
      };
      colormaps.push_back(colormap);
    }

    std::string filename(image_name);
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    annotation_str = lightnet->getSegmentationAnnotationStr(filename, width, height, colormaps);
    return annotation_str.c_str();
  }

  void infer_batches(
    void* instance,
    unsigned char** imgs,
    int* heights,
    int* widths,
    int* channels,
    int batch_size,
    BBoxInfoC** out_bboxes,
    int* out_bbox_count
		     )
  {		     
    std::vector<cv::Mat> images;
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    for (int i = 0; i < batch_size; ++i) {
      int h = heights[i];
      int w = widths[i];
      int c = channels[i];

      cv::Mat img;
      if (c == 1) {
	img = cv::Mat(h, w, CV_8UC1, imgs[i]);
      } else if (c == 3) {
	img = cv::Mat(h, w, CV_8UC3, imgs[i]);
      } else {
	continue;
      }
      images.push_back(img);
    }

    if (images.empty()) {
        *out_bboxes = nullptr;
        *out_bbox_count = 0;
        return;
    }    
    
    lightnet->preprocess(images);
    lightnet->doInference(static_cast<int>(images.size()));


    std::vector<BBoxInfoC> all_results;

    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        auto bboxes = lightnet->getBbox(images[i].rows, images[i].cols, i);
        for (const auto& b : bboxes) {
            BBoxInfoC out{};
            out.box = { b.box.x1, b.box.y1, b.box.x2, b.box.y2 };
            out.label = b.label;
            out.classId = b.classId;
            out.prob = b.prob;
            out.isHierarchical = b.isHierarchical;
            out.subClassId = b.subClassId;
            out.sin = b.sin;
            out.cos = b.cos;
            //out.num_keypoints = static_cast<int>(b.keypoints.size());
            out.batch_index = i;
            //std::strncpy(out.attribute, b.attribute.c_str(), MAX_STRING_SIZE - 1);
            //out.attribute[MAX_STRING_SIZE - 1] = '\0';
            out.attribute_prob = 0.0;
	    //out.keypoints = nullptr;
            all_results.push_back(out);
        }
    }

    // Copy results to C heap for Python access
    int total = all_results.size();
    *out_bbox_count = total;
    *out_bboxes = static_cast<BBoxInfoC*>(malloc(sizeof(BBoxInfoC) * total));
    std::memcpy(*out_bboxes, all_results.data(), sizeof(BBoxInfoC) * total); 
}  

/**
 * @brief Perform inference on a subset of bounding boxes detected by the main network using a subnet.
 *
 * @param lightnet Pointer to the main TrtLightnet instance.
 * @param subnet Pointer to the subnet TrtLightnet instance.
 * @param image Input image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param cuda Flag indicating whether to use CUDA.
 * @param target Array of target class names.
 * @param count Number of target classes.
 */
void infer_subnet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> lightnet, std::shared_ptr<tensorrt_lightnet::TrtLightnet> subnet, cv::Mat &image, int width, int height, bool cuda, char** target, int count) {
    lightnet->clearSubnetBbox();
    std::vector<tensorrt_lightnet::BBoxInfo> bbox = lightnet->getBbox();
    auto names = lightnet->getNames();
    int num = static_cast<int>(bbox.size());
    std::vector<tensorrt_lightnet::BBoxInfo> subnetBbox;

    for (int i = 0; i < num; i++) {
        auto b = bbox[i];
        bool flg = false;

        for (int t = 0; t < count; t++) {
            if (std::string(target[t]) == names[b.classId]) {
                flg = true;
                break;
            }
        }

        if (!flg) continue;

        cv::Rect roi(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1);
        cv::Mat cropped = image(roi);
        subnet->preprocess({cropped});
        subnet->doInference();
        subnet->makeBbox(cropped.rows, cropped.cols);

        auto bb = subnet->getBbox();
        for (auto &box : bb) {
            box.box.x1 += b.box.x1;
            box.box.y1 += b.box.y1;
            box.box.x2 += b.box.x1;
            box.box.y2 += b.box.y1;
        }
        subnetBbox.insert(subnetBbox.end(), bb.begin(), bb.end());
    }
    lightnet->appendSubnetBbox(subnetBbox);
    lightnet->doNonMaximumSuppressionForSubnetBbox();
}

/**
 * @brief Perform batch inference on a subset of bounding boxes detected by the main network using a subnet.
 *
 * @param lightnet Pointer to the main TrtLightnet instance.
 * @param subnet Pointer to the subnet TrtLightnet instance.
 * @param image Input image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param cuda Flag indicating whether to use CUDA.
 * @param target Array of target class names.
 * @param count Number of target classes.
 */
void infer_batch_subnet(std::shared_ptr<tensorrt_lightnet::TrtLightnet> lightnet, std::shared_ptr<tensorrt_lightnet::TrtLightnet> subnet, cv::Mat &image, int width, int height, bool cuda, char** target, int count) {
    lightnet->clearSubnetBbox();
    std::vector<tensorrt_lightnet::BBoxInfo> bbox = lightnet->getBbox();
    auto names = lightnet->getNames();
    int num = static_cast<int>(bbox.size());
    int maxBatchSize = subnet->getBatchSize();    

    std::vector<tensorrt_lightnet::BBoxInfo> subnetBbox;
    std::vector<cv::Mat> cropped;

    for (int bs = 0; bs < num; bs++) {
        auto b = bbox[bs];
        bool flg = false;
        for (int t = 0; t < count; t++) {
            if (std::string(target[t]) == names[b.classId]) {
                flg = true;
                break;
            }
        }
	
        if (!flg) continue;

        cv::Rect roi(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1);
        cropped.push_back(image(roi));
	if (static_cast<int>(cropped.size()) > maxBatchSize) {
	  break;
	}	
    }

    if (!cropped.size()) {
      return;
    }
    
    subnet->preprocess(cropped);
    subnet->doInference(static_cast<int>(cropped.size()));

    int actual_batch_size = 0;
    for (int bs = 0; bs < num; bs++) {
        auto b = bbox[bs];
        bool flg = false;

        for (int t = 0; t < count; t++) {
            if (std::string(target[t]) == names[b.classId]) {
                flg = true;
                break;
            }
        }

        if (!flg) continue;

        auto bb = subnet->getBbox(cropped[actual_batch_size].rows, cropped[actual_batch_size].cols, actual_batch_size);
        for (auto &box : bb) {
            box.box.x1 += b.box.x1;
            box.box.y1 += b.box.y1;
            box.box.x2 += b.box.x1;
            box.box.y2 += b.box.y1;
        }
        subnetBbox.insert(subnetBbox.end(), bb.begin(), bb.end());
        actual_batch_size++;
	if (static_cast<int>(actual_batch_size) >= maxBatchSize) {
	  break;
	}		
    }    
    lightnet->appendSubnetBbox(subnetBbox);
    lightnet->doNonMaximumSuppressionForSubnetBbox();
}


  

  
  /**
   * @brief Perform inference with the TrtLightnet model.
   * 
   * @param instance Pointer to the TrtLightnet instance.
   * @param img_data Pointer to the image data.
   * @param width Image width.
   * @param height Image height.
   * @param cuda If true, use GPU for preprocessing; otherwise, use CPU.
   */
  void infer_multi_stage_lightnet_wrapper(void* instance, void* sub_instance, unsigned char* img_data, int width, int height, bool cuda, char** target, int count)
  {

    if (!instance || !img_data || !sub_instance) {
      std::cerr << "Error: Null pointer in infer_lightnet_wrapper." << std::endl;
      return;
    }

    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    auto subnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(sub_instance);    
    cv::Mat image(height, width, CV_8UC3, img_data);

    // Preprocessing
    if (cuda) {
      lightnet->preprocess_gpu({image});
    } else {
      lightnet->preprocess({image});
    }

    // Inference
    lightnet->doInference();

    // Postprocessing
    lightnet->makeBbox(image.rows, image.cols);


    int maxBatchSize = subnet->getBatchSize();
    if (maxBatchSize > 1) {
      infer_batch_subnet(lightnet, subnet, image, width, height, cuda, target, count); 
    } else {
      infer_subnet(lightnet, subnet, image, width, height, cuda, target, count);       
    }
  }
  
 /**
 * @brief Retrieve the bounding boxes detected by the subnet in a C-compatible array.
 * 
 * @param instance Pointer to the main TrtLightnet instance.
 * @param size Output parameter for the size of the bounding box array.
 * @return Pointer to an array of BBoxInfoC.
 */
BBoxInfoC* get_subnet_bbox_array(void* instance, int* size) {
    if (!instance) {
        std::cerr << "Error: Null pointer in get_bbox_array." << std::endl;
        return nullptr;
    }
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    std::vector<tensorrt_lightnet::BBoxInfo> bbox_ = lightnet->getSubnetBbox();

    bbox_c_array.clear();
    for (const auto& bbox : bbox_) {
        BBoxInfoC bbox_c;
        bbox_c.box = {bbox.box.x1, bbox.box.y1, bbox.box.x2, bbox.box.y2};
        bbox_c.label = bbox.label;
        bbox_c.classId = bbox.classId;
        bbox_c.prob = bbox.prob;
        bbox_c.isHierarchical = bbox.isHierarchical;
        bbox_c.subClassId = bbox.subClassId;
        bbox_c.sin = bbox.sin;
        bbox_c.cos = bbox.cos;

        bbox_c.num_keypoints = bbox.keypoint.size();
        static std::vector<KeypointInfoC> keypoints_c;
        keypoints_c.clear();
        for (const auto& kp : bbox.keypoint) {
            keypoints_c.push_back({kp.id_prob, kp.isOccluded, kp.attr_prob,
                kp.lx0, kp.ly0, kp.lx1, kp.ly1, kp.rx0, kp.ry0, kp.rx1, kp.ry1, kp.bot, kp.left});
        }
        bbox_c.keypoints = keypoints_c.data();
        bbox_c_array.push_back(bbox_c);
    }
    *size = bbox_c_array.size();
    return bbox_c_array.data();
}

/**
 * @brief Apply a blur effect to image regions based on bounding boxes detected by the subnet.
 * 
 * @param instance Pointer to the main TrtLightnet instance.
 * @param sub_instance Pointer to the subnet TrtLightnet instance.
 * @param img_data Image data in BGR format.
 * @param width Image width.
 * @param height Image height.
 */
void blur_image(void* instance, void* sub_instance, unsigned char* img_data, int width, int height) {
    cv::Mat image(height, width, CV_8UC3, img_data);
    auto lightnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(instance);
    auto subnet = *static_cast<std::shared_ptr<tensorrt_lightnet::TrtLightnet>*>(sub_instance);    
    cv::Mat cropped;
    int kernel = 15;
    std::vector<tensorrt_lightnet::BBoxInfo> bbox = lightnet->getBbox();

    auto names = lightnet->getNames();
    auto subnet_names = subnet->getNames();

    int num = static_cast<int>(bbox.size());
    for (int i = 0; i < num; i++) {
        auto b = bbox[i];
        bool flg = false;
        for (auto &t : subnet_names) {
            if (t == names[b.classId]) {
                flg = true;
                break;
            }
        }
        if (!flg) continue;

        int subnet_id = -1;
        for (int j = 0; j < static_cast<int>(subnet_names.size()); j++) {
            if (names[b.classId] == subnet_names[j]) {
                subnet_id = j;
                break;
            }
        }
        if (subnet_id != -1) {
            b.label = subnet_id;
            b.classId = subnet_id;    
            auto bb = {b};
            lightnet->appendSubnetBbox(bb);
        }    
    }
    lightnet->doNonMaximumSuppressionForSubnetBbox();

    std::vector<tensorrt_lightnet::BBoxInfo> subnet_bbox = lightnet->getSubnetBbox();

    for (auto &b : subnet_bbox) {
        if ((b.box.x2 - b.box.x1) > kernel && (b.box.y2 - b.box.y1) > kernel / 2) {
            int width = b.box.x2 - b.box.x1;
            int height = b.box.y2 - b.box.y1;
            int w_offset = width * 0.0;
            int h_offset = height * 0.0;      
            cv::Rect roi(b.box.x1 + w_offset, b.box.y1 + h_offset, width - w_offset * 2, height - h_offset * 2);
            cropped = image(roi);

            if (width > 320 && height > 320) {
                cv::blur(cropped, cropped, cv::Size(kernel * 16, kernel * 16));
            } else if (width > 160 && height > 160) {
                cv::blur(cropped, cropped, cv::Size(kernel * 8, kernel * 8));
            } else {
                cv::blur(cropped, cropped, cv::Size(kernel, kernel));
            }
        }
    }
}  
}


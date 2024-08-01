#include <memory>
#include <string>
#include <gflags/gflags.h>
#include <NvInfer.h>
#include <map>
#include <tensorrt_lightnet/tensorrt_lightnet.hpp>

typedef struct _window_info{
  unsigned int x;
  unsigned int y;
  unsigned int w;
  unsigned int h;
} Window_info;

typedef struct _cropping_info{
  int x;
  int y;
  int w;
  int h;
} Cropping_info;

/**
 * Retrieves the path to the ONNX model file.
 *
 * @return The file path as a string.
 */
std::string get_onnx_path(void);

/**
 * Retrieves the directory path where resources or outputs are stored.
 *
 * @return The directory path as a string.
 */
std::string get_directory_path(void);

/**
 * Retrieves the video file path for processing.
 *
 * @return The path to the video file as a string.
 */
std::string get_video_path(void);

/**
 * Retrieves the camera device ID for capturing video.
 *
 * @return The camera ID as an integer.
 */
int get_camera_id(void);

/**
 * Retrieves the precision setting for the network computation.
 *
 * @return The precision as a string (e.g., "FP16", "FP32").
 */
std::string get_precision(void);

/**
 * Checks if the display of output windows is disabled.
 *
 * @return True if display is disabled, false otherwise.
 */
bool is_dont_show(void);

/**
 * Retrieves the path to the directory containing calibration images.
 *
 * @return The file path as a string.
 */
std::string get_calibration_images(void);

/**
 * Checks if profiling is enabled.
 *
 * @return True if profiling is enabled, false otherwise.
 */
bool get_prof_flg(void);

/**
 * Retrieves the batch size for processing.
 *
 * @return The batch size as an integer.
 */
int get_batch_size(void);

/**
 * Retrieves the width of the input images.
 *
 * @return The width as an integer.
 */
int get_width(void);

/**
 * Retrieves the height of the input images.
 *
 * @return The height as an integer.
 */
int get_height(void);

/**
 * Retrieves the number of classes the network can detect.
 *
 * @return The number of classes as an integer.
 */
int get_classes(void);

/**
 * Retrieves the DLA core ID (Deep Learning Accelerator) for NVIDIA platforms.
 *
 * @return The DLA ID as an integer.
 */
int get_dla_id(void);

/**
 * Retrieves the color map for class visualization.
 *
 * @return A 2D vector with RGB values for each class.
 */
std::vector<std::vector<int>> get_colormap(void);

/**
 * Retrieves the names of the classes.
 *
 * @return A vector containing the names of the classes.
 */
std::vector<std::string> get_names(void);

/**
 * Retrieves the threshold for scoring detections.
 *
 * @return The threshold value as a double.
 */
double get_score_thresh(void);

/**
 * Checks if CUDA acceleration is enabled.
 *
 * @return True if CUDA is enabled, false otherwise.
 */
bool get_cuda_flg(void);

/**
 * Checks if sparse computations are enabled.
 *
 * @return True if sparse computations are enabled, false otherwise.
 */
bool get_sparse_flg(void);

/**
 * Checks if the first layer is flagged for special handling.
 *
 * @return True if the first layer has a special flag, false otherwise.
 */
bool get_first_flg(void);

/**
 * Checks if the last layer is flagged for special handling.
 *
 * @return True if the last layer has a special flag, false otherwise.
 */
bool get_last_flg(void);

/**
 * Retrieves the path where debugging dumps should be stored.
 *
 * @return The path as a string.
 */
std::string get_dump_path(void);

/**
 * Retrieves the calibration type for the network.
 *
 * @return The calibration type as a string.
 */
std::string get_calib_type(void);

/**
 * Retrieves the maximum value to which outputs are clipped.
 *
 * @return The clipping value as a double.
 */
double get_clip_value(void);

/**
 * Checks if detections should be saved.
 *
 * @return True if detections should be saved, false otherwise.
 */
bool getSaveDetections(void);

/**
 * Retrieves the path where detections should be saved.
 *
 * @return The path as a string.
 */
std::string getSaveDetectionsPath(void);

/**
 * Retrieves the window information for display or processing purposes.
 *
 * @return A Window_info struct containing the coordinates and size.
 */
Window_info get_window_info(void);

/**
 * Retrieves the output path for processed data or results.
 *
 * @return The path as a string.
 */
std::string get_output_path(void);

/**
 * Retrieves the color map for segmenting the outputs.
 *
 * @return A vector of Colormap structures for segmentation.
 */
std::vector<tensorrt_lightnet::Colormap> get_seg_colormap(void);

/**
 * Retrieves the number of anchors used in the network.
 *
 * @return The number of anchors as an integer.
 */
int get_num_anchors(void);

/**
 * Retrieves the dimensions of the anchors used in the network.
 *
 * @return A vector of integers representing anchor dimensions.
 */
std::vector<int> get_anchors(void);

// For subnet
/**
 * Retrieves the ONNX path for the subnet model.
 *
 * @return The ONNX model path as a string.
 */
std::string get_subnet_onnx_path(void);

/**
 * Retrieves the names associated with the subnet.
 *
 * @return A vector of strings representing the names.
 */
std::vector<std::string> get_subnet_names(void);

/**
 * Retrieves the number of anchors used by the subnet.
 *
 * @return The number of anchors as an integer.
 */
int get_subnet_num_anchors(void);

/**
 * Retrieves the dimensions of the anchors used by the subnet.
 *
 * @return A vector of integers representing anchor dimensions.
 */
std::vector<int> get_subnet_anchors(void);

/**
 * Retrieves the number of classes detected by the subnet.
 *
 * @return The number of classes as an integer.
 */
int get_subnet_classes(void);

/**
 * Retrieves threshold by the subnet.
 *
 * @return value of threshold.
 */
double
get_subnet_score_thresh(void);


/**
 * Retrieves the color map for the subnet visualization.
 *
 * @return A 2D vector with RGB values for each class.
 */
std::vector<std::vector<int>> get_subnet_colormap(void);

/**
 * Retrieves the names of the target classes for specialized processing.
 *
 * @return A vector of strings representing the target names.
 */
std::vector<std::string> get_target_names(void);

/**
 * Retrieves the names of the classes for which blurring effects should be applied.
 *
 * @return A vector of strings representing the names for blurring.
 */
std::vector<std::string> get_bluron_names(void);

/**
 * Retrieves a list of debug tensor names from a global flag.
 * 
 * This function parses the `FLAGS_debug_tensors` global variable, which is expected to contain a comma-separated list of tensor names. If the variable is set to "not-specified", no tensors are retrieved. This function splits the string by commas, trims each resulting substring, and adds it to a vector of strings.
 *
 * @return std::vector<std::string> - A vector containing the names of the debug tensors. If the `FLAGS_debug_tensors` is "not-specified" or empty, an empty vector is returned.
 */
std::vector<std::string>
get_debug_tensors(void);

bool
get_save_debug_tensors(void);

int
get_workers(void);

bool
get_calc_entropy_flg(void);

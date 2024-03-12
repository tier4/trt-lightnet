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

std::string
get_onnx_path(void);

std::string
get_directory_path(void);

std::string
get_video_path(void);

int
get_camera_id(void);
  
std::string
get_precision(void);

bool
is_dont_show(void);

std::string
get_calibration_images();

bool
get_prof_flg(void);
int
get_batch_size(void);
int
get_width(void);

int
get_height(void);

int
get_classes(void);

int
get_dla_id(void);

std::vector<std::vector<int>>
get_colormap(void);

std::vector<std::string>
get_names(void);

double
get_score_thresh(void);

bool
get_cuda_flg(void);

bool
get_sparse_flg(void);


bool
get_first_flg(void);

bool
get_last_flg(void);


std::string
get_dump_path(void);

std::string
get_calib_type(void);

double
get_clip_value(void);


bool getSaveDetections();
std::string getSaveDetectionsPath();


Window_info
get_window_info(void);


std::string
get_output_path(void);

std::vector<tensorrt_lightnet::Colormap>
get_seg_colormap(void);

int
get_num_anchors(void);

std::vector<int>
get_anchors(void);

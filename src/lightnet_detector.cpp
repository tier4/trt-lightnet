#include "lightnet_detector.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <filesystem>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"

namespace trt_lightnet {
TrtLightnetNode::TrtLightnetNode(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("trt_lightnet", node_options) {
  const auto flagfile = declare_parameter("flagfile", "");
  const auto config_path = declare_parameter("config_path", "");

  if (!flagfile.empty()) {
    std::string temp_flagfile = create_temp_flagfile(flagfile);

    // Prepare dummy arg
    std::vector<char*> argv_vec;

    // argv[0]
    std::string prog_name = "trt-lightnet";
    argv_vec.push_back(const_cast<char*>(prog_name.c_str()));

    // --flagfile option
    std::string flagfile_opt = "--flagfile=" + temp_flagfile;
    argv_vec.push_back(const_cast<char*>(flagfile_opt.c_str()));

    int argc = static_cast<int>(argv_vec.size());
    char** argv = argv_vec.data();

    gflags::ParseCommandLineFlags(&argc, &argv, true);
  }

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

  path_config_ = {
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
  visualization_config_ = {
    .dont_show = is_dont_show(),
    .colormap = get_colormap(),
    .names = get_names(),
    .argmax2bgr = getArgmaxToBgr(get_seg_colormap()),
    .seg_colormap = get_seg_colormap(),
    .road_ids = get_road_ids()
  };

  std::string save_path = path_config_.save_path;
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

  trt_lightnet_ =
      std::make_shared<tensorrt_lightnet::TrtLightnet>(model_config, inference_config, build_config);
  
  // Subnet configuration
  std::vector<std::string> bluron = get_bluron_names();
  int numWorks = get_workers();  // omp_get_max_threads();

  if (get_subnet_onnx_path() != "not-specified") {
    ModelConfig subnet_model_config = {
      .model_path = get_subnet_onnx_path(),
      .num_class = get_subnet_classes(),
      .score_threshold = static_cast<float>(get_subnet_score_thresh()),
      .anchors = get_subnet_anchors(),
      .num_anchors = get_subnet_num_anchors(),
      .nms_threshold = 0.25f // Assuming this value is fixed or retrieved similarly.
    };
    subnet_visualization_config_ = {
      .dont_show = is_dont_show(),
      .colormap = get_subnet_colormap(),
      .names = get_subnet_names(),
      .argmax2bgr = getArgmaxToBgr(seg_colormap)
    };
    target_ = get_target_names();
    for (int w = 0; w < numWorks; w++) {
      if (build_config.dla_core_id >= 2) {
        // use multiple dlas [DLA0 and DLA1]
        build_config.dla_core_id = (int)w / 2;
      }
      subnet_trt_lightnets_.push_back(
          std::make_shared<tensorrt_lightnet::TrtLightnet>(subnet_model_config, inference_config, build_config));
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
    keypoint_target_ = get_keypoint_names();
    for (int w = 0; w < numWorks; w++) {
      if (build_config.dla_core_id >= 2) {
        // use multiple dlas [DLA0 and DLA1]
        build_config.dla_core_id = (int)w / 2;
      }
      keypoint_trt_lightnets_.push_back(
          std::make_shared<tensorrt_lightnet::TrtLightnet>(keypoint_model_config, inference_config, build_config));
    }
  }

  if (get_fswp_onnx_path() != "not-specified") {
    int bs = 8;
    std::filesystem::path fswp_path(get_fswp_onnx_path());
    fswp_model_ = std::make_shared<fswp::FaceSwapper>(fswp_path, build_config, bs, "fp16");
    target_ = get_target_names();
  }

  compressed_image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      "~/input/compressed_image", rclcpp::SensorDataQoS(),
      std::bind(&TrtLightnetNode::onCompressedImage, this, std::placeholders::_1));
  raw_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/output/raw_image", rclcpp::SensorDataQoS());
  mask_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/output/mask_image", rclcpp::SensorDataQoS());
  depth_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/output/depth_image", rclcpp::SensorDataQoS());
  bev_image_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/output/bev_image", rclcpp::SensorDataQoS());
}

void TrtLightnetNode::onCompressedImage(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr input_compressed_image_msg) {
  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

  // Decode image
  try {
    cv_ptr->header = input_compressed_image_msg->header;
    cv_ptr->image = cv::imdecode(cv::Mat(input_compressed_image_msg->data), cv::IMREAD_COLOR);
    cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
  } catch (cv::Exception& e) {
    RCLCPP_ERROR(get_logger(), "%s", e.what());
  }

  inferLightNetPipeline(trt_lightnet_, subnet_trt_lightnets_, keypoint_trt_lightnets_, fswp_model_, cv_ptr->image, visualization_config_, subnet_visualization_config_, target_, keypoint_target_, bluron_, path_config_, "", "", "");

  auto& image = cv_ptr->image;
  std::vector<tensorrt_lightnet::BBoxInfo> bbox = trt_lightnet_->getBbox();
  std::vector<cv::Mat> masks = trt_lightnet_->getMask();
  std::vector<cv::Mat> depthmaps = trt_lightnet_->getDepthmap();
  std::vector<tensorrt_lightnet::KeypointInfo> keypoint = trt_lightnet_->getKeypoints();
  for (const auto& mask : masks) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
    cv::addWeighted(image, 1.0, resized, 0.45, 0.0, image);
  }

  trt_lightnet_->drawBbox(image, bbox, visualization_config_.colormap, visualization_config_.names);
  if (target_.size() > 0 ) {
    if (get_plot_circle_flg()) {
      Calibration calibdata = {
        .u0 = (float)(image.cols/2.0),
        .v0 = (float)(image.rows/2.0),
        .fx = get_fx(),
        .fy = get_fy(),
        .max_distance = get_max_distance(),
      };
      trt_lightnet_->plotCircleIntoBevmap(image.cols, image.rows, calibdata, visualization_config_.names, target_);
    }
  }
  if (keypoint.size() > 0) {
    trt_lightnet_->drawKeypoint(image, keypoint);
  }

  size_t rows = cv_ptr->image.rows;
  size_t cols = cv_ptr->image.cols;
  if ((rows > 0) && (cols > 0)) {
    auto image_ptr = std::make_unique<sensor_msgs::msg::Image>(*cv_ptr->toImageMsg());
    raw_image_pub_->publish(std::move(image_ptr));

    // Publish mask image
    {
      // cv::Mat resized;
      // cv::resize(masks[0], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
      cv_ptr->image = masks[0];
      image_ptr = std::make_unique<sensor_msgs::msg::Image>(*cv_ptr->toImageMsg());
      mask_image_pub_->publish(std::move(image_ptr));
    }

    // Publish depth image
    {
      // cv::Mat resized;
      // cv::resize(depthmaps[0], resized, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_NEAREST);
      cv_ptr->image = depthmaps[0];
      image_ptr = std::make_unique<sensor_msgs::msg::Image>(*cv_ptr->toImageMsg());
      depth_image_pub_->publish(std::move(image_ptr));
    }

    // Publish bevmap
    {
      cv_ptr->image = trt_lightnet_->getBevMap();
      image_ptr = std::make_unique<sensor_msgs::msg::Image>(*cv_ptr->toImageMsg());
      bev_image_pub_->publish(std::move(image_ptr));
    }

  }
}

std::string TrtLightnetNode::create_temp_flagfile(const std::string& original_flagfile) {
  std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("trt_lightnet");

  // オリジナルのファイル名を取得
  std::filesystem::path original_path(original_flagfile);
  std::string original_filename = original_path.filename().string();

  // 一時ファイルのパスを生成
  std::filesystem::path temp_path = std::filesystem::temp_directory_path() /
                                    ("ros2_" + original_filename);
  std::string temp_flagfile = temp_path.string();

  std::ifstream in_file(original_flagfile);
  std::ofstream out_file(temp_flagfile);
  std::string line;

  // "../" で始まるパスを書き換える
  std::regex path_pattern(R"(=\s*\.\./(.*))");

  while (std::getline(in_file, line)) {
    std::smatch matches;
    if (std::regex_search(line, matches, path_pattern)) {
      // "../" を pkg_share_dir に置換
      std::string new_path = "=" + pkg_share_dir + "/" + matches[1].str();
      line = std::regex_replace(line, path_pattern, new_path);
    }
    out_file << line << "\n";
  }

  in_file.close();
  out_file.close();

  return temp_flagfile;
}
}  // namespace trt_lightnet

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(trt_lightnet::TrtLightnetNode)

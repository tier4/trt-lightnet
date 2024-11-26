#include <pcdUtils/pcd2image.hpp>
#include <chrono>
/**
 * Removes specified extensions from the filename in the given file path.
 *
 * @param filepath The file path to process.
 * @param extensions A list of extensions to remove.
 * @return The filename with the specified extensions removed.
 */
std::string removeExtensions(const std::filesystem::path& filepath, const std::vector<std::string>& extensions) {
  std::string filename = filepath.filename().string();
  for (const auto& ext : extensions) {
    if (filename.size() >= ext.size() &&
	filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
      return filename.substr(0, filename.size() - ext.size());
    }
  }
  return filename;
}

namespace pcd2image
{
  Pcd2image::Pcd2image() {}

  Pcd2image::~Pcd2image() {}

  /**
   * Creates a transformation matrix from the camera to the LiDAR sensor using quaternion and translation data.
   *
   * @param cam_calib Camera calibration information containing translation and rotation.
   * @return A 4x4 transformation matrix.
   */
  Eigen::Matrix4f Pcd2image::createCameraToLidarTransform(CalibratedSensorInfo cam_calib) {
    Eigen::Vector3f translation(cam_calib.translation[0], cam_calib.translation[1], cam_calib.translation[2]);
    Eigen::Quaternionf rotation(cam_calib.rotation[0], cam_calib.rotation[1], cam_calib.rotation[2], cam_calib.rotation[3]);

    // Compute inverse rotation and translation
    Eigen::Matrix3f rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix3f R_inv = rotation_matrix.transpose();  // Inverse rotation
    Eigen::Vector3f t_inv = -R_inv * translation;         // Inverse translation

    // Create the transformation matrix
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = R_inv;
    transform.block<3, 1>(0, 3) = t_inv;

    return transform;
  }

  /**
   * Loads a point cloud file and converts it into a vector of 3D points.
   *
   * @param filename The file path to the point cloud file.
   * @param format The format of the point cloud data.
   * @return A vector of 3D points.
   */
  std::vector<Point3D> Pcd2image::loadPointCloud(const std::string& filename, PointCloudFormat format) {
    std::vector<Point3D> points;
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
      std::cerr << "Error: Could not open point cloud file!" << std::endl;
      return points;
    }

    Point3D point;
    while (!input.eof()) {
      switch (format) {
      case PointCloudFormat::XYZ_FLOAT: {
	// x, y, z (float)を読み込む
	input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.z), sizeof(float));
	point.intensity = 0.0f; // 強度がない場合は0に設定
	break;
      }
      case PointCloudFormat::XYZ_DOUBLE: {
	// x, y, z (double)を読み込み、floatに変換して格納
	double x, y, z;
	input.read(reinterpret_cast<char*>(&x), sizeof(double));
	input.read(reinterpret_cast<char*>(&y), sizeof(double));
	input.read(reinterpret_cast<char*>(&z), sizeof(double));
	point.x = static_cast<float>(x);
	point.y = static_cast<float>(y);
	point.z = static_cast<float>(z);
	point.intensity = 0.0f; // 強度がない場合は0に設定
	break;
      }
      case PointCloudFormat::XYZRGB_FLOAT: {
	// x, y, z, r, g, b (float)を読み込み、RGBは無視
	float r, g, b;
	input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.z), sizeof(float));
	input.read(reinterpret_cast<char*>(&r), sizeof(float));
	input.read(reinterpret_cast<char*>(&g), sizeof(float));
	input.read(reinterpret_cast<char*>(&b), sizeof(float));
	point.intensity = 0.0f; // 強度がない場合は0に設定
	break;
      }
      case PointCloudFormat::XYZI_FLOAT: {
	// x, y, z, intensity (float)を読み込む
	input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.z), sizeof(float));      
	input.read(reinterpret_cast<char*>(&point.intensity), sizeof(float));
	input.read(reinterpret_cast<char*>(&point.tmp), sizeof(float));                  
	break;
      }
      default:
	std::cerr << "Error: Unsupported point cloud format!" << std::endl;
	input.close();
	return points;
      }

      if (input) {
	points.push_back(point);
      }
    }

    input.close(); 
    return points;
  }

  /**
   * Creates a camera intrinsic matrix from the given intrinsic parameters.
   *
   * @param intrinsics A 1D array of 9 intrinsic parameters.
   * @return A 3x3 camera intrinsic matrix.
   */
  cv::Mat Pcd2image::createCameraIntrinsics(std::vector<std::vector<double>> &intrinsics) {
    return (cv::Mat_<double>(3, 3) << intrinsics[0][0], intrinsics[0][1], intrinsics[0][2],
	    intrinsics[1][0], intrinsics[1][1], intrinsics[1][2],
	    intrinsics[2][0], intrinsics[2][1], intrinsics[2][2]);
  }

  /**
   * Creates a distortion coefficient matrix.
   *
   * @param distortion A 1D array of 5 distortion coefficients.
   * @return A 1x5 distortion coefficient matrix.
   */
  cv::Mat Pcd2image::createDistortionCoefficients(std::vector<double> &distortion) {
    return (cv::Mat_<double>(1, 5) << distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]);
    //return (cv::Mat_<double>(1, 5) << distortion[0]*(-1), distortion[1]*(-1), distortion[2]*(-1), distortion[3]*(-1), distortion[4]*(-1));    
    //return (cv::Mat_<double>(1, 5) << distortion[4], distortion[3], distortion[2], distortion[1], distortion[0]);    
  }

  /**
   * Projects LiDAR points onto the image plane using camera calibration parameters.
   *
   * @param lidar_points A vector of 3D LiDAR points.
   * @param camera_intrinsics The camera intrinsic matrix.
   * @param camera_to_lidar_transform The transformation matrix from the camera to the LiDAR.
   * @param distortion_coeffs The distortion coefficient matrix.
   * @param cam_calib Camera calibration information.
   * @return A vector of 2D projections on the image plane.
   */
  std::vector<Projection2D> Pcd2image::projectLidarToImage(
							   const std::vector<Point3D>& lidar_points,
							   const cv::Mat& camera_intrinsics,
							   const Eigen::Matrix4f& camera_to_lidar_transform,
							   const cv::Mat& distortion_coeffs,
							   CalibratedSensorInfo cam_calib) {

    std::vector<Projection2D> projection_points;

    for (const auto& point : lidar_points) {
      Eigen::Vector4f lidar_point_homo(point.x, point.y, point.z, 1.0f);
      Eigen::Vector4f camera_point_homo = camera_to_lidar_transform * lidar_point_homo;

      if (camera_point_homo(2) <= 0) continue;

      cv::Point2f distorted_point(
				  (camera_intrinsics.at<double>(0, 0) * camera_point_homo(0) / camera_point_homo(2)) + camera_intrinsics.at<double>(0, 2),
				  (camera_intrinsics.at<double>(1, 1) * camera_point_homo(1) / camera_point_homo(2)) + camera_intrinsics.at<double>(1, 2));

      std::vector<cv::Point2f> distorted_points = {distorted_point};
      std::vector<cv::Point2f> undistorted;
      cv::undistortPoints(distorted_points, undistorted, camera_intrinsics, distortion_coeffs, cv::noArray(), camera_intrinsics);

      int u = static_cast<int>(undistorted[0].x);
      int v = static_cast<int>(undistorted[0].y);

      if (u >= 0 && u < cam_calib.width && v >= 0 && v < cam_calib.height) {
	Projection2D p2d;
	p2d.coord = cv::Point2f(u, v);
	p2d.intensity = point.intensity;
	auto d = camera_point_homo(2);
	auto y = camera_point_homo(1);	
	p2d.distance = d;
	p2d.height = ((y) * (-1) + 2.0) / 8.0;	
	projection_points.push_back(p2d);
      }
    }

    return projection_points;
  }

  /**
   * Generates a range image from LiDAR data based on camera calibration.
   *
   * @param inputName The file path to the input point cloud.
   * @param cam_calib Camera calibration information.
   * @param max_distance The maximum distance for normalization.
   * @return The generated range image.
   */
  cv::Mat Pcd2image::makeRangeImageFromCalibration(std::string inputName, CalibratedSensorInfo cam_calib, float max_distance) {
    cv::Mat camera_intrinsics = createCameraIntrinsics(cam_calib.camera_intrinsic);
    //cv::Mat distortion_coeffs = createDistortionCoefficients(cam_calib.camera_distortion);

    cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);    
    Eigen::Matrix4f camera_to_lidar_transform = createCameraToLidarTransform(cam_calib);

    cv::Mat image = cv::Mat::zeros(cam_calib.height, cam_calib.width, CV_8UC3);
    std::vector<Point3D> lidar_points = loadPointCloud(inputName, PointCloudFormat::XYZI_FLOAT);
    if (lidar_points.empty()) {
      return image;
    }

    std::vector<Projection2D> projected_points = projectLidarToImage(lidar_points, camera_intrinsics, camera_to_lidar_transform, distortion_coeffs, cam_calib);
    
    for (const auto& point : projected_points) {
      unsigned char h = static_cast<unsigned char>(std::min(std::max(point.height * 255.0f, 0.0f), 255.0f));
      unsigned char d = static_cast<unsigned char>(std::min(std::max(point.distance / max_distance * 255.0f, 0.0f), 255.0f));
      cv::circle(image, point.coord, 2, cv::Scalar(h, 0, d), -1);
    }

    return image;
  }
} // namespace pcd2image

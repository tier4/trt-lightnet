#ifndef PCD2IMAGE_HPP
#define PCD2IMAGE_HPP

#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include "CalibratedSensorParser.h"
/**
 * Removes specified extensions from the given file path.
 *
 * @param filepath The input file path.
 * @param extensions A list of extensions to be removed.
 * @return The filename without the specified extensions.
 */
extern std::string removeExtensions(const std::filesystem::path& filepath, const std::vector<std::string>& extensions);

namespace pcd2image {

  /**
   * Structure to store a 3D point in the LiDAR coordinate system.
   */
  struct Point3D {
    float x;          // X coordinate
    float y;          // Y coordinate
    float z;          // Z coordinate
    float intensity;  // Intensity value
    float tmp;        // Temporary data (optional)
  };

  /**
   * Structure to represent a 2D projection of a LiDAR point onto an image plane.
   */
  typedef struct _Projection2D {
    cv::Point2f coord; // 2D coordinates in the image
    float distance;    // Distance of the point from the LiDAR
    float height;      // Normalized height
    float intensity;   // Intensity value
  } Projection2D;

  /**
   * Structure for storing camera calibration information.
   */
  typedef struct _CameraCalibrationInfo {
    int id;                  // Camera ID
    std::string name;        // Camera name
    float intrinsics[9];     // 3x3 intrinsic matrix as a 1D array
    float distortion[5];     // Distortion coefficients
    float translation[3];    // Translation vector (T_camera_to_lidar)
    float rotation[4];       // Quaternion rotation (R_camera_to_lidar)
    int width;               // Image width
    int height;              // Image height
  } CameraCalibrationInfo;

  /**
   * Enum to define point cloud data formats.
   */
  enum class PointCloudFormat {
    XYZ_FLOAT,    // x, y, z as float
    XYZ_DOUBLE,   // x, y, z as double
    XYZRGB_FLOAT, // x, y, z, r, g, b as float
    XYZI_FLOAT    // x, y, z, intensity as float
  };

  /**
   * Class for processing LiDAR point clouds and projecting them onto images.
   */
  class Pcd2image {
  public:
    /**
     * Default constructor.
     */
    Pcd2image();

    /**
     * Destructor.
     */
    ~Pcd2image();

    /**
     * Creates a 4x4 transformation matrix from the camera to the LiDAR coordinate system.
     *
     * @param cam_calib Camera calibration information.
     * @return The transformation matrix (4x4).
     */
    Eigen::Matrix4f createCameraToLidarTransform(CalibratedSensorInfo cam_calib);

    /**
     * Loads a point cloud from a file.
     *
     * @param filename The path to the point cloud file.
     * @param format The format of the point cloud data.
     * @return A vector of 3D points.
     */
    std::vector<Point3D> loadPointCloud(const std::string& filename, PointCloudFormat format);

    /**
     * Creates a camera intrinsic matrix from a 1D array of parameters.
     *
     * @param intrinsics A 1D array of 9 intrinsic parameters.
     * @return A 3x3 camera intrinsic matrix.
     */
    cv::Mat createCameraIntrinsics(std::vector<std::vector<double>> &intrinsics);

    /**
     * Creates a distortion coefficient matrix.
     *
     * @param distortion A 1D array of 5 distortion coefficients.
     * @return A 1x5 distortion coefficient matrix.
     */
    cv::Mat createDistortionCoefficients(std::vector<double> &camera_distortion);

    /**
     * Projects LiDAR points onto the image plane.
     *
     * @param lidar_points A vector of 3D LiDAR points.
     * @param camera_intrinsics The camera intrinsic matrix.
     * @param camera_to_lidar_transform The transformation matrix from the camera to the LiDAR.
     * @param distortion_coeffs The distortion coefficient matrix.
     * @param cam_calib Camera calibration information.
     * @return A vector of 2D projections on the image plane.
     */
    std::vector<Projection2D> projectLidarToImage(
						  const std::vector<Point3D>& lidar_points,
						  const cv::Mat& camera_intrinsics,
						  const Eigen::Matrix4f& camera_to_lidar_transform,
						  const cv::Mat& distortion_coeffs,
						  CalibratedSensorInfo cam_calib);

    /**
     * Generates a range image from LiDAR data using camera calibration.
     *
     * @param inputName The file path to the input point cloud.
     * @param cam_calib Camera calibration information.
     * @param max_distance The maximum distance for normalization.
     * @return The generated range image as a cv::Mat.
     */
    cv::Mat makeRangeImageFromCalibration(
					  std::string inputName,
					  CalibratedSensorInfo cam_calib,
					  float max_distance);
  };

} // namespace pcd2image

#endif // PCD2IMAGE_HPP

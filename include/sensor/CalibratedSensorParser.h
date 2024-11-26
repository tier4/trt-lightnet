#ifndef CALIBRATEDSENSORPARSER_H
#define CALIBRATEDSENSORPARSER_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <sensor/SensorParser.h>

// Alias for the JSON library
using json = nlohmann::json;

/**
 * Structure to hold calibrated sensor information.
 */
struct CalibratedSensorInfo {
  std::string token;                           ///< Unique token for the calibrated sensor.
  std::string sensor_token;                    ///< Token linking to the associated sensor.
  std::vector<double> translation;            ///< Translation vector [x, y, z].
  std::vector<double> rotation;               ///< Rotation quaternion [w, x, y, z].
  std::vector<std::vector<double>> camera_intrinsic; ///< Camera intrinsic matrix.
  std::vector<double> camera_distortion;      ///< Camera distortion coefficients.
  std::string name;                           ///< Channel name derived from the associated sensor token.
  std::string modality;
  int width;
  int height;
};

/**
 * Class to parse calibrated sensor data from a JSON file.
 */
class CalibratedSensorParser {
 public:
  /**
   * Parses a JSON file containing calibrated sensor information and populates a vector of CalibratedSensorInfo structures.
   *
   * @param fileName The name of the JSON file to parse.
   * @param entries A vector to store the parsed CalibratedSensorInfo structures.
   * @throws std::runtime_error If the file cannot be opened or parsed.
   */
  static void parse(const std::string& fileName, std::vector<CalibratedSensorInfo>& entries);
};

#endif // CALIBRATEDSENSORPARSER_H

#ifndef SENSORPARSER_H
#define SENSORPARSER_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

// Alias for the JSON library
using json = nlohmann::json;

/**
 * Structure to represent a sensor.
 */
struct Sensor {
  std::string token;     ///< Unique identifier for the sensor.
  std::string channel;   ///< The channel or name associated with the sensor.
  std::string modality;  ///< Modality of the sensor (e.g., "lidar", "camera").
};

struct Resolution {
  int width;
  int height;
};

using ResolutionMap = std::map<std::string, Resolution>;

/**
 * Class to parse sensor data from a JSON file.
 */
class SensorParser {
 public:
  /**
   * Parses a JSON file containing sensor information and populates a vector of Sensor structures.
   */
  static void parse(const std::string& fileName, std::vector<Sensor>& sensors);

  /**
   * Retrieves the channel name associated with a given sensor token.
   */
  static std::string getSensorNameFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken);

  static std::string getSensorModalityFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken);
  static void parseCameraResolutions(const std::string& fileName, ResolutionMap& outResolutions);
};

#endif // SENSORPARSER_H


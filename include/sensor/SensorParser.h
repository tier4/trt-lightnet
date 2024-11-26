#ifndef SENSORPARSER_H
#define SENSORPARSER_H

#include <string>
#include <vector>
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

/**
 * Class to parse sensor data from a JSON file.
 */
class SensorParser {
 public:
  /**
   * Parses a JSON file containing sensor information and populates a vector of Sensor structures.
   *
   * @param fileName The name of the JSON file to parse.
   * @param sensors A vector to store the parsed Sensor structures.
   * @throws std::runtime_error If the file cannot be opened or parsed.
   */
  static void parse(const std::string& fileName, std::vector<Sensor>& sensors);

  /**
   * Retrieves the channel name associated with a given sensor token.
   *
   * @param sensors A vector of Sensor structures to search.
   * @param sensorToken The token of the sensor to find.
   * @return The channel name corresponding to the sensor token, or an empty string if not found.
   */
  static std::string getSensorNameFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken);

  static std::string getSensorModalityFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken);
};

#endif // SENSORPARSER_H


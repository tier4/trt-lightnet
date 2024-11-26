#include <sensor/SensorParser.h>
#include <fstream>
#include <stdexcept>

/**
 * Parses a JSON file containing sensor information and populates a vector of Sensor structures.
 *
 * @param fileName The name of the JSON file to parse.
 * @param sensors A vector to store the parsed Sensor structures.
 * @throws std::runtime_error If the file cannot be opened.
 */
void SensorParser::parse(const std::string& fileName, std::vector<Sensor>& sensors) {
  // Open the JSON file for reading
  std::ifstream inputFile(fileName);
  if (!inputFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + fileName);
  }

  // Parse the JSON data
  json jsonData;
  inputFile >> jsonData;

  // Iterate through the JSON array and populate the Sensor vector
  for (const auto& item : jsonData) {
    Sensor sensor;
    // Extract data from the JSON object
    sensor.token = item["token"];
    sensor.channel = item["channel"];
    sensor.modality = item["modality"];
    sensors.push_back(sensor);
  }
}

/**
 * Retrieves the channel name associated with a given sensor token.
 *
 * @param sensors A vector of Sensor structures to search.
 * @param sensorToken The token of the sensor to search for.
 * @return The channel name corresponding to the sensor token, or an empty string if not found.
 */
std::string SensorParser::getSensorNameFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken) {
  // Iterate through the vector to find the sensor with the given token
  for (const auto& sensor : sensors) {
    if (sensor.token == sensorToken) {
      return sensor.channel; // Return the channel name if found
    }
  }
  return ""; // Return an empty string if the token is not found
}



std::string SensorParser::getSensorModalityFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken) {
  // Iterate through the vector to find the sensor with the given token
  for (const auto& sensor : sensors) {
    if (sensor.token == sensorToken) {
      return sensor.modality; // Return the channel name if found
    }
  }
  return ""; // Return an empty string if the token is not found
}

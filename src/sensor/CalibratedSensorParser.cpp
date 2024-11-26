#include <sensor/CalibratedSensorParser.h>
#include <fstream>
#include <stdexcept>

/**
 * Parses a JSON file containing calibrated sensor information and populates a vector of CalibratedSensorInfo structures.
 *
 * @param fileName The name of the JSON file to parse.
 * @param entries A vector to store the parsed CalibratedSensorInfo structures.
 * @throws std::runtime_error If the file cannot be opened.
 */
void CalibratedSensorParser::parse(const std::string& fileName, std::vector<CalibratedSensorInfo>& entries) {
  // Open the JSON file for reading
  std::ifstream inputFile(fileName);
  if (!inputFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + fileName);
  }

  // Parse the JSON data
  json jsonData;
  inputFile >> jsonData;

  // Iterate through the JSON array and populate the CalibratedSensorInfo vector
  for (const auto& item : jsonData) {
    CalibratedSensorInfo entry;
    // Extract required data from the JSON object
    entry.token = item["token"];
    entry.sensor_token = item["sensor_token"];
    entry.translation = item["translation"].get<std::vector<double>>();
    entry.rotation = item["rotation"].get<std::vector<double>>();
    entry.camera_intrinsic = item["camera_intrinsic"].get<std::vector<std::vector<double>>>();
    entry.camera_distortion = item["camera_distortion"].get<std::vector<double>>();

    // Add the entry to the vector
    entries.push_back(entry);
  }
}

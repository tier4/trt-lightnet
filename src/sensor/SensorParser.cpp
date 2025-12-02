#include <sensor/SensorParser.h>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <iostream>
#include <sstream>

// Assuming the using declaration for nlohmann::json is defined elsewhere or in the header.
// To use nlohmann/json, #include <nlohmann/json.hpp> is required in SensorParser.h or this file.
using json = nlohmann::json;

// =========================================================================
// SensorParser (sensor.json / calibrated_sensor.json related)
// =========================================================================

/**
 * @brief A set of common file formats used for camera data.
 * This is used to infer the modality if the 'sensor_modality' field is missing.
 */
const std::set<std::string> CAMERA_FORMATS = {"jpg", "jpeg", "png", "webp"};

/**
 * @brief Parses a JSON file containing sensor information and populates a vector of Sensor structures.
 *
 * This function reads a sensor configuration file (typically `sensor.json`),
 * extracts the token, channel name, and modality for each sensor, and stores
 * them in the provided vector.
 *
 * @param fileName The path to the sensor configuration JSON file.
 * @param sensors Reference to a vector where the parsed Sensor structures will be stored.
 * @throws std::runtime_error If the file cannot be opened or if JSON parsing fails.
 */
void SensorParser::parse(const std::string& fileName, std::vector<Sensor>& sensors) {
    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    json jsonData;
    try {
        inputFile >> jsonData;
    } catch (const json::exception& e) {
        throw std::runtime_error("Error parsing " + fileName + ": " + e.what());
    }

    // Iterate over the JSON array and extract sensor properties
    for (const auto& item : jsonData) {
        Sensor sensor;
        sensor.token = item.value("token", "");
        sensor.channel = item.value("channel", "");
        sensor.modality = item.value("modality", "");
        sensors.push_back(sensor);
    }
}

/**
 * @brief Retrieves the channel name (e.g., "CAM_FRONT") associated with a given sensor token.
 *
 * @param sensors The vector of parsed Sensor structures to search within.
 * @param sensorToken The unique token identifying the sensor.
 * @return std::string The channel name of the sensor, or an empty string if the token is not found.
 */
std::string SensorParser::getSensorNameFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken) {
    for (const auto& sensor : sensors) {
        if (sensor.token == sensorToken) {
            return sensor.channel;
        }
    }
    return "";
}

/**
 * @brief Retrieves the modality (e.g., "camera", "lidar") associated with a given sensor token.
 *
 * @param sensors The vector of parsed Sensor structures to search within.
 * @param sensorToken The unique token identifying the sensor.
 * @return std::string The modality of the sensor, or an empty string if the token is not found.
 */
std::string SensorParser::getSensorModalityFromToken(const std::vector<Sensor>& sensors, const std::string& sensorToken) {
    for (const auto& sensor : sensors) {
        if (sensor.token == sensorToken) {
            return sensor.modality;
        }
    }
    return "";
}

// =========================================================================
// SampleDataParser Functionality (sample_data.json related)
// =========================================================================

/**
 * @brief Internal structure used to temporarily hold and compare resolution values.
 */
struct TempResolution {
    int width = 0;
    int height = 0;

    /**
     * @brief Checks if two TempResolution objects are unequal.
     * @param other The other resolution to compare against.
     * @return true if the width or height are different, false otherwise.
     */
    bool operator!=(const TempResolution& other) const {
        return width != other.width || height != other.height;
    }
};

/**
 * @brief Parses the `sample_data.json` file to extract consistent resolutions for each camera channel.
 *
 * This function iterates through all sample data entries, identifies camera entries,
 * extracts the channel name from the "filename" tag, and checks for resolution
 * consistency across all samples for that channel. Channels with inconsistent
 * resolutions are excluded.
 *
 * @param fileName The path to the `sample_data.json` file.
 * @param outResolutions A map where the resolved, consistent camera resolutions (channel name -> Resolution) will be stored.
 */
void SensorParser::parseCameraResolutions(const std::string& fileName, ResolutionMap& outResolutions) {
    outResolutions.clear();
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
      std::cerr << "Error: Could not open sample_data.json at " << fileName << std::endl;
      return;
    }

    json j;
    try {
      ifs >> j;
    } catch (const json::exception& e) {
      std::cerr << "Error parsing sample_data.json: " << e.what() << std::endl;
      return;
    }

    // Maps to track unique resolutions and channels with resolution mismatches
    std::map<std::string, TempResolution> uniqueResolutions;
    std::set<std::string> invalidChannels;

    if (!j.is_array()) {
      std::cerr << "Error: sample_data.json is not a JSON array." << std::endl;
      return;
    }

    // Iterate through the JSON array
    for (const auto& item : j) {
      std::string modality = item.value("sensor_modality", "");
      std::string fileformat = item.value("fileformat", "");
      std::string filename = item.value("filename", "");

      // --- Camera determination logic ---
      // An entry is considered a camera if its modality is "camera" OR if modality is empty but fileformat matches a known camera format.
      bool isCamera = (modality == "camera") ||
        (modality.empty() && CAMERA_FORMATS.count(fileformat) > 0);

      // Process only valid camera entries
      if (isCamera) {      

        // --- Logic to extract channel name from filename ---
        std::string channel = "";
        if (!filename.empty()) {
          std::stringstream ss(filename);
          std::string segment;
          std::vector<std::string> segments;
          // Split the filename path by '/'
          while (std::getline(ss, segment, '/')) {
            if (!segment.empty()) {
              segments.push_back(segment);
            }
          }

          // Assuming the format is "data/CAM_NAME/00000.ext", the second segment is the camera name.
          if (segments.size() >= 2) {
            channel = segments[1]; // The second segment is the camera name
          }
        }

        if (channel.empty()) {
          // Skip if the channel name could not be extracted
          continue;
        }

        int width = item.value("width", 0);
        int height = item.value("height", 0);
        TempResolution currentRes = {width, height};

        if (uniqueResolutions.find(channel) == uniqueResolutions.end()) {
          // First observation for that channel: store the resolution
          uniqueResolutions[channel] = currentRes;
        } else {
          // Subsequent observation: check for mismatch
          if (uniqueResolutions[channel] != currentRes) {
            // Resolution mismatch detected: mark the channel as invalid
            invalidChannels.insert(channel);
          }
        }
      }
    }

    // Transfer valid channels to the final output map
    for (const auto& pair : uniqueResolutions) {
      const std::string& channel = pair.first;
      if (invalidChannels.find(channel) == invalidChannels.end()) {
        // Only include channels without any detected resolution mismatch
        outResolutions[channel] = {pair.second.width, pair.second.height};
        std::cout << "Resolved resolution for " << channel << ": "
          << pair.second.width << "x" << pair.second.height << std::endl;
      } else {
        // Log a warning for channels that were dropped
        std::cerr << "Warning: Dropping resolution for camera " << channel
          << " due to internal resolution mismatch in sample_data.json." << std::endl;
      }
    }
}

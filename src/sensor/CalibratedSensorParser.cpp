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

/**
 * @brief Retrieves the calibrated sensor information for a specified camera from the given calibration data path.
 *
 * @param caliibrationInfoPath Path to the directory containing calibration data files.
 * @param camera_name Name of the camera whose calibrated information is to be retrieved.
 * @return CalibratedSensorInfo The calibrated sensor information for the specified camera.
 */
CalibratedSensorInfo getTargetCalibratedInfo(std::string caliibrationInfoPath, std::string camera_name) {
  std::string sensorFileName;
  std::string calibratedSensorFileName;
  std::string sampleDataFileName;
  std::vector<Sensor> sensors;
  std::vector<CalibratedSensorInfo> calibratedSensors;
  CalibratedSensorInfo targetCalibratedInfo;
  ResolutionMap cameraResolutions;
  // Construct file paths for sensor.json and calibrated_sensor.json
  sensorFileName = (std::filesystem::path(caliibrationInfoPath) / "sensor.json").string();
  calibratedSensorFileName = (std::filesystem::path(caliibrationInfoPath) / "calibrated_sensor.json").string();
  sampleDataFileName = (std::filesystem::path(caliibrationInfoPath) / "sample_data.json").string(); 
  try {
    // Parse the sensor data from sensor.json
    SensorParser::parse(sensorFileName, sensors);

    // Parse the calibrated sensor data from calibrated_sensor.json
    CalibratedSensorParser::parse(calibratedSensorFileName, calibratedSensors);

    SensorParser::parseCameraResolutions(sampleDataFileName, cameraResolutions);    
    
    // Iterate through each calibrated sensor and match it with its corresponding sensor information
    for (auto& calibratedSensor : calibratedSensors) {
      // Retrieve and set the sensor name and modality based on the sensor token
      calibratedSensor.name = SensorParser::getSensorNameFromToken(sensors, calibratedSensor.sensor_token);
      calibratedSensor.modality = SensorParser::getSensorModalityFromToken(sensors, calibratedSensor.sensor_token);

      if (calibratedSensor.modality == "camera") {
	auto it = cameraResolutions.find(calibratedSensor.name);
	if (it != cameraResolutions.end()) {
	  calibratedSensor.width = it->second.width;
	  calibratedSensor.height = it->second.height;
	} else {
	  std::cerr << "Warning: Skipping camera " << calibratedSensor.name 
		    << " because resolution data is missing or inconsistent in sample_data.json." << std::endl;
	}
      }      
      
      // If the current sensor matches the target camera name, store its information
      if (calibratedSensor.name == camera_name) {
	targetCalibratedInfo = calibratedSensor;
      }
    }
  } catch (const std::exception& e) {
    // Handle any errors that occur during parsing or processing
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // Return the calibrated sensor information for the specified camera
  return targetCalibratedInfo;
}

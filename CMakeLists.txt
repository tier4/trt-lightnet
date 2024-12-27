cmake_minimum_required(VERSION 3.10) # Recommended version
project(trt_lightnet VERSION 1.0 LANGUAGES CXX CUDA)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Compiler settings
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")
# Debug settings (optional, toggle as needed)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

# CUDA settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")


# Include directories and library paths
include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64)

# Find external libraries
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Define compile-time constants
add_compile_options(-D LIGHTNET_STANDALONE)

# Find ROS 2 libraries
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Gather source files
file(GLOB_RECURSE sources
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
  src/tensorrt_lightnet/tensorrt_lightnet.cpp
  src/sensor/CalibratedSensorParser.cpp
  src/sensor/SensorParser.cpp
  src/pcdUtils/pcd2image.cpp
  src/fswp/fswp.cpp
  src/preprocess.cu  
  )

# Create a shared library
add_library(detector SHARED ${sources})
target_link_libraries(detector
  nvinfer nvinfer_plugin nvparsers nvonnxparser
  cuda cudart
  )
target_include_directories(detector PRIVATE
  extra/
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  include/pcdUtils
  include/sensor
  )

# Create an executable
# add_executable(trt_lightnet
#   src/lightnet_detector.cpp
#   src/config_parser.cpp
#   ${sources}
#   )
ament_auto_add_library(trt_lightnet SHARED
  src/lightnet_detector.cpp
  src/config_parser.cpp
  ${sources}
)

target_include_directories(trt_lightnet PRIVATE
  ${EIGEN3_INCLUDE_DIR}  
  include
  include/pcdUtils
  include/sensor
  )
target_link_libraries(trt_lightnet
  ${OpenCV_LIBS}
  nvinfer nvinfer_plugin nvparsers nvonnxparser
  cuda cudart
  gflags
  boost_system boost_filesystem
  cnpy
  z
  dl
  "stdc++fs"
  )
  
rclcpp_components_register_node(trt_lightnet
  PLUGIN "trt_lightnet::TrtLightnetNode"
  EXECUTABLE lightnet_node
  )

ament_auto_package(INSTALL_TO_SHARE
  launch
  configs
  data
  )

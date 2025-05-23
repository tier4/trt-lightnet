cmake_minimum_required(VERSION 3.10) # Recommended version
project(trt-lightnet VERSION 1.0 LANGUAGES CXX CUDA)

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
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
else()
    set(CUDA_HOME /usr/local/cuda)
endif()

include_directories(${CUDA_HOME}/include include)
link_directories(${CUDA_HOME}/lib64)

# Find external libraries
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Define compile-time constants
add_compile_options(-D LIGHTNET_STANDALONE)

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
  src/tensorrt_lightnet/tensorrt_lightnet_ctypes.cpp
  )

# Create a shared library
add_library(lightnetinfer SHARED ${sources})

target_link_libraries(lightnetinfer
  nvinfer nvinfer_plugin nvparsers nvonnxparser
  cuda cudart
  gflags
  boost_system boost_filesystem
  cnpy
  z
  dl
  ${OpenCV_LIBS}
  "stdc++fs"
  )
target_include_directories(lightnetinfer PRIVATE
  extra/
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_HOME}/include
  ${EIGEN3_INCLUDE_DIR}
  include/pcdUtils
  include/sensor
  )

# Create an executable
add_executable(trt-lightnet
  src/lightnet_detector.cpp
  src/config_parser.cpp
  ${sources}
  )
target_include_directories(trt-lightnet PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  include
  include/pcdUtils
  include/sensor
  )
target_link_libraries(trt-lightnet
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

install(TARGETS lightnetinfer LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

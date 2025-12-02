cmake_minimum_required(VERSION 3.10)

# ============================================
# Project
# ============================================
project(trt-lightnet VERSION 1.0 LANGUAGES CXX CUDA)

# ============================================
# Build configuration
# ============================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")
# Uncomment for debugging:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")

# Ensure the loader can find co-located shared libraries at runtime
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# ============================================
# Dependencies (OpenCV, Eigen, Boost)
# ============================================
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost QUIET COMPONENTS filesystem system)

# ============================================
# CUDA Toolkit discovery
# - Prefer CUDAToolkit package
# - Fallback to auto-detect CUDA 12.x "targets/<arch>" layout
# ============================================
find_package(CUDAToolkit QUIET)

# Variables filled by either path:
set(_CUDA_INCLUDE_DIRS "")
set(_CUDA_LINK_DIRS     "")
set(_CUDA_CUDART_LIB    "")

if(CUDAToolkit_FOUND)
  # Use modern imported targets and include directories
  list(APPEND _CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  # cudart as imported target
  set(_CUDA_CUDART_LIB CUDA::cudart)
else()
  # Try environment variable first
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
  endif()

  # Auto-detect from typical locations (CUDA 12.x layout)
  if(NOT CUDA_HOME)
    file(GLOB _CUDA_HDRS /usr/local/cuda*/targets/*/include/cuda_runtime_api.h)
    list(LENGTH _CUDA_HDRS _CUDA_HDRS_LEN)
    if(_CUDA_HDRS_LEN GREATER 0)
      list(GET _CUDA_HDRS 0 _CUDA_HDR)
      get_filename_component(_CUDA_INC        "${_CUDA_HDR}" DIRECTORY)      # .../targets/<arch>/include
      get_filename_component(_CUDA_TARGET_DIR "${_CUDA_INC}" DIRECTORY)       # .../targets/<arch>
      get_filename_component(CUDA_HOME       "${_CUDA_TARGET_DIR}" DIRECTORY) # .../cuda-XX.Y
      message(STATUS "Auto-detected CUDA_HOME=${CUDA_HOME}")
    endif()
  endif()

  if(CUDA_HOME)
    # Include both top-level and targets include directories
    list(APPEND _CUDA_INCLUDE_DIRS
      ${CUDA_HOME}/include
      ${CUDA_HOME}/targets/x86_64-linux/include
      )
    # Link directories for both classic and targets libs
    list(APPEND _CUDA_LINK_DIRS
      ${CUDA_HOME}/lib64
      ${CUDA_HOME}/targets/x86_64-linux/lib
      )
    # Fallback to link cudart by name when no CUDAToolkit package
    set(_CUDA_CUDART_LIB cudart)
  else()
    message(FATAL_ERROR
      "CUDA Toolkit headers not found. Install CUDA or pass -DCUDAToolkit_ROOT=/path/to/cuda "
      "or set CUDA_HOME (e.g., /usr/local/cuda-12.x).")
  endif()
endif()

# Expose CUDA link directories (needed when not using CUDAToolkit imported targets)
foreach(_dir IN LISTS _CUDA_LINK_DIRS)
  link_directories(${_dir})
endforeach()

# ============================================
# TensorRT discovery
# - NVINFER / NVINFER_PLUGIN are required
# - NVONNXPARSER and legacy NVPARSERS are optional
# ============================================
if(DEFINED ENV{TENSORRT_ROOT})
  list(APPEND CMAKE_PREFIX_PATH  $ENV{TENSORRT_ROOT})
  list(APPEND CMAKE_LIBRARY_PATH $ENV{TENSORRT_ROOT}/lib)
  list(APPEND CMAKE_INCLUDE_PATH $ENV{TENSORRT_ROOT}/include)
endif()

find_library(NVINFER_LIB        nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)
if(NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT libraries nvinfer / nvinfer_plugin not found. "
    "Set TENSORRT_ROOT or extend CMAKE_LIBRARY_PATH.")
endif()

# Optional: ONNX parser (TRT8/10)
find_library(NVONNXPARSER_LIB   nvonnxparser)
# Optional: legacy parsers (usually absent on TRT10)
find_library(NVPARSERS_LIB      nvparsers)

# ============================================
# cnpy (Numpy .npy/.npz reader)
# - Try find_package
# - Try find_library
# - Optionally FetchContent if missing
# ============================================
find_package(cnpy QUIET)
if(NOT cnpy_FOUND)
  find_library(CNPY_LIB cnpy)
endif()

option(FETCH_CNPY_IF_MISSING "Fetch cnpy if not installed" ON)
if(NOT cnpy_FOUND AND NOT CNPY_LIB AND FETCH_CNPY_IF_MISSING)
  include(FetchContent)
  FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        master
    )
  FetchContent_MakeAvailable(cnpy)
endif()

# ============================================
# Sources
# - Build a shared library (lightnetinfer)
# - Executable links against the library
# ============================================
add_compile_options(-D LIGHTNET_STANDALONE)

file(GLOB_RECURSE SOURCES
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
  src/tensorrt_lightnet/tensorrt_lightnet.cpp
  src/sensor/CalibratedSensorParser.cpp
  src/sensor/SensorParser.cpp
  src/pcdUtils/pcd2image.cpp
  src/fswp/fswp.cpp
  src/preprocess.cu
  src/tensorrt_lightnet/tensorrt_lightnet_ctypes.cpp
  src/config_parser.cpp
  )

add_library(lightnetinfer SHARED ${SOURCES})
set_target_properties(lightnetinfer PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(lightnetinfer PRIVATE
  extra/
  include
  include/pcdUtils
  include/sensor
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${_CUDA_INCLUDE_DIRS}
  )

# ============================================
# Linking for the shared library
# ============================================
# Core TensorRT
target_link_libraries(lightnetinfer PRIVATE ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB})

# Optional TensorRT parsers (link only if found)
if(NVONNXPARSER_LIB)
  target_link_libraries(lightnetinfer PRIVATE ${NVONNXPARSER_LIB})
endif()
if(NVPARSERS_LIB)
  target_link_libraries(lightnetinfer PRIVATE ${NVPARSERS_LIB})
endif()

# CUDA runtime
if(_CUDA_CUDART_LIB)
  target_link_libraries(lightnetinfer PRIVATE ${_CUDA_CUDART_LIB})
endif()

# OpenMP
target_link_libraries(lightnetinfer PRIVATE OpenMP::OpenMP_CXX)

# gflags: PUBLIC so downstream executables get symbols for flag registration
find_package(gflags QUIET)
if(gflags_FOUND)
  target_link_libraries(lightnetinfer PUBLIC gflags)
else()
  target_link_libraries(lightnetinfer PUBLIC gflags)
endif()

# Boost (prefer imported targets; fallback to raw names)
if(Boost_FOUND)
  target_link_libraries(lightnetinfer PRIVATE Boost::filesystem Boost::system)
else()
  target_link_libraries(lightnetinfer PRIVATE boost_system boost_filesystem)
endif()

# OpenCV, zlib, dl
target_link_libraries(lightnetinfer PRIVATE ${OpenCV_LIBS} z dl)

# --- cnpy (PUBLIC so executables get the symbols)
if(cnpy_FOUND)
  target_link_libraries(lightnetinfer PUBLIC cnpy)
elseif(CNPY_LIB)
  target_link_libraries(lightnetinfer PUBLIC ${CNPY_LIB})
endif()

# stdc++fs is required if code uses experimental::filesystem symbols on some libstdc++ versions
# Link PUBLIC to ensure executables inherit the dependency
target_link_libraries(lightnetinfer PUBLIC stdc++fs)

# ============================================
# Executable
# ============================================
add_executable(trt-lightnet
  src/lightnet_detector.cpp
  )

# Get cnpy include directories from the target if available
if(TARGET cnpy)
  get_target_property(CNPY_INCLUDE_DIRS cnpy INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT CNPY_INCLUDE_DIRS)
    # Fallback: use FetchContent source directory
    set(CNPY_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/_deps/cnpy-src")
  endif()
endif()

target_include_directories(trt-lightnet PRIVATE
  include
  include/pcdUtils
  include/sensor
  ${EIGEN3_INCLUDE_DIR}
  ${_CUDA_INCLUDE_DIRS}
  ${CNPY_INCLUDE_DIRS}
  )

# Link against the shared library and other required libs
# z is explicitly added because the executable may reference zlib symbols directly/indirectly
# cnpy is explicitly linked to resolve symbols and inherit include directories
if(TARGET cnpy)
  target_link_libraries(trt-lightnet PRIVATE lightnetinfer cnpy ${OpenCV_LIBS} z)
elseif(CNPY_LIB)
  target_link_libraries(trt-lightnet PRIVATE lightnetinfer ${CNPY_LIB} ${OpenCV_LIBS} z)
else()
  target_link_libraries(trt-lightnet PRIVATE lightnetinfer ${OpenCV_LIBS} z)
endif()

# If using CUDAToolkit, include dirs are already covered; fallback path handled above

# ============================================
# Install
# ============================================
install(TARGETS lightnetinfer
  LIBRARY DESTINATION lib
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
  )

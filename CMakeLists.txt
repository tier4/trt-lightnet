cmake_minimum_required(VERSION 3.1) 
project(trt-lightnet VERSION 1.0)            
enable_language(CUDA)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings -Wall ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

add_compile_options(-D LIGHTNET_STANDALONE)

include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
find_package(OpenCV REQUIRED)

#detector lib (cuda)
file(GLOB_RECURSE sources src/tensorrt_common/tensorrt_common.cpp src/tensorrt_common/simple_profiler.cpp src/tensorrt_lightnet/tensorrt_lightnet.cpp)
add_library(detector SHARED ${sources})

target_link_libraries(detector  nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart  "stdc++fs")
target_include_directories(detector PRIVATE extra/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include )


add_executable(trt-lightnet src/lightnet_detector.cpp src/config_parser.cpp ${sources})

target_include_directories(trt-lightnet PRIVATE include)
target_link_libraries(trt-lightnet  ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart gflags boost_system boost_filesystem "stdc++fs" "stdc++ -ldl")





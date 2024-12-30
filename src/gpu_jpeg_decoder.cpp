// ref: https://docs.nvidia.com/cuda/nvjpeg/index.html

#include "gpu_jpeg_decoder.hpp"
#include <opencv2/core/core.hpp>

namespace trt_lightnet {
GPUJpegDecoder::GPUJpegDecoder()
{
  CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handler_));
  CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handler_, &nvjpeg_state_));
  for (size_t i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
    nvjpeg_image_.channel[i] = nullptr;
    nvjpeg_image_.pitch[i] = 0;
  }
}

GPUJpegDecoder::~GPUJpegDecoder()
{
  CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state_));
  CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handler_));

  for (size_t i = 0; i < NVJPEG_MAX_COMPONENT; i++) {
    if (nvjpeg_image_.channel[i] != nullptr) {
      CHECK_CUDA(cudaFree(nvjpeg_image_.channel[i]));
      nvjpeg_image_.pitch[i] = 0;
    }
  }
}

cv::Mat GPUJpegDecoder::decode(const std::vector<uint8_t>& encoded_data)
{
  // Retrieve the width and height information from the JPEG-encoded image
  int num_components = 0;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT] = {};
  int heights[NVJPEG_MAX_COMPONENT] = {};
  CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handler_, encoded_data.data(),
                                  encoded_data.size(), &num_components, &subsampling,
                                  &widths[0], &heights[0]));

  // the width/height of the fist channel should be equal to the original image size
  int width = widths[0];
  int height = heights[0];

  if (nvjpeg_image_.channel[0] == nullptr) {
    // For output format NVJPEG_OUTPUT_BGRI, the output is written only to channel[0] of
    // nvjpegImage_t, and the other channels are not touched
    // ref: https://docs.nvidia.com/cuda/nvjpeg/index.html#single-image-decoding
    CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&nvjpeg_image_.channel[0]),
                               &nvjpeg_image_.pitch[0],
                               width * 3,  // in bytes
                               height));
  }

  // Decode
  // Since this operation is asynchronous with regard to the host,
  // synchronization is basically required. However, cudaMemcpy2D performs synchronization
  // to the launched stream implicitly, no explicit synchronization operation is called here.
  CHECK_NVJPEG(nvjpegDecode(nvjpeg_handler_, nvjpeg_state_, encoded_data.data(),
                            encoded_data.size(), NVJPEG_OUTPUT_BGRI, &nvjpeg_image_,
                            cudaStreamDefault));

  // Pack decode result into cv::Mat
  cv::Mat decoded_result = cv::Mat::zeros(height, width, CV_8UC3);
  CHECK_CUDA(cudaMemcpy2D(decoded_result.data, decoded_result.step, nvjpeg_image_.channel[0],
                          nvjpeg_image_.pitch[0], width * 3, height, cudaMemcpyDeviceToHost));
  return decoded_result;
}

}  // namespace trt_lightnet

/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <fswp/fswp.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"

namespace fswp {

FaceSwapper::FaceSwapper(const std::filesystem::path &onnx_path,
                         const tensorrt_common::BuildConfig &build_config,
                         const std::size_t batch_size, std::string precision,
                         const std::size_t max_workspace_size)
    : batch_size_(batch_size) {
  tensorrt_common::BatchConfig batch_config;
  std::int32_t bs = static_cast<std::int32_t>(batch_size);
  if (bs <= 0) {
    throw std::invalid_argument("Expects batch_size > 1.");
  } else if (bs == 1) {
    batch_config = {1, 1, 1};
  } else {
    batch_config = {1, bs / 2, bs};
  }
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      onnx_path.string(), precision, nullptr, batch_config, max_workspace_size, build_config);

  trt_common_->setup();
  if (!trt_common_->isInitialized()) {
    throw std::runtime_error("TensorRT engine initialization failed.");
  }

  // Allocate GPU memory for inputs and outputs based on tensor dimensions.
  allocateMemory();
}

void FaceSwapper::allocateMemory() {
  for (std::int32_t i = 0; i < trt_common_->getNbBindings(); i++) {
    const auto name = trt_common_->getIOTensorName(i);
    const auto dims = trt_common_->getBindingDimensions(i);
    const auto data_type = trt_common_->getBindingDataType(i);
    const bool is_input = trt_common_->bindingIsInput(i);
    if (is_input) {
      std::cout << "(Input)  ";
    } else {
      std::cout << "(Output) ";
    }
    std::string shape;
    for (std::int32_t j = 0; j < dims.nbDims; j++) {
      shape += std::to_string(dims.d[j]);
      if (j != dims.nbDims - 1) shape += "x";
    }
    std::cout << name << " => " << shape << " (" << trt_common_->dataType2String(data_type) << ")"
              << std::endl;

    // Calculate the tensor volume (without batch dimension).
    const std::size_t volume =
        std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int>());
    if (is_input) {  // Input tensor
      input_d_.push_back(cuda_utils::make_unique<float[]>(batch_size_ * volume));
    } else {  // Output tensors
      output_d_.push_back(cuda_utils::make_unique<float[]>(batch_size_ * volume));
      output_h_.push_back(
          cuda_utils::make_unique_host<float[]>(batch_size_ * volume, cudaHostAllocPortable));
    }
  }
}

FaceSwapper::~FaceSwapper() {
  // Cleanup if needed.
}

void FaceSwapper::printProfiling(void) {
  if (trt_common_) {  // Ensure trt_common_ is not a nullptr before attempting
                      // to call its methods.
    trt_common_->printProfiling();
  } else {
    std::cerr << "Error: TRTCommon instance is not initialized." << std::endl;
  }
}

cv::Mat FaceSwapper::inpaint(const cv::Mat &image, const std::vector<fswp::BBox> &bboxes) {
  if (!trt_common_->isInitialized())
    throw std::runtime_error("Initialize trt_common_ first before perform inpainting.");
  if (bboxes.size() == 0) return image;
  nvinfer1::Dims input_image_shape = trt_common_->getBindingDimensions(0);
  nvinfer1::Dims input_mask_shape = trt_common_->getBindingDimensions(1);
  const std::size_t input_image_h = input_image_shape.d[2];
  const std::size_t input_image_w = input_image_shape.d[3];
  const std::size_t input_image_c = input_image_shape.d[1];
  assert(input_image_h == input_image_w);
  const auto input_size = cv::Size(input_image_w, input_image_h);
  const std::size_t num_bboxes = bboxes.size();

  // Create crops and masks
  std::vector<cv::Mat> masks;
  std::vector<cv::Mat> crops;
  std::vector<cv::Rect> crop_rois;
  std::vector<cv::Rect> bbox_rois;

  for (std::size_t i = 0; i < num_bboxes; i++) {
    const auto &bbox = bboxes[i];
    const auto w = bbox.x2 - bbox.x1;
    const auto h = bbox.y2 - bbox.y1;
    const auto xc = (bbox.x1 + bbox.x2) / 2.0f;
    const auto yc = (bbox.y1 + bbox.y2) / 2.0f;
    const cv::Rect bbox_roi = cv::Rect(bbox.x1, bbox.y1, w, h);
    bbox_rois.push_back(bbox_roi);


    // Create crop
    const auto crop_size = 1.4f * std::max(w, h);
    const auto crop_xmin = xc - crop_size / 2.0f;
    const auto crop_ymin = yc - crop_size / 2.0f;
    const cv::Rect crop_roi = cv::Rect(crop_xmin, crop_ymin, crop_size, crop_size);
    const cv::Rect image_roi = cv::Rect({}, image.size());
    const auto and_roi = image_roi & crop_roi;
    cv::Mat crop = cv::Mat::zeros(crop_roi.size(), image.type());
    image(and_roi).copyTo(crop(and_roi - crop_roi.tl()));
    cv::resize(crop, crop, input_size);
    crop.convertTo(crop, CV_32FC3, 1 / 127.5, -1.0);  // [-1.0, 1.0]
    crop_rois.push_back(crop_roi);

    // Create mask
    const cv::Rect mask_roi = bbox_roi - crop_roi.tl();
    const float scale = static_cast<float>(input_image_h) / crop_size;
    cv::Rect mask_roi_scaled(mask_roi.x * scale, mask_roi.y * scale, mask_roi.width * scale,
                             mask_roi.height * scale);
    cv::Mat mask = cv::Mat::ones(input_size, CV_32FC1);
    mask(mask_roi_scaled).setTo(cv::Scalar(0));
    crop(mask_roi_scaled).setTo(cv::Scalar(0));
    crops.push_back(crop);
    masks.push_back(mask);
  }
  
  std::vector<cv::Mat> outs;
  std::size_t batch_size = 0;
  for (std::size_t i = 0; i < num_bboxes; i += batch_size) {
    batch_size = std::min(batch_size_, static_cast<std::size_t>(num_bboxes - i));
    input_image_shape.d[0] = batch_size;
    input_mask_shape.d[0] = batch_size;
    trt_common_->setBindingDimensions(0, input_image_shape);
    trt_common_->setBindingDimensions(1, input_mask_shape);
    const auto batch_crops = cv::dnn::blobFromImages(
        std::vector<cv::Mat>(crops.begin() + i, crops.begin() + i + batch_size), 1.0, input_size,
        cv::Scalar(0, 0, 0), true);
    const auto batch_masks = cv::dnn::blobFromImages(
        std::vector<cv::Mat>(masks.begin() + i, masks.begin() + i + batch_size));

    // Host -> GPU
    std::vector<float> batch_crops_flatten =
        batch_crops.isContinuous() ? batch_crops.reshape(1, batch_crops.total())
                                   : batch_crops.reshape(1, batch_crops.total()).clone();
    std::vector<float> batch_masks_flatten =
        batch_masks.isContinuous() ? batch_masks.reshape(1, batch_masks.total())
                                   : batch_masks.reshape(1, batch_masks.total()).clone();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_d_[0].get(), batch_crops_flatten.data(),
                                     batch_crops_flatten.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_d_[1].get(), batch_masks_flatten.data(),
                                     batch_masks_flatten.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, *stream_));

   
    // Infer
    std::vector<void *> buffs{input_d_[0].get(), input_d_[1].get(), output_d_[0].get()};
    trt_common_->enqueueV2(buffs.data(), *stream_, nullptr);
    
    // GPU -> Host
    const auto output_dims = trt_common_->getBindingDimensions(2);
    const std::size_t sample_output_size = std::accumulate(
        output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<std::size_t>());
    const std::size_t batch_output_size = batch_size * sample_output_size;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_h_[0].get(), output_d_[0].get(),
                                     sizeof(float) * batch_output_size, cudaMemcpyDeviceToHost,
                                     *stream_));
    cudaStreamSynchronize(*stream_);

    // NCHW -> NHWC
    unsigned char *nhwc = new unsigned char[batch_output_size];    
    float *buf = output_h_[0].get();    
    const std::size_t size_chw = input_image_c * input_image_h * input_image_w;
    const std::size_t size_hw = input_image_h * input_image_w;
    const std::size_t size_wc = input_image_w * input_image_c;
    for (std::size_t b_idx = 0; b_idx < batch_size; b_idx++) {
      for (std::size_t c_idx = 0; c_idx < input_image_c; c_idx++) {
        for (std::size_t h_idx = 0; h_idx < input_image_h; h_idx++) {
          for (std::size_t w_idx = 0; w_idx < input_image_w; w_idx++) {
            const std::size_t nchw_idx =
	      b_idx * size_chw + c_idx * size_hw + h_idx * input_image_w + w_idx;
            const std::size_t nhwc_idx =
	      b_idx * size_chw + h_idx * size_wc + w_idx * input_image_c + (input_image_c-1-c_idx);
            nhwc[nhwc_idx] =  (unsigned char)(buf[nchw_idx] * 127.5 + 127.5); //RGB2BGR + Denormalization
          }
        }
      }
    }

    for (std::size_t j = 0; j < batch_size; j++) {
      unsigned char *buff = nhwc + (sample_output_size * j);
      cv::Mat out(input_size, CV_8UC3);      
      std::memcpy(out.data, buff, sample_output_size * sizeof (unsigned char));
      outs.push_back(out);
    }
    delete[] nhwc;    
  }
  assert(outs.size() == num_bboxes);

  // Swap
  cv::Mat inpainted(image);
  for (std::size_t i = 0; i < outs.size(); i++) {
    const auto &crop_roi = crop_rois[i];
    const auto &bbox_roi = bbox_rois[i];
    const auto image_roi = cv::Rect({}, inpainted.size());
    auto &out = outs[i];
    cv::resize(out, out, crop_roi.size());
    out(bbox_roi - crop_roi.tl()).copyTo(inpainted(bbox_roi - image_roi.tl()));
  }
  
  return inpainted;
}

}  // namespace fswp

// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORRT_LIGHTNET__PREPROCESS_HPP_
#define TENSORRT_LIGHTNET__PREPROCESS_HPP_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <tensorrt_lightnet/tensorrt_lightnet.hpp>

namespace tensorrt_lightnet
{
struct Roi
{
  int x;
  int y;
  int w;
  int h;
};

/**
 * @brief Resize a image using bilinear interpolation on gpus
 * @param[out] dst Resized image
 * @param[in] src image
 * @param[in] d_w width for resized image
 * @param[in] d_h height for resized image
 * @param[in] d_c channel for resized image
 * @param[in] s_w width for input image
 * @param[in] s_h height for input image
 * @param[in] s_c channel for input image
 * @param[in] stream cuda stream
 */
extern void resize_bilinear_gpu(
  unsigned char * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c,
  cudaStream_t stream);

/**
 * @brief Letterbox a image on gpus
 * @param[out] dst letterbox-ed image
 * @param[in] src image
 * @param[in] d_w width for letterbox-ing
 * @param[in] d_h height for letterbox-ing
 * @param[in] d_c channel for letterbox-ing
 * @param[in] s_w width for input image
 * @param[in] s_h height for input image
 * @param[in] s_c channel for input image
 * @param[in] stream cuda stream
 */
extern void letterbox_gpu(
  unsigned char * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c,
  cudaStream_t stream);

/**
 * @brief NHWC to NHWC conversion
 * @param[out] dst converted image
 * @param[in] src image
 * @param[in] d_w width for a image
 * @param[in] d_h height for a image
 * @param[in] d_c channel for a image
 * @param[in] stream cuda stream
 */
extern void nchw_to_nhwc_gpu(
  unsigned char * dst, unsigned char * src, int d_w, int d_h, int d_c, cudaStream_t stream);

/**
 * @brief Unsigned char to float32 for inference
 * @param[out] dst32 converted image
 * @param[in] src image
 * @param[in] d_w width for a image
 * @param[in] d_h height for a image
 * @param[in] d_c channel for a image
 * @param[in] stream cuda stream
 */
extern void to_float_gpu(
  float * dst32, unsigned char * src, int d_w, int d_h, int d_c, cudaStream_t stream);

/**
 * @brief Resize and letterbox a image using bilinear interpolation on gpus
 * @param[out] dst processed image
 * @param[in] src image
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] d_c channel for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] stream cuda stream
 */
extern void resize_bilinear_letterbox_gpu(
  unsigned char * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c,
  cudaStream_t stream);

/**
 * @brief Optimized preprocessing including resize, letterbox, nhwc2nchw, toFloat and normalization
 * for LIGHTNET on gpus
 * @param[out] dst processed image
 * @param[in] src image
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] d_c channel for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] norm normalization
 * @param[in] stream cuda stream
 */
extern void resize_bilinear_letterbox_nhwc_to_nchw32_gpu(
  float * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c,
  float norm, cudaStream_t stream);

/**
 * @brief Optimized preprocessing including resize, letterbox, nhwc2nchw, toFloat and normalization
 * with batching for LIGHTNET on gpus
 * @param[out] dst processed image
 * @param[in] src image
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] d_c channel for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] batch batch size
 * @param[in] norm normalization
 * @param[in] stream cuda stream
 */
extern void resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
  float * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c, int batch,
  float norm, cudaStream_t stream);

/**
 * @brief Optimized preprocessing including crop, resize, letterbox, nhwc2nchw, toFloat and
 * normalization with batching for LIGHTNET on gpus
 * @param[out] dst processed image
 * @param[in] src image
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] d_c channel for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] d_roi regions of interest for cropping
 * @param[in] batch batch size
 * @param[in] norm normalization
 * @param[in] stream cuda stream
 */
extern void crop_resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
  float * dst, unsigned char * src, int d_w, int d_h, int d_c, Roi * d_roi, int s_w, int s_h,
  int s_c, int batch, float norm, cudaStream_t stream);

/**
 * @brief Optimized multi-scale preprocessing including crop, resize, letterbox, nhwc2nchw, toFloat
 * and normalization with batching for LIGHTNET on gpus
 * @param[out] dst processed image
 * @param[in] src image
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] d_c channel for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] d_roi regions of interest for cropping
 * @param[in] batch batch size
 * @param[in] norm normalization
 * @param[in] stream cuda stream
 */
extern void multi_scale_resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
  float * dst, unsigned char * src, int d_w, int d_h, int d_c, Roi * d_roi, int s_w, int s_h,
  int s_c, int batch, float norm, cudaStream_t stream);

/**
 * @brief Argmax on GPU
 * @param[out] dst processed image
 * @param[in] src probabilty map
 * @param[in] d_w width for output
 * @param[in] d_h height for output
 * @param[in] s_w width for input
 * @param[in] s_h height for input
 * @param[in] s_c channel for input
 * @param[in] batch batch size
 * @param[in] stream cuda stream
 */
extern void argmax_gpu(
  unsigned char * dst, float * src, int d_w, int d_h, int s_w, int s_h, int s_c, int batch, cudaStream_t stream);
}  // namespace tensorrt_lightnet
#endif  // TENSORRT_LIGHTNET__PREPROCESS_HPP_

extern void blobFromImageGpu(float *dst, unsigned char*src, int d_w, int d_h, int d_c,
		      int s_w, int s_h, int s_c, float norm, cudaStream_t stream);

extern void generateDepthmapGpu(unsigned char *depthmap, const float *buf, const unsigned char* colormap,
				int width, int height, cudaStream_t stream);

extern void smoothDepthmapGpu(float *depthmap, const int *argmax, const int depthWidth, const int depthHeight, const int segWidth, const int segHeight,  int* road_ids, int numRoadIds, cudaStream_t stream);

extern void addWeightedGpu(unsigned char *output, unsigned char *src1,  unsigned char *src2,
			   float alpha, float beta, float gamma,
			   int width, int height, int channel, cudaStream_t stream);

extern void resizeNearestNeighborGpu(unsigned char *dst, unsigned char*src, int d_w, int d_h, int d_c,
				     int s_w, int s_h, int s_c, cudaStream_t stream);


extern void computeEntropyMapGpu(const float* buf, float* entropy_maps,
				 int chan, int height, int width, cudaStream_t stream);

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <tensorrt_lightnet/preprocess.hpp>

#define BLOCK 512

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d;
    d.x = x;
    d.y = y;
    d.z = 1;
    return d;
}

__device__ double lerp1d(int a, int b, float w)
{
  return fma(w, (float)b, fma(-w, (float)a, (float)a));
}

__device__ float lerp2d(int f00, int f01, int f10, int f11, float centroid_h, float centroid_w)
{
  centroid_w = (1 + lroundf(centroid_w) - centroid_w) / 2;
  centroid_h = (1 + lroundf(centroid_h) - centroid_h) / 2;

  float r0, r1, r;
  r0 = lerp1d(f00, f01, centroid_w);
  r1 = lerp1d(f10, f11, centroid_w);

  r = lerp1d(r0, r1, centroid_h);  //+ 0.00001
  return r;
}


__global__ void SimpleblobFromImageKernel(int N, float* dst_img, unsigned char* src_img, 
				       int dst_h, int dst_w, int src_h, int src_w, 
				    float stride_h, float stride_w, float norm)
{
  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if (index >= N) return;
  int chan = 3; 
  int w = index % dst_w;
  int h = index / dst_w;
  float centroid_h, centroid_w;
  int c;
  centroid_h = stride_h * (float)(h + 0.5); 
  centroid_w = stride_w * (float)(w + 0.5);
  int src_h_idx = lroundf(centroid_h)-1;
  int src_w_idx = lroundf(centroid_w)-1;
  if (src_h_idx<0){src_h_idx=0;}
  if (src_w_idx<0){src_w_idx=0;}  
  index = chan * src_w_idx + chan* src_w * src_h_idx;

  for (c = 0; c < chan; c++) {
    int dst_index = w + (dst_w*h) + (dst_w*dst_h*(2-c));              
    dst_img[dst_index] = (float)src_img[index+c]*norm;
  }
}

__global__ void blobFromImageKernel(int N, float* dst_img, unsigned char* src_img, 
				       int dst_h, int dst_w, int src_h, int src_w, 
				    float stride_h, float stride_w, float norm)
{
  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if (index >= N) return;
  int chan = 3; 
  int w = index % dst_w;
  int h = index / dst_w;
  float centroid_h, centroid_w;
  int c;
  centroid_h = stride_h * (float)(h + 0.5); 
  centroid_w = stride_w * (float)(w + 0.5);
  int src_h_idx = lroundf(centroid_h)-1;
  int src_w_idx = lroundf(centroid_w)-1;
  if (src_h_idx<0){src_h_idx=0;}
  if (src_w_idx<0){src_w_idx=0;}
  int next_w, next_h;
  next_w = ((src_w_idx+1) < src_w) ? src_w_idx+1 : src_w_idx;
  next_h = ((src_h_idx+1) < src_h) ? src_h_idx+1 : src_h_idx;  
  int index00 = chan * src_w_idx + chan* src_w * src_h_idx;
  int index01 = chan * (next_w) + chan* src_w * src_h_idx;
  
  int index10 = chan * src_w_idx + chan* src_w * (next_h);
  int index11 = chan * (next_w) + chan* src_w * (next_h);  

  for (c = 0; c < chan; c++) {
    int dst_index = w + (dst_w*h) + (dst_w*dst_h*(2-c));
    float rs = lroundf(lerp2d(
			      (int)src_img[index00+c], (int)src_img[index01+c], (int)src_img[index10+c], (int)src_img[index11+c], centroid_h,
			      centroid_w));    
    dst_img[dst_index] = rs*norm;
  }
}

void blobFromImageGpu(float *dst, unsigned char*src, int d_w, int d_h, int d_c,
			 int s_w, int s_h, int s_c, float norm, cudaStream_t stream)
{
  int N =  d_w * d_h;
  float stride_h = (float)s_h / (float)d_h;
  float stride_w = (float)s_w / (float)d_w;
  //SimpleblobFromImageKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, dst, src,
  blobFromImageKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, dst, src,   
							      d_h, d_w,
							      s_h, s_w,
							      stride_h, stride_w, norm);

}

__global__ void resizeNearestNeighborKernel(int N, unsigned char* dst_img, unsigned char* src_img, 
				       int dst_h, int dst_w, int src_h, int src_w, 
				    float stride_h, float stride_w)
{
  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if (index >= N) return;
  int chan = 3; 
  int w = index % dst_w;
  int h = index / dst_w;
  float centroid_h, centroid_w;
  int c;
  centroid_h = stride_h * (float)(h + 0.5); 
  centroid_w = stride_w * (float)(w + 0.5);
  int src_h_idx = lroundf(centroid_h)-1;
  int src_w_idx = lroundf(centroid_w)-1;
  if (src_h_idx<0){src_h_idx=0;}
  if (src_w_idx<0){src_w_idx=0;}  
  index = chan * src_w_idx + chan* src_w * src_h_idx;
  int dst_index = (chan * dst_w * h) + (chan * w);
    
  for (c = 0; c < chan; c++) {
    //NHWC
    dst_img[dst_index + c] = src_img[index + c];
  }
}

void resizeNearestNeighborGpu(unsigned char *dst, unsigned char*src, int d_w, int d_h, int d_c,
			 int s_w, int s_h, int s_c, cudaStream_t stream)
{
  int N =  d_w * d_h;
  float stride_h = (float)s_h / (float)d_h;
  float stride_w = (float)s_w / (float)d_w;
  resizeNearestNeighborKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, dst, src,   
							      d_h, d_w,
							      s_h, s_w,
							      stride_h, stride_w);

}


__global__ void smoothDepthmapKernel(float* depthBuf, const int* segBuf,
				     int width, int height, int segWidth, int segHeight,
				     float scaleW, float scaleH, int* road_ids, int numRoadIds) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height) {
    float sum = 0.0f;
    int count = 0;    
    for (int x = 0; x < width; ++x) {
      int segX = static_cast<int>(x / scaleW);
      int segY = static_cast<int>(y / scaleH);
      int segIdx = segX + segWidth * segY;

      int id = segBuf[segIdx];
      bool isRoad = false;
      for (int i = 0; i < numRoadIds; ++i) {
	if (id == road_ids[i]) {
	  isRoad = true;
	  break;
	}
      }
      if (isRoad) {  // Road-related segmentation ID
	count++;
	sum += depthBuf[x + y * width];
      }
    }
    if (count > 0) {
      sum /= count;
    }
    for (int x = 0; x < width; ++x) {
      int segX = static_cast<int>(x / scaleW);
      int segY = static_cast<int>(y / scaleH);
      int segIdx = segX + segWidth * segY;

      int id = segBuf[segIdx];
      bool isRoad = false;
      for (int i = 0; i < numRoadIds; ++i) {
	if (id == road_ids[i]) {
	  isRoad = true;
	  break;
	}
      }
      if (isRoad) {  // Road-related segmentation ID
	depthBuf[x + y * width] = sum;
      }
    }
  }
}

void smoothDepthmapGpu(float *depthmap, const int *argmax, const int depthWidth, const int depthHeight, const int segWidth, const int segHeight,  int* road_ids, int numRoadIds, cudaStream_t stream)
{
  dim3 threadsPerBlock(256);
  dim3 numBlocks((depthHeight + threadsPerBlock.x - 1) / threadsPerBlock.x);
  const float scaleW = static_cast<float>(depthWidth) / segWidth;
  const float scaleH = static_cast<float>(depthHeight) / segHeight;
  smoothDepthmapKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(depthmap, argmax, depthWidth, depthHeight, segWidth, segHeight,
								  scaleW, scaleH, road_ids, numRoadIds);

}

__global__ void generateDepthmapKernel(int N, unsigned char* depthmap, const float *buf,  const unsigned char* colormap, int width, int height)
{
  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if (index >= N) return;
  int chan = 3; 
  int w = index % width;
  int h = index / width;
  int c;
  for (c = 0; c < chan; c++) {
    //src : NCHW (CUDA)
    //Dst : NHWC (cv::Mat)
    int src_index = w + (width * h);
    int dst_index = c + (chan * w) + (chan * width * h);
    float rel = 1.0f - buf[src_index];
    int value = static_cast<int>(rel * 255);
    int colormapIdx = 255 - value;
    depthmap[dst_index] = colormap[colormapIdx * 3 + (2-c)];
  }
}

void generateDepthmapGpu(unsigned char *depthmap, const float *buf, const unsigned char* colormap,
		     int width, int height, cudaStream_t stream)
{
  int N =  width * height;

  generateDepthmapKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, depthmap, buf, colormap, 
							     width, height);

}

__global__ void backProjectionKernel(
				     const float* depthMap, const int outputW, const int outputH, const float scale_w, const float scale_h,
				     const float gran_h, const float max_distance, const float u0, const float v0, const float fx, const float fy,
				     const int mask_w, const int mask_h, const float mask_scale_w, const float mask_scale_h,
				     const unsigned char* mask, const int maskStep, unsigned char* bevMap, const int bevStep, const int gridW, const int gridH) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= outputW || y >= outputH) return;

  int stride = outputW * y;
  float distance = depthMap[stride + x] * max_distance;
  distance = distance > max_distance ? max_distance : distance;

  float src_x = x * scale_w;
  float x3d = ((src_x - u0) / fx) * distance;

  if (x3d > 20.0) return;
  x3d = (x3d + 20.0) * gridW / 40.0;
  if (x3d > gridH || x3d < 0.0) return;

  int x_bev = static_cast<int>(x3d);
  int y_bev = static_cast<int>(gridH - static_cast<int>(distance * gran_h));
  
  if (y_bev >= 0 && y_bev < gridH && x_bev >= 0 && x_bev < gridW) {
    int mask_x = static_cast<int>(x * mask_scale_w);
    int mask_y = static_cast<int>(y * mask_scale_h);
    int xx, yy;
    if (mask) {
      bevMap[y_bev * bevStep + 3 * x_bev + 0] = mask[mask_y * maskStep + 3 * mask_x + 0];
      bevMap[y_bev * bevStep + 3 * x_bev + 1] = mask[mask_y * maskStep + 3 * mask_x + 1];
      bevMap[y_bev * bevStep + 3 * x_bev + 2] = mask[mask_y * maskStep + 3 * mask_x + 2];
      for (yy = -2; yy <= 2; yy++) {
	for (xx = 0; xx <= 0; xx++) {	
	  if ((y_bev+yy) >= 0 && (y_bev+yy) < gridH) {
	    if ((x_bev+xx) >= 0 && (x_bev+xx) < gridW) {
	      bevMap[(y_bev+yy) * bevStep + 3 * (x_bev+xx) + 0] = mask[mask_y * maskStep + 3 * mask_x + 0];
	      bevMap[(y_bev+yy) * bevStep + 3 * (x_bev+xx) + 1] = mask[mask_y * maskStep + 3 * mask_x + 1];
	      bevMap[(y_bev+yy) * bevStep + 3 * (x_bev+xx) + 2] = mask[mask_y * maskStep + 3 * mask_x + 2];
	    }
	  }
	}
      }
      
    } else {
      bevMap[y_bev * bevStep + 3 * x_bev + 0] = 255;
      bevMap[y_bev * bevStep + 3 * x_bev + 1] = 255;
      bevMap[y_bev * bevStep + 3 * x_bev + 2] = 255;
    }
  }
}

void getBackProjectionGpu(const float* d_depth, int outputW, int outputH, float scale_w, float scale_h, const Calibration calibdata,
		     int mask_w, int mask_h, const unsigned char *d_mask, int mask_step, unsigned char *d_bevMap, int bevmap_step, cudaStream_t stream)
{
  //int N =  width * height;

  dim3 blockDim(16, 16);
  dim3 gridDim((outputW + blockDim.x - 1) / blockDim.x, (outputH + blockDim.y - 1) / blockDim.y);
  float gran_h = (float)GRID_H / calibdata.max_distance;
  float mask_scale_w = mask_w / (float)outputW;
  float mask_scale_h = mask_h / (float)outputH;  
  backProjectionKernel<<<gridDim, blockDim, 0, stream>>>(
					      d_depth, outputW, outputH, scale_w, scale_h, gran_h, calibdata.max_distance,
					      calibdata.u0, calibdata.v0, calibdata.fx, calibdata.fy, mask_w, mask_h,
					      mask_scale_w, mask_scale_h,
					      d_mask, mask_step, d_bevMap, bevmap_step, GRID_W, GRID_H);

}

__global__ void mapArgmaxToColorKernel(unsigned char *output, const int *input, int width, int height, const ucharRGB *colorMap)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = y * width + x;
    int classId = (int)(input[idx]);
    //int classId = __float2int_rz(input[idx]);    
    if (classId > 0) {
      printf("%d,", classId);
    }
	
    ucharRGB color = colorMap[classId];
    //NHWC
    idx = y * width * 3 + x * 3 + 0;    
    output[idx] = color.b;
    idx = y * width * 3 + x * 3 + 1;    
    output[idx] = color.g;
    idx = y * width * 3 + x * 3 + 2;    
    output[idx] = color.r;    
  }
}

void mapArgmaxtoColorGpu(unsigned char *output, int *input, 
			 int width, int height, const ucharRGB *colorMap, cudaStream_t stream)
{
  int N =  width * height;

  mapArgmaxToColorKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(output, input,
								 width, height, colorMap);

}

__global__ void addWeightedKernel(int N, unsigned char* dst, const unsigned char* src1, const unsigned char* src2,
				  float alpha, float beta, float gamma, int width, int height, int channels) {

  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (index >= N) return;
  int x = index % width;
  int y = index / width;  

  int idx = (y * width + x) * channels;
  for (int c = 0; c < channels; ++c) {
    float value = alpha * src1[idx + c] + beta * src2[idx + c] + gamma;
    dst[idx + c] = min(max(static_cast<int>(value), 0), 255);
  }
}

void addWeightedGpu(unsigned char *output, unsigned char *src1,  unsigned char *src2,
		      float alpha, float beta, float gamma,
		      int width, int height, int channel, cudaStream_t stream)
{
  int N =  width * height;

  addWeightedKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, output, src1, src2,
							    alpha, beta, gamma,
							    width, height, channel);

}

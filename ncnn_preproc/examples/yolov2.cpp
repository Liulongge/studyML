// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

// #if defined(USE_NCNN_SIMPLEOCV)
// #include "simpleocv.h"
// #else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #endif
#include <stdio.h>
#include <vector>

 // fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float *din, float *dout, int size,
                     const std::vector<float> &mean,
                     const std::vector<float> &scale)
{
  if (mean.size() != 3 || scale.size() != 3) {
    // LOGE("[ERROR] mean or scale size must equal to 3");
    return;
  }

  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
}


// resize + normalize
int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat image = cv::imread("../images/64-ncnn.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    cv::imwrite("img_in.jpg", image);

    int img_w = image.cols;
    int img_h = image.rows;

    int resize_w = image.cols / 2.0;
    int resize_h = image.rows / 2.0;


    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC3);

   
    // ncnn::Mat image_float = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, resize_w, resize_h);
    
    // float mean[3] = { 128.f, 128.f, 128.f };
    // float norm[3] = { 1.f/128.f, 1.f/128.f, 1.f/128.f };
    // image_float.crop
    // image_float.substract_mean_normalize(mean, norm);

    uint8_t *image_u8 = (uint8_t *)malloc(resize_w * resize_h * 3);
    float *data = (float *)image_float.data;
    for(int i  = 0; i < resize_w * resize_h * 3; i++)
    {
        printf("%f\n", data[i]);
        image_u8[i] = (uint8_t )(data[i] * 128.f + 128.f);//(uint8_t )(data[i]);//
    }

    cv::Mat img_out = cv::Mat::zeros(resize_h, resize_w, CV_8UC3);
    img_out.data = image_u8;
    cv::imwrite("img_out.jpg", img_out);
    return 0;
}

// crop + resize + normalize 
// int main(int argc, char** argv)
// {
//     const char* imagepath = argv[1];

//     cv::Mat image = cv::imread("../images/64-ncnn.png", cv::IMREAD_COLOR);
//     if (image.empty())
//     {
//         fprintf(stderr, "cv::imread %s failed\n", imagepath);
//         return -1;
//     }
//     cv::imwrite("img_in.jpg", image);

//     int img_w = image.cols;
//     int img_h = image.rows;

//     int resize_w = image.cols / 2.0;
//     int resize_h = image.rows / 2.0;

//     ncnn::Mat image_float = ncnn::Mat::from_pixels_roi_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 4, 4, 32, 32, resize_w, resize_h);
//     // ncnn::Mat image_float = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, resize_w, resize_h);

//     float mean[3] = { 128.f, 128.f, 128.f };
//     float norm[3] = { 1.f/128.f, 1.f/128.f, 1.f/128.f };
//     // image_float.crop
//     image_float.substract_mean_normalize(mean, norm);

//     uint8_t *image_u8 = (uint8_t *)malloc(resize_w * resize_h * 3);
//     float *data = (float *)image_float.data;
//     for(int i  = 0; i < resize_w * resize_h * 3; i++)
//     {
//         printf("%f\n", data[i]);
//         image_u8[i] = (uint8_t )(data[i] * 128.f + 128.f);//(uint8_t )(data[i]);//
//     }

//     cv::Mat img_out = cv::Mat::zeros(resize_h, resize_w, CV_8UC3);
//     img_out.data = image_u8;
//     cv::imwrite("img_out.jpg", img_out);
//     return 0;
// }


// int main(int argc, char** argv)
// {
//     const char* imagepath = argv[1];

//     cv::Mat image = cv::imread("../images/64-ncnn.png", cv::IMREAD_COLOR);
//     if (image.empty())
//     {
//         fprintf(stderr, "cv::imread %s failed\n", imagepath);
//         return -1;
//     }
//     cv::imwrite("img_in.jpg", image);

//     int img_w = image.cols;
//     int img_h = image.rows;

//     int resize_w = image.cols / 2.0;
//     int resize_h = image.rows / 2.0;

//     ncnn::Mat image_float = ncnn::Mat::from_pixels_roi_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, 4, 4, 32, 32, resize_w, resize_h);
//     // ncnn::Mat image_float = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, resize_w, resize_h);

//     float mean[3] = { 128.f, 128.f, 128.f };
//     float norm[3] = { 1.f/128.f, 1.f/128.f, 1.f/128.f };
//     // image_float.crop
//     image_float.substract_mean_normalize(mean, norm);

//     uint8_t *image_u8 = (uint8_t *)malloc(resize_w * resize_h * 3);
//     float *data = (float *)image_float.data;
//     for(int i  = 0; i < resize_w * resize_h * 3; i++)
//     {
//         printf("%f\n", data[i]);
//         image_u8[i] = (uint8_t )(data[i] * 128.f + 128.f);//(uint8_t )(data[i]);//
//     }

//     cv::Mat img_out = cv::Mat::zeros(resize_h, resize_w, CV_8UC3);
//     img_out.data = image_u8;
//     cv::imwrite("img_out.jpg", img_out);
//     return 0;
// }
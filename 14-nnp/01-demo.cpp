#include <npp.h>
#include <opencv2/opencv.hpp>
#include <iostream>  
#include <chrono>  
#include <iomanip> // 用于std::put_time  

// auto start = std::chrono::high_resolution_clock::now();  
// auto end = std::chrono::high_resolution_clock::now();  
// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
// std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;  

// g++ -o YUV2RGB YUV2RGB.cpp `pkg-config --cflags --libs opencv4` -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart  -lnppicc -lnppig
int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("./bin image\n");
        return 0;
    }
    // opencv转yuv420 有问题 转出来的图像是单通道的，高度是原始图像的1.5倍，这里使用yuv444格式进行测试，如需使用yuv420可以使用其他库，如ffmpeg
    // 读取 BGR 图像并转换为 YUV444 格式
    cv::Mat mat_bgr_img = cv::imread(argv[1]);
    cv::Mat mat_yuv_img;
    cv::cvtColor(mat_bgr_img, mat_yuv_img, cv::COLOR_BGR2YUV); // packed YUV
auto start = std::chrono::high_resolution_clock::now();  
    cv::cvtColor(mat_yuv_img, mat_bgr_img, cv::COLOR_YUV2BGR); // packed YUV
auto end = std::chrono::high_resolution_clock::now();  
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;  
    cv::cvtColor(mat_bgr_img, mat_yuv_img, cv::COLOR_BGR2YUV); // packed YUV
    cv::imwrite("./yuv444.jpg", mat_yuv_img);
 
    int width = mat_yuv_img.cols;
    int height = mat_yuv_img.rows;
    int step = mat_yuv_img.step;
    printf("step : %d\nwidth : %d\nheight : %d\nwidth * 3 : %d\n", step, width, height, width * 3);
    if (step != width * 3) {
        printf("step != width * 3\n");
    }
    /*YUV->RGB*/
    // YUV
    Npp8u *pu8_yuv_dev = nullptr;
    cudaMalloc((void **)&pu8_yuv_dev, step * height);
    cudaMemcpy(pu8_yuv_dev, mat_yuv_img.data, step * height, cudaMemcpyHostToDevice);
    // RGB
    Npp8u *pu8_rgb_dev = nullptr;
    cudaMalloc((void **)&pu8_rgb_dev, width * height * 3);
    // 输入:packed YUV  输出:packed RGB
start = std::chrono::high_resolution_clock::now(); 
    NppStatus npp_ret = nppiYUVToRGB_8u_C3R(pu8_yuv_dev, step, pu8_rgb_dev, width * 3, {width, height});
end = std::chrono::high_resolution_clock::now(); 
duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;  
    printf("npp_ret = %d \n", npp_ret);
 
    //write to file
    unsigned char *pu8_rgb_host = (unsigned char *)malloc(width * height * 3);
    memset(pu8_rgb_host, 0, width * height * 3);
    cudaMemcpy(pu8_rgb_host, pu8_rgb_dev, width * height * 3, cudaMemcpyDeviceToHost);
 
    FILE *file = fopen("RGB.raw", "wb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open the file.\n");
        return 1;
    }
    fwrite(pu8_rgb_host, 1, width * height * 3, file);
    fclose(file);
 
    cv::Mat newimage(height, width, CV_8UC3);
    memcpy(newimage.data, pu8_rgb_host, width * height * 3);
    cv::cvtColor(newimage, newimage, cv::COLOR_RGB2BGR); // opencv默认使用的是BGR
    cv::imwrite("./yuv2BGR.jpg", newimage);
    /*resize*/ 
    Npp8u *pu8_src_data_dev = pu8_rgb_dev;
 
    Npp8u *pu8_dst_data_dev = NULL;
    int resize_width = width / 2, resize_height = height / 2;
    NppiSize npp_src_size{width, height};
    NppiSize npp_dst_size{resize_width, resize_height};
    cudaMalloc((void **)&pu8_dst_data_dev, resize_width * resize_height * 3 * sizeof(Npp8u));
    cudaMemset(pu8_dst_data_dev, 0, resize_width * resize_height * 3 * sizeof(Npp8u));
    nppiResize_8u_C3R((Npp8u *)pu8_src_data_dev, width * 3, npp_src_size, NppiRect{0, 0, width, height},
                      (Npp8u *)pu8_dst_data_dev, resize_width * 3, npp_dst_size, NppiRect{0, 0, resize_width, resize_height},
                      NPPI_INTER_LINEAR);
 
    cv::Mat newimage_resize(resize_height, resize_width, CV_8UC3);
    cudaMemcpy(newimage_resize.data, pu8_dst_data_dev, resize_height * resize_width * 3, cudaMemcpyDeviceToHost);
    cv::cvtColor(newimage_resize, newimage_resize, cv::COLOR_RGB2BGR); // opencv默认使用的是BGR
    cv::imwrite("./rzImage_npp.jpg", newimage_resize);
 
    cudaFree(pu8_dst_data_dev);
    cudaFree(pu8_rgb_dev);
    cudaFree(pu8_yuv_dev);
    free(pu8_rgb_host);
    return 0;
}

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <string>
#include <iostream>
#include <time.h>

using namespace std;

// 获取时间us
uint64_t getTimeUs(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) 
    {
        printf("clock_gettime failed\n");
        return 0;
    }

    long micros = ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    return micros;
}

// 加载内核文件
void loadProgramSource(const char **files, size_t length, char **buffer, size_t *sizes)
{
    /* Read each source file (*.cl) and store the contents into a temporary datastore */
    for (size_t i = 0; i < length; i++)
    {
        FILE *file = fopen(files[i], "r");
        if (file == NULL)
        {
            printf("couldn't read the program file: %s\n", files[i]);
            exit(-1);
        }
        fseek(file, 0, SEEK_END);
        sizes[i] = ftell(file);
        rewind(file); // reset the file pointer so that 'fread' reads from the front
        buffer[i] = (char *)malloc(sizes[i] + 1);
        buffer[i][sizes[i]] = '\0';
        fread(buffer[i], sizeof(char), sizes[i], file);
        fclose(file);
    }
}


int main(void)
{
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue cmd_queue;

    /* 加载图像 */
    std::string src_img_path = "../data/cat.jpeg";
    int src_img_w = 658;
    int src_img_h = 494;
    int src_img_c = 3;
    auto src_image = stbi_load(src_img_path.c_str(), &src_img_w, &src_img_h, &src_img_c, 0);
    if (nullptr == src_image)
    {
        printf("load source image failed: %s\n", src_img_path.c_str());
        return -1;
    }

    /* 申请输出图像内存 */
    void *dst_image = malloc(src_img_w * src_img_h);

    error = clGetPlatformIDs(1, &platform, NULL);                           // 获取平台id
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // 获取设备id
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);        // 创建上下文
    cmd_queue = clCreateCommandQueue(context, device, 0, &error);        // 创建命令队列

    const char *file_names[] = {"../rgb2gray.cl"}; // 待编译的内核文件
    const int NUMBER_OF_FILES = 1;
    char *buffer[NUMBER_OF_FILES];
    size_t sizes[NUMBER_OF_FILES];
    loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);                                       // 读取内核文件文本
    program = clCreateProgramWithSource(context, NUMBER_OF_FILES, (const char **)buffer, sizes, &error); // 创建program对象
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);                                       // 编译程序
    if (error != CL_SUCCESS)
    {
        // If there's an error whilst building the program, dump the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("\n=== ERROR ===\n\n%s\n=============\n", program_log);
        free(program_log);
        return -1;
    }

    error = clCreateKernelsInProgram(program, 1, &kernel, NULL); // 创建内核
    // 创建缓存对象
    cl_mem mem_src_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src_img_c * src_img_h * src_img_w, src_image, &error); // CL_MEM_COPY_HOST_PTR指定创建缓存对象后拷贝数据
    cl_mem mem_dst_img = clCreateBuffer(context, CL_MEM_WRITE_ONLY, src_img_h * src_img_w, NULL, &error);
    cl_mem mem_img_h = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &src_img_h, &error);
    cl_mem mem_img_w = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &src_img_w, &error);
    // 向内核函数传递参数
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_src_img);
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_dst_img);
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_img_h);
    error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &mem_img_w);

    size_t localThreads[2] = {32, 4}; // 工作组中工作项的排布
    // 计算方式：通常，你需要确保全局工作大小能够完全覆盖你的数据，并且适当地划分为局部工作组。
    // 例如，如果你的图像宽度是1024像素，而你选择的局部工作组大小是32，那么全局工作大小的宽度就应该是1024，这样可以刚好分成32个工作组（1024 / 32 = 32）。
    // (w + 31) / 32 * 32
    // (h + 3) / 4 * 4
    size_t globalThreads[2] = {((src_img_w + localThreads[0] - 1) / localThreads[0]) * localThreads[0],
                               ((src_img_h + localThreads[1] - 1) / localThreads[1]) * localThreads[1]}; // 确保全局工作组完全

    uint64_t start_time = getTimeUs();
    cl_event evt;
    error = clEnqueueNDRangeKernel(cmd_queue, kernel, // 启动内核
                                   2, 0, globalThreads, localThreads,
                                   0, NULL, &evt); // 内核执行完成后，会将evt置为CL_SUCCESS/CL_COMPLETE
    clWaitForEvents(1, &evt);                      // 等待命令事件发生
    clReleaseEvent(evt);
    // 读回数据
    error = clEnqueueReadBuffer(cmd_queue, mem_dst_img, CL_TRUE, 0, sizeof(unsigned char) * src_img_h * src_img_w, dst_image, 0, NULL, NULL);
    uint64_t end_time = getTimeUs();
    printf("cost: %d\n", end_time - start_time);

    // 释放资源
    clReleaseMemObject(mem_src_img);
    clReleaseMemObject(mem_dst_img);
    clReleaseMemObject(mem_img_h);
    clReleaseMemObject(mem_img_w);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);

    int success = stbi_write_png("../data/output.png", src_img_w, src_img_h, 1, dst_image, src_img_w * 1);
    if (success != 0)
    {
        printf("Successfully saved PNG image.\n");
    }
    else
    {
        printf("Failed to save PNG image.\n");
    }

    for (int i = 0; i < NUMBER_OF_FILES; i++)
        free(buffer[i]);

    free(dst_image);
    stbi_image_free(src_image);
    return 0;
}
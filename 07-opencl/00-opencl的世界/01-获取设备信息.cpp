// clang++ -std=c++11 -framework OpenCL ./00-demo.cpp -o demo

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
int main() 
{
    cl_int err;
    cl_uint num_platform;
    cl_platform_id platform_id;

    // 获取平台数量
    err = clGetPlatformIDs(0, NULL, &num_platform);
    std::cout << "num_platform: " << num_platform << std::endl;
    // 分配平台空间
    cl_platform_id platforms[num_platform];
    // 初始化可用平台
    err = clGetPlatformIDs(num_platform, platforms, NULL);
    platform_id = platforms[0];
    size_t size;
    // 获取平台信息
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &size);
    char *PName = (char *)malloc(size);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, size, PName, NULL);
    std::cout << "平台名字: " << PName << std::endl;
    // 获取平台供应商信息
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL, &size);
    char *PVendor = (char *)malloc(size);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, size, PVendor, NULL);
    std::cout << "平台供应商: " << PVendor << std::endl;

    // 获取平台版本信息
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, NULL, &size);
    char *PVersion = (char *)malloc(size);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, size, PVersion, NULL);
    std::cout << "平台版本信息: " << PVersion << std::endl;

    // 获取平台的配置或类型信息
    // FULL_PROFILE(普通版本): 这表示平台支持完整的 OpenCL 功能集，适用于大多数通用计算任务。
    // EMBEDDED_PROFILE(嵌入式版本): 这表示平台是一个简化的、针对嵌入式系统优化的版本，可能不支持所有的 OpenCL 功能，但更适合在资源受限的环境中运行。
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, 0, NULL, &size);
    char *PProfile = (char *)malloc(size);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, size, PProfile, NULL);
    std::cout << "平台的配置或类型信息: " << PProfile << std::endl;

    // 获取平台支持的扩展列表信息
    // 扩展名：OpenCL 的扩展通常由厂商提供，用于给设备增加新的功能。扩展的类型包括 Khronos OpenCL 工作组批准的扩展（如 cl_khr 开头的扩展）、外部扩展（如 cl_ext 开头的扩展）以及某个厂商自己的扩展（如 AMD 的特定扩展）。
    // 功能描述：每个扩展通常都有其特定的功能或特性，这些功能可能并不是 OpenCL 标准的一部分。例如，某些扩展可能提供了对双精度或 half 精度的支持，或者提供了原子操作等。
    // 版本信息：在某些情况下，平台支持的扩展可能还有其特定的版本信息。这些版本信息可以帮助开发者了解扩展的稳定性和与其他功能的兼容性。
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
    char *PExten = (char *)malloc(size);
    err = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, size, PExten, NULL);
    std::cout << "平台的配置或类型信息: " << PExten << std::endl;

    free(PName);
    free(PVendor);
    free(PVersion);
    free(PProfile);
    free(PExten);
}
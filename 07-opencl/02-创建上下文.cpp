// 上下文为关联的设备、内存对象、命令队列、程序对象、内核对象提供一个容器。上下文是opencl应用的核心

// clang++ -std=c++11 -framework OpenCL ./00-demo.cpp -o demo

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
using namespace std;

int main() 
{
    cl_int err;
    cl_uint num_device;
    cl_platform_id platform_id;
    cl_device_id *device;
    cl_platform_id platform;
    // 选择第一个平台
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    device = (cl_device_id *)malloc(sizeof(cl_device_id) * num_device);
    // 选择GPU设备
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_device, device, NULL);
    // 创建上下文
    cl_context_properties properities[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    // 指定设备创建上下文
    // clCreateContext函数把平台中查询到的所有GPU设备都关联到创建的上下文中
    // cl_context context = clCreateContext(properities, num_device, device, NULL, NULL, &err);

    // 指定设备类型创建上下文
    // clCreateContextFromType则是选择第一个平台中的GPU设备
    cl_context context = clCreateContextFromType(properities, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);

    // 获取上下文信息
    // 获取上文中设备数
    err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_device, NULL);
    std::cout << "上文中设备数: \n" << num_device << std::endl;

    // 获取上文引用计数
    cl_uint ref_count;
    err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, NULL);
    std::cout << "上文引用计数: \n" << ref_count << std::endl;

    // 获取赏析阿文设备列表
    size_t size;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    cl_device_id *context_devices = (cl_device_id*)malloc(size);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, context_devices, NULL);
    std::cout << "上文设备列表: \n" << context_devices[0] << std::endl;

    // 获取上下文properies参数
    size_t prop_size;
    err = clGetContextInfo(context, CL_CONTEXT_PROPERTIES, 0, NULL, &prop_size);
    cl_context_properties *properties = (cl_context_properties *)malloc(prop_size);
    err = clGetContextInfo(context, CL_CONTEXT_PROPERTIES, prop_size, properties, NULL);
    for (size_t i = 0; i < prop_size / sizeof(cl_context_properties); i += 3) 
    {
        printf("上下文properies参数: \nkey: 0x%lx, value: 0x%lx\n", properties[i], properties[i + 1]);
    }

    // clRetainContext(cl_context context)  增加引用计数(+1)
    // clReleaseContext(cl_context context) 减少引用计数(-1)
    clRetainContext(context);
    err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, NULL);
    std::cout << "上文引用计数: \n" << ref_count << std::endl;

    free(device);
    free(context_devices);
    free(properties);
}
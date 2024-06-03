// 在opencl上下文中, 有内存、程序和内核对象, 对这些对象的操作就需要使用命令队列。
// 一条名对就是主机发送给设备的一条信息, 用来钙素设备执行一个操作, 这个操作包含主机与设备间, 设备内的数据拷贝和内核执行。
// 命令提交到命令队列中, 命令队列吧需要执行的命令发送给设备。
// 每条命令只能关联一个设备，如果要同时勇士多个设备，则需要创建多个命令队列，每个命令对列关联到一个设备。
// 命令队列中的命令，只能是主机发送给设备，而设备不能发送命令给主机。
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

    // 创建命令队列
    cl_command_queue commandQueue;
    // commandQueue = clCreateCommandQueue(context, device[0], 0, &err);

    // 从OpenCL 2.0开始，推荐使用clCreateCommandQueueWithProperties函数来替代clCreateCommandQueue，因为它提供了更多的灵活性，允许直接传递属性结构或指针。
    // 启用乱序执行和性能分析
    cl_queue_properties_APPLE props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
    commandQueue = clCreateCommandQueueWithPropertiesAPPLE(context, device[0], &props, &err);

    // cl_int clRetainCommandQueue(cl_command_queue command_queue);  增加命令队列引用计数+1
    // cl_int clReleaseCommandQueue(cl_command_queue command_queue); 减少命令队列引用计数-1
}
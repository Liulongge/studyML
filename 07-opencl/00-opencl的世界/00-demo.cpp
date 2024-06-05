// clang++ -std=c++11 -framework OpenCL ./00-demo.cpp -o demo

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {
    // 1. 初始化OpenCL平台和设备
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    /* clGetPlatformIDs函数在OpenCL（Open Computing Language）中用于获取系统上可用的计算平台信息。
    函数原型：cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
    参数说明：
        num_entries：指定platforms数组中可以容纳的cl_platform_id表项的数目。如果platforms为NULL，则此参数应设为0。
        platforms：返回所找到的OpenCL平台清单。如果此参数为NULL，则忽略。
        num_platforms：返回实际可用的OpenCL平台数目。如果此参数为NULL，则忽略。
    */
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    /* clGetDeviceIDs 是 OpenCL API 中的一个函数，用于查询给定平台上可用的 OpenCL 设备，并返回它们的设备 ID。
    cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
    cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
    参数说明：
        platform：一个有效的平台标识符，通常通过 clGetPlatformIDs 函数获得。
        device_type：指定想要获取的设备类型，可以使用 CL_DEVICE_TYPE_CPU、CL_DEVICE_TYPE_GPU、CL_DEVICE_TYPE_ACCELERATOR、CL_DEVICE_TYPE_CUSTOM、CL_DEVICE_TYPE_ALL 等预定义的值。
        num_entries：指定 devices 数组中可以容纳的设备 ID 数量。如果 devices 为 NULL，则 num_entries 必须为 0。
        devices：指向一个 cl_device_id 数组的指针，用于存储返回的设备 ID。如果此参数为 NULL，则忽略。
        num_devices：一个可选参数，指向一个 cl_uint 变量的指针，用于存储实际返回的设备数量。如果此参数为 NULL，则忽略。
    */
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

    // 2. 创建OpenCL上下文
    /* cl_context clCreateContext(  
        const cl_context_properties *properties,  // 上下文属性列表  
        cl_uint                      num_devices,  // 参与上下文的设备数量  
        const cl_device_id          *devices,      // 参与上下文的设备列表  
        void (CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data),  // 错误通知函数  
        void                        *user_data,    // 用户自定义数据，传递给错误通知函数  
        cl_int                      *errcode_ret    // 返回的错误码  
        );
    */
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // 3. 创建命令队列
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // 4. 准备数据
    const int ARRAY_SIZE = 5;
    float A[ARRAY_SIZE] = {0, 1, 2, 3, 4};
    float B[ARRAY_SIZE] = {5, 6, 7, 8, 9};
    float C[ARRAY_SIZE];

    // 5. 创建缓冲区并复制数据到设备
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*ARRAY_SIZE, A, &ret);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*ARRAY_SIZE, B, &ret);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*ARRAY_SIZE, NULL, &ret);

    // 6. 加载并编译OpenCL内核代码
    const char *source_str = "__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {"
                            "   int i = get_global_id(0);"
                            "   c[i] = a[i] + b[i];"
                            "}";
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // 7. 创建内核对象
    cl_kernel kernel = clCreateKernel(program, "vec_add", &ret);

    // 8. 设置内核参数
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufferA);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufferB);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufferC);

    // 9. 执行内核
    size_t global_item_size = ARRAY_SIZE; // 全局工作项大小

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);

    // 10. 读取结果
    ret = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeof(float)*ARRAY_SIZE, C, 0, NULL, NULL);

    // 11. 显示结果
    for(int i = 0; i < ARRAY_SIZE; i++)
        printf("%f + %f = %f\n", A[i], B[i], C[i]);

    // 12. 清理资源
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
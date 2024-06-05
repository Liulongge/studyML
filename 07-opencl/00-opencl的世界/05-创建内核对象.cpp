// 一个程序对象就是内核的一个容器
// 在opencl中，使用cl_kernel数据结构来表示内核对象

// cl_kernel clCreateKernel(cl_program program,  
//                         const char *kernel_name,  
//                         cl_int *errcode_ret);
// 参数说明：
// program：一个有效的 OpenCL 程序对象。
// kernel_name：要创建的内核的名称（以空字符结尾的字符串）。
// errcode_ret：返回错误码。如果不需要错误码，可以传递 NULL。

// demo
// 假设已经有一个有效的 cl_context 和 cl_command_queue  
// cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);  
// 设置编译选项并编译程序  
// clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
// cl_int err;  
// cl_kernel kernel = clCreateKernel(program, "simple_kernel", &err);  


// clCreateKernelInProgram 函数用于在一个已编译和链接的OpenCL程序（program）中创建一个内核（kernel）对象。
// cl_kernel clCreateKernelInProgram(cl_program program,  
//                                   const char *kernel_name,  
//                                   cl_int *errcode_ret);
// 参数说明
// program：一个有效的OpenCL程序对象。
// kernel_name：要创建的内核的名称，以空字符结尾的字符串。
// errcode_ret：返回的错误码。如果不需要错误码，可以传递NULL。

// demo
// 加载并构建OpenCL程序  
// const char *program_source = "__kernel void sample_kernel() { /* ... */ }\n";  
// program = clCreateProgramWithSource(context, 1, (const char **)&program_source, NULL, &err);  
// err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);  
// 创建内核对象  
// kernel = clCreateKernelInProgram(program, "sample_kernel", &err);  



// 设置内核参数
// clSetKernelArg函数在OpenCL中用于为内核的特定参数设置参数值。
// cl_int clSetKernelArg(cl_kernel kernel, 
//                      cl_uint arg_index, 
//                      size_t arg_size, 
//                      const void *arg_value);
// 参数解释：
// cl_kernel kernel：一个有效的内核对象。
// cl_uint arg_index：参数索引。内核的参数由索引引用，最左边的参数索引为0，依次递增。
// size_t arg_size：参数值的大小（以字节为单位）。对于内存对象，如缓冲区或图像，这个值通常为sizeof(cl_mem)。
// const void *arg_value：指向数据的指针，该数据应用作arg_index指定的参数的参数值。

// demo
// cl_mem inputBuffer; // 假设你已经创建并初始化了这个缓冲区  
// cl_int err;  
// err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);  


// clSetKernelArgSVMPointer是OpenCL中用于设置内核参数为SVM（Shared Virtual Memory）指针的函数。
// SVM是OpenCL 2.0引入的一个特性，它允许主机和设备共享同一块内存区域，从而简化了数据传输过程并提高了效率。
// 这对于需要频繁数据交换的应用特别有用。
// cl_int clSetKernelArgSVMPointer(cl_kernel kernel,
//                                 cl_uint arg_index,
//                                 const void *arg_value);
// 参数说明
// kernel：一个有效的内核对象。
// arg_index：指定要设置的参数的索引，索引值从0开始。
// arg_value：一个指向SVM缓冲区或者指针的指针。这个参数实际上是SVM区域的起始地址。

// demo
// 创建SVM缓冲区
// float *data = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, 0);
// for(int i = 0; i < DATA_SIZE; ++i) {
//     data[i] = i;
// }

// // 创建并编译程序（这里假设已经有一个合适的内核源代码文件"svm_example.cl"）
// const char* source_str = "__kernel void svmAdd(__global float* svmData) { 
//     int gid = get_global_id(0);
//     svmData[gid] += gid; 
// }";
// size_t source_size = strlen(source_str);
// program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &err);
// err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
// kernel = clCreateKernel(program, "svmAdd", &err);
// // 设置内核参数
// err = clSetKernelArgSVMPointer(kernel, 0, data);


// clSetKernelExecInfo函数用于向内核对象传递执行时的信息，特别是当这些信息不是作为内核参数直接传递时。
// 它常用于指定那些内核将会访问但没有作为参数传递的SVM（Shared Virtual Memory）指针。
// 这是在OpenCL中利用SVM特性时非常关键的一部分，因为SVM允许主机和设备共享内存，从而提升数据交互的效率。
// cl_int clSetKernelExecInfo(cl_kernel           kernel,
//                          cl_kernel_exec_info param_name,
//                          size_t              param_value_size,
//                          const void *        param_value
// );
// 参数说明
// kernel：一个有效的内核对象。
// param_name：指定要设置的信息类型。对于SVM指针，通常使用CL_KERNEL_EXEC_INFO_SVM_PTRS。
// param_value_size：param_value参数的大小，以字节为单位。
// param_value：一个指向包含实际信息的数据的指针。当设置SVM指针时，这通常是一个指针数组，每个元素都是一个SVM指针。

// demo
// 分配SVM内存
// float *svmData1 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, 0);
// float *svmData2 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, 0);
// // 假设有一个内核需要这两个SVM指针，但它们不是直接作为内核参数传递
// const void* svm_ptrs[] = {svmData1, svmData2};
// size_t num_ptrs = sizeof(svm_ptrs) / sizeof(svm_ptrs[0]);
// // 创建并编译程序
// const char* source_str = "__kernel void svmProcess(__global float* svmData1, __global float* svmData2) { ... }";
// // 编译、创建内核等步骤（省略）
// // 使用clSetKernelExecInfo传递SVM指针
// err = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, num_ptrs * sizeof(void*), svm_ptrs);


// clGetKernelInfo 是 OpenCL 中的一个重要函数，用于查询内核对象的属性信息。
// cl_int clGetKernelInfo(cl_kernel     /* kernel */,  
//                      cl_kernel_info  /* param_name */,  
//                      size_t        /* param_value_size */,  
//                      void *        /* param_value */,  
//                      size_t *      /* param_value_size_ret */  
//                      );
// 参数说明
// kernel：要查询信息的内核对象。
// param_name：指定要查询的内核对象属性名称。例如，CL_KERNEL_FUNCTION_NAME 用于查询内核函数的名称。
// param_value_size：param_value 指向的内存块的大小（以字节为单位）。这个值必须大于或等于 param_name 对应的类型的大小。
// param_value：用于存储查询结果的内存位置的指针。如果此参数为 NULL，则忽略它。
// param_value_size_ret：返回实际写入 param_value 的字节数。如果 param_value 为 NULL，则此参数不会被使用。

// demo
// 假设 kernel 已经被正确创建和初始化  
// cl_kernel kernel; // 假设这是你的内核对象  
// // 查询内核函数的名称  
// size_t size = 0;  
// clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &size); // 查询需要的大小  
// char name[size + 1]; // 分配足够的内存来存储名称（包括终止符）  
// clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, size, name, NULL); // 查询名称  
// // 打印内核函数的名称  
// printf("Kernel function name: %s\n", name);   


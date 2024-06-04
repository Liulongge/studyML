// 程序对象包含多个在设备上执行的内核函数，是内核对象的集合。
// 在opencl中用cl_program类型表示程序对象。

// 1、创建程序对象
// 1.1、cl_program clCreateProgramWithSource(  
//     cl_context context,  
//     cl_uint count,  
//     const char **strings,  
//     const size_t *lengths,  
//     cl_int *errcode_ret);

// 参数说明
// context：一个有效的OpenCL上下文。程序对象将在该上下文中创建。
// count：表示strings和lengths数组中的元素数量。这通常是源代码字符串的数量。
// strings：一个指向字符串数组的指针，每个字符串包含一段OpenCL源代码。
// lengths：（可选）一个指向大小的数组的指针，表示strings中每个字符串的长度（以字节为单位）。如果此参数为NULL，则OpenCL会假定每个字符串都以null字符（'\0'）结尾，并计算到该字符的长度。
// errcode_ret：（可选）返回的错误码。如果此参数不为 NULL，并且没有错误发生，则指定的内存位置将被设置为 CL_SUCCESS。

// demo:
// cl_program program;  
// const char *program_source = "__kernel void hello(__global float *a) { ... }";  
// cl_int err;  
// program = clCreateProgramWithSource(context, 1, (const char **)&program_source, NULL, &err);  
// if (err != CL_SUCCESS) {  
//     // 错误处理  
// }


// 1.2、cl_program clCreateProgramWithBinary(  
//     cl_context           /* context */,  
//     cl_uint              /* num_devices */,  
//     const cl_device_id * /* device_list */,  
//     const size_t *       /* lengths */,  
//     const unsigned char **/* binaries */,  
//     cl_int *             /* binary_status */,  
//     cl_int *             /* errcode_ret */  
// );

// 参数说明
// context：执行程序的设备所在的上下文。
// num_devices：目标设备的数量。这指定了device_list和lengths数组中的元素数量。
// device_list：一个指向cl_device_id的指针数组，表示目标设备列表。这些设备将用于执行由该函数创建的程序。
// lengths：一个指向size_t的指针数组，表示每个二进制代码的长度。这些长度应该与binaries数组中的元素相对应。
// binaries：一个指向unsigned char指针的指针数组，表示预编译的二进制代码。这些二进制代码将被加载到OpenCL设备中执行。
// binary_status（可选）：一个指向cl_int的指针数组，用于返回每个二进制代码加载到设备上的状态。如果此参数为NULL，则不返回任何状态信息。
// errcode_ret（可选）：一个指向cl_int的指针，用于返回错误代码。如果函数执行成功，则返回CL_SUCCESS。

// demo:
// 假设你已经有了以下变量：  
// cl_platform_id platform_id;  
// cl_device_id device_id;  
// cl_context context;  
// cl_uint num_devices = 1; // 如果你有多个设备，这个值可能会不同  
// 假设 binary_data 是你的 OpenCL 二进制代码，binary_size 是其大小（以字节为单位）  
// const unsigned char* binary_data = ...; // 从文件或其他源加载二进制数据  
// size_t binary_size = ...; // 二进制数据的大小  
// cl_int err_code;  
// cl_program program = clCreateProgramWithBinary(context,  // OpenCL 上下文  
//                                                num_devices, // 设备数量  
//                                                &device_id, // 设备列表  
//                                                &binary_size, // 每个设备的二进制大小列表（可以是一个数组，但在这里我们只有一个设备）  
//                                                (const unsigned char**)&binary_data, // 二进制数据列表（同样，可以是一个数组）  
//                                                NULL, // 错误列表（如果非 NULL，将返回每个设备的编译错误）  
//                                                &err_code); // 错误码  

// 2、构建程序对象
// cl_int clBuildProgram(  
//     cl_program            program,  
//     cl_uint               num_devices,  
//     const cl_device_id *  device_list,  
//     const char *          options,  
//     void (CL_CALLBACK *   pfn_notify)(cl_program program, void * user_data),  
//     void *                user_data  
// );

// program：要编译的 OpenCL 程序对象。
// num_devices：device_list 中设备的数量。
// device_list：指向要编译程序的设备 ID 的指针数组。如果为 NULL，则对与 program 关联的所有设备编译程序。
// options：编译选项字符串，可以为 NULL。例如："-cl-std=CL1.2 -Werror" 表示使用 OpenCL 1.2 标准并启用所有警告作为错误。
// pfn_notify：编译完成后的回调函数。如果非 NULL，clBuildProgram 会立即返回，并在编译完成后调用此函数。如果为 NULL，则 clBuildProgram 会等待编译完成后再返回。
// user_data：传递给回调函数的用户数据指针。

// demo
// 假设已经有了以下变量：  
// cl_platform_id platform_id;  
// cl_device_id device_id;  
// cl_context context;  
// cl_program program; // 使用 clCreateProgramWithSource 创建的程序对象  
  
// 编译选项  
// const char *options = "-cl-std=CL1.2 -Werror";  
// cl_int err_code;  
// 编译程序  
// err_code = clBuildProgram(program, 1, &device_id, options, NULL, NULL);  

// 3、clLinkProgram
// clLinkProgram 是 OpenCL 1.2 及更高版本中引入的一个函数，它允许开发者将多个已编译的 OpenCL 程序对象（类似于 obj 文件）
// 链接成一个可执行程序。在 OpenCL 1.2 之前，开发者通常使用 clBuildProgram 来完成编译和链接的整个过程。从 OpenCL 1.2 开始，开发者可以将这两个过程分开，以便更细粒度地控制程序构建流程。
// cl_int clLinkProgram(  
//     cl_context            context,  
//     cl_uint               num_devices,  
//     const cl_device_id *  device_list,  
//     const char *          options,  
//     cl_uint               num_input_programs,  
//     const cl_program *    input_programs,  
//     void (CL_CALLBACK *   pfn_notify)(cl_program program, void * user_data),  
//     void *                user_data,  
//     cl_program *          errcode_ret_program  
// );

// context：一个有效的 OpenCL 上下文。
// num_devices：要链接程序的设备数量。
// device_list：指向要链接程序的设备 ID 的指针数组。如果为 NULL，则对所有与上下文关联的设备进行链接。
// options：链接选项字符串，可以为 NULL。这些选项与编译器选项类似，但特定于链接过程。
// num_input_programs：输入程序对象的数量。
// input_programs：指向要链接的已编译程序对象的指针数组。
// pfn_notify：链接完成后的回调函数。如果非 NULL，clLinkProgram 会立即返回，并在链接完成后调用此函数。如果为 NULL，则 clLinkProgram 会等待链接完成后再返回。
// user_data：传递给回调函数的用户数据指针。
// errcode_ret_program：如果 pfn_notify 为 NULL，并且链接失败，则此参数将返回一个包含错误代码的无效程序对象。否则，此参数将被忽略。

// demo
// 假设已经有了以下变量：  
// cl_context context;  
// cl_device_id device_id;  
// cl_program program1, program2; // 这两个程序对象已经通过 clCompileProgram 或其他方式编译  
// 链接选项（这里为示例，可能不需要）  
// const char *options = NULL;  
// cl_int err_code;  
// cl_program final_program = NULL;  
// 链接程序  
// err_code = clLinkProgram(context, 1, &device_id, options, 2, &program1, &program2, NULL, NULL, &final_program);  

// 4、clGetProgramInfo
// clGetProgramInfo 是 OpenCL 中的一个函数，用于查询 OpenCL 程序对象的信息。
// 这个函数允许你获取关于程序对象的各种属性，如源代码、二进制大小、编译状态等。
// cl_int clGetProgramInfo(  
//     cl_program            program,  
//     cl_program_info       param_name,  
//     size_t                param_value_size,  
//     void *                param_value,  
//     size_t *              param_value_size_ret  
// );

// program：要查询的 OpenCL 程序对象。
// param_name：要查询的程序对象的属性名称。例如，CL_PROGRAM_SOURCE、CL_PROGRAM_BINARY_SIZES、CL_PROGRAM_BUILD_STATUS 等。
// param_value_size：用于存储查询结果的缓冲区大小（以字节为单位）。
// param_value：指向存储查询结果的缓冲区的指针。
// param_value_size_ret：如果此参数不为 NULL，则返回实际查询结果的大小（以字节为单位）。这可以用于确定如果缓冲区太小而无法容纳整个结果时，需要多大的缓冲区。

// cl_program program; // 假设这是一个有效的 OpenCL 程序对象  
// 查询程序源代码  
// size_t source_size_ret;  
// cl_int err;  
// 首先，尝试获取源代码大小  
// err = clGetProgramInfo(program, CL_PROGRAM_SOURCE_SIZE, 0, NULL, &source_size_ret);  
// 分配足够的内存来存储源代码  
// char *source = (char *)malloc(source_size_ret);  
// 获取源代码  
// err = clGetProgramInfo(program, CL_PROGRAM_SOURCE, source_size_ret, source, NULL);  


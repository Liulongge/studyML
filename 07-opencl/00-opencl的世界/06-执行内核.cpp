// 利用命令队列使将在设备上执行的内核排队通过

// clEnqueueNDRangeKernel 是 OpenCL 中的一个核心函数，用于在设备上执行内核（kernel）。
// 这个函数的主要作用是将内核执行命令加入到命令队列中，以便在设备上异步执行。
// cl_int clEnqueueNDRangeKernel(  
//                 cl_command_queue  command_queue,  
//                 cl_kernel         kernel,  
//                 cl_uint           work_dim,  
//                 const size_t *    global_work_offset,  
//                 const size_t *    global_work_size,  
//                 const size_t *    local_work_size,  
//                 cl_uint           num_events_in_wait_list,  
//                 const cl_event *  event_wait_list,  
//                 cl_event *        event  
// );
// 参数详解
// command_queue：有效的命令队列。内核将排队等待在与该命令队列关联的设备上执行。
// kernel：有效的内核对象。与 kernel 和 command_queue 关联的 OpenCL 上下文必须相同。
// work_dim：用于指定全局工作项和工作组中工作项的维度数。work_dim 必须大于零且小于或等于 3。
// global_work_offset：当前必须为空值（NULL）。在未来的 OpenCL 版本中，它可用于指定计算工作项的全局 ID 的偏移量。
// global_work_size：指向一个 work_dim 无符号值数组，描述将执行内核函数的 work_dim 维度中的全局工作项的数量。
// local_work_size：指向一个 work_dim 无符号值数组，描述每个工作组中工作项的数量。如果为 NULL，则 OpenCL 实现将选择适当的工作组大小。
// num_events_in_wait_list 和 event_wait_list：这两个参数用于指定一个事件列表，这些事件必须在执行内核之前完成。如果不希望等待任何事件，可以将 num_events_in_wait_list 设置为 0，并将 event_wait_list 设置为 NULL。
// event：返回一个事件对象，该对象标识此特定的命令，并且可以用于查询命令的执行状态。如果不需要此事件对象，可以将其设置为 NULL。

// demo
// size_t global_work_size[2] = { 2048, 1024 }; // 全局工作项数量  
// size_t local_work_size[2] = { 16, 16 };     // 工作组大小  
  
// ... 假设 command_queue 和 kernel 已经被正确创建和初始化 ...  
// 执行内核  
// cl_int status = clEnqueueNDRangeKernel(  
//             command_queue,   
//             kernel,   
//             2,   
//             NULL,   
//             global_work_size,   
//             local_work_size,   
//             0,   
//             NULL,   
//             NULL  
//             );  

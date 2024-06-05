// OpenCL C编程语言用来编写在OPenCL计算设备上执行的内核程序
// OpenCL C基于ISO/IEC 9899:1999 C语言规范(C99)，并针对冰箱计算特性对语言做客一些限制和特定扩展

// OpenCL C是在C语言语法上进行的扩展，其中一个就是C语言中修饰符的扩展
// OpenCL中修饰符：地址限定符，函数限定符，存储类说明符(即static与extern)，对象访问限定符

// 1、地址空间修饰符
// OpenCL设备中有：
// 全局存储器：__global(或global)
// 局部存储器：__local(或local)
// 常量存储器：__constant(或constant)
// 私有存储器：__private(或private)，程序中的函数参数，函数中缺省地址修饰的局部变量，修饰符为private

// 2、函数修饰符
// kernel修饰符：__kernel(或kernel)
// 内核可选属性修饰符：__attribute__((xxx))
// 对象访问修饰符：__read_only(或read_only)
//              __write_only(write_only)
//              __read_write(read_write)

// 3、标量数据类型
// bool、char、uchar、short、ushort、int、uint、long、ulong、float、double、half、size_t、ptrdiff_t、intptr_t、uintptr_t、void

// 4、矢量数据类型
// charn：n个8位有符号整数值的矢量
// ucharn：n个8位无符号整数值的矢量
// shortn：n个16位有符号整数值的矢量
// ushortn：n个16位无符号整数值的矢量
// intn：n个32位有符号整数值的矢量
// uintn：n个32位无符号整数值的矢量
// longn：n个64位有符号整数值的矢量
// ulongn：n个64位无符号整数值的矢量
// floatn：n个32位浮点数值的矢量
// doublen：n个64位浮点数值的矢量

// 矢量初始化
// float4 DataVec = (float4){1.0, 1.0, 1.0, 1.0};
// 读取与修改：
//      数值索引：012345
//      字母索引：xyzw

// 5、运算符


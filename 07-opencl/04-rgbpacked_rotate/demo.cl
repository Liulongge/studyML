
// 逆时针90度旋转内核示例
// __kernel void rotate90(__global const unsigned char *input_image, 
//                        __global unsigned char *output_image,
//                        __global const unsigned int *p_height, 
//                        __global const unsigned int *p_width)
// {
//     int x = get_global_id(0);
//     int y = get_global_id(1);
//     int height = *p_height;
//     int width = *p_width;
//     int new_height = width;
//     int new_width = height;
//     if(x < width && y < height)
//     {
//         int index = y * width + x;
//         int new_x = y;
//         int new_y = width - 1 - x;
//         int new_index = new_y * new_width + new_x;
//         output_image[new_index * 3] = input_image[index * 3];
//         output_image[new_index * 3 + 1] = input_image[index * 3 + 1];
//         output_image[new_index * 3 + 2] = input_image[index * 3 + 2];
//     }
// }

// 顺时针90度旋转
// __kernel void rotate90(__global const unsigned char *input_image, 
//                        __global unsigned char *output_image,
//                        __global const unsigned int *p_height, 
//                        __global const unsigned int *p_width)
// {
//     int x = get_global_id(0);
//     int y = get_global_id(1);
//     int height = *p_height;
//     int width = *p_width;
//     int new_height = width;
//     int new_width = height;
//     if(x < width && y < height)
//     {
//         int index = y * width + x;
//         int new_y = x;
//         int new_x = height - 1 - y;
//         int new_index = new_y * new_width + new_x;
//         output_image[new_index * 3] = input_image[index * 3];
//         output_image[new_index * 3 + 1] = input_image[index * 3 + 1];
//         output_image[new_index * 3 + 2] = input_image[index * 3 + 2];
//     }
// }

// 180度旋转
__kernel void rotate180(__global const unsigned char *input_image, 
                       __global unsigned char *output_image,
                       __global const unsigned int *p_height, 
                       __global const unsigned int *p_width)
{
    printf("Thread %d: Data value = %d\n", *p_height, *p_width);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int height = *p_height;
    int width = *p_width;
    if(x < width && y < height)
    {
        int index = y * width + x;
        int new_y = height - y;
        int new_x = width - x;
        int new_index = new_y * width + new_x;
        output_image[new_index * 3] = input_image[index * 3];
        output_image[new_index * 3 + 1] = input_image[index * 3 + 1];
        output_image[new_index * 3 + 2] = input_image[index * 3 + 2];
    }
}
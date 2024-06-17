
__kernel void cropImage(__global unsigned char* input_image, 
                        __global unsigned char* output_image, 
                        __global const unsigned int *p_src_w,
                        __global const unsigned int *p_src_h, 
                        __global const unsigned int *p_dst_x,
                        __global const unsigned int *p_dst_y,
                        __global const unsigned int *p_dst_w,
                        __global const unsigned int *p_dst_h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int src_w = *p_src_w;
    int src_h = *p_src_h;
    int dst_x = *p_dst_x;
    int dst_y = *p_dst_y;
    int dst_w = *p_dst_w;
    int dst_h = *p_dst_h;

    if(x < src_w && y < src_h)
    {
        if((x >= dst_x) && (x < dst_x + dst_w) && (y >= dst_y) && (y < dst_y + dst_h))
        {
            // printf("x: %d, y: %d\n", x, y);
            int idx = y * src_w + x;
            int new_idx = (y - dst_y) * dst_w + (x - dst_x);
            output_image[new_idx * 3] = input_image[idx * 3];
            output_image[new_idx * 3 + 1] = input_image[idx * 3 + 1];
            output_image[new_idx * 3 + 2] = input_image[idx * 3 + 2];
        }

    }
}

__kernel void resizeImage(__global unsigned char* input_image, 
                        __global unsigned char* output_image, 
                        __global const unsigned int *p_src_w,
                        __global const unsigned int *p_src_h, 
                        __global const unsigned int *p_dst_w,
                        __global const unsigned int *p_dst_h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int src_w = *p_src_w;
    int src_h = *p_src_h;
    int dst_w = *p_dst_w;
    int dst_h = *p_dst_h;

    if(x < dst_w && y < dst_h)
    {
        float scale_w = (float)src_w / dst_w;
        float scale_h = (float)src_h / dst_h;
        int dst_idx = y * dst_w + x;
        int src_idx = (int)(y * scale_h * scale_w * dst_w + x * scale_w);
        // printf("sw %d, sh %d\n", dst_idx, src_idx);
        output_image[dst_idx * 3] = input_image[src_idx * 3];
        output_image[dst_idx * 3 + 1] = input_image[src_idx * 3 + 1];
        output_image[dst_idx * 3 + 2] = input_image[src_idx * 3 + 2];
    }
}
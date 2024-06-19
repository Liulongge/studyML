
__kernel void gaussian_filter(__global unsigned char* input_image, 
                              __global unsigned char* output_image, 
                              __global const unsigned int *p_src_w,
                              __global const unsigned int *p_src_h)
{
    int kernelWeights[9] = { 1, 2, 1,
                             2, 4, 2,
                             1, 2, 1 };

    int src_w = *p_src_w;
    int src_h = *p_src_h;
    int2 img_coord   = (int2) (get_global_id(0), get_global_id(1));
    // 无padding实现
    if ((img_coord.x > 0) && (img_coord.x < (src_w - 1))
       && (img_coord.y > 0) && img_coord.y < (src_h - 1))
    {
        int idx0 = (img_coord.y - 1) * src_w + img_coord.x -1;
        int idx1 = (img_coord.y - 1) * src_w + img_coord.x;
        int idx2 = (img_coord.y - 1) * src_w + img_coord.x + 1;

        int idx3 = img_coord.y * src_w + img_coord.x - 1;
        int idx4 = img_coord.y * src_w + img_coord.x;
        int idx5 = img_coord.y * src_w + img_coord.x + 1;

        int idx6 = (img_coord.y + 1) * src_w + img_coord.x - 1;
        int idx7 = (img_coord.y + 1) * src_w + img_coord.x;
        int idx8 = (img_coord.y + 1) * src_w + img_coord.x + 1;
        // x/16 = x >> 4
        output_image[idx4 * 3] =    (int)((input_image[idx0 * 3] * kernelWeights[0]
                                    + input_image[idx1 * 3] * kernelWeights[1]
                                    + input_image[idx2 * 3] * kernelWeights[2]
                                    + input_image[idx3 * 3] * kernelWeights[3]
                                    + input_image[idx4 * 3] * kernelWeights[4]
                                    + input_image[idx5 * 3] * kernelWeights[5]
                                    + input_image[idx6 * 3] * kernelWeights[6]
                                    + input_image[idx7 * 3] * kernelWeights[7]
                                    + input_image[idx8 * 3] * kernelWeights[8]) >> 4);

        output_image[idx4 * 3 + 1] =   (int)((input_image[idx0 * 3 + 1] * kernelWeights[0]
                                    + input_image[idx1 * 3 + 1] * kernelWeights[1]
                                    + input_image[idx2 * 3 + 1] * kernelWeights[2]
                                    + input_image[idx3 * 3 + 1] * kernelWeights[3]
                                    + input_image[idx4 * 3 + 1] * kernelWeights[4]
                                    + input_image[idx5 * 3 + 1] * kernelWeights[5]
                                    + input_image[idx6 * 3 + 1] * kernelWeights[6]
                                    + input_image[idx7 * 3 + 1] * kernelWeights[7]
                                    + input_image[idx8 * 3 + 1] * kernelWeights[8]) >> 4);

        output_image[idx4 * 3 + 2] =  (int)((input_image[idx0 * 3 + 2] * kernelWeights[0]
                                    + input_image[idx1 * 3 + 2] * kernelWeights[1]
                                    + input_image[idx2 * 3 + 2] * kernelWeights[2]
                                    + input_image[idx3 * 3 + 2] * kernelWeights[3]
                                    + input_image[idx4 * 3 + 2] * kernelWeights[4]
                                    + input_image[idx5 * 3 + 2] * kernelWeights[5]
                                    + input_image[idx6 * 3 + 2] * kernelWeights[6]
                                    + input_image[idx7 * 3 + 2] * kernelWeights[7]
                                    + input_image[idx8 * 3 + 2] * kernelWeights[8]) >> 4);

    }
}
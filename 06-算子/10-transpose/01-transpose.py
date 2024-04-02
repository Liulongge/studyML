import numpy as np  

input = np.array([[[[1, 2, 3, 4],
                   [4, 5, 6, 7],
                   [7, 8, 9, 10]],

                  [[1, 2, 3, 4],
                   [4, 5, 6, 7],
                   [7, 8, 9, 10]],

                   [[1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [7, 8, 9, 10]]]])
print("原始数据", input)


def transpose_wh(image):
    batch, channels, height, width = image.shape
    output_height = width
    output_width = height
    output_channel = channels

    output = np.zeros((1, output_channel, output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            for c in range(output_channel):
                output[0, c, h, w] = image[0, c, w, h]

    return output

output = transpose_wh(input)
print("python, 转置:\n", output)

# 使用transpose函数进行转置, 交换2, 3轴，其他轴不变
output = input.transpose(0, 1, 3, 2)
print("numpy, 转置:\n", output)

output = input.transpose(0, 1, 3, 2)
print("torch, 转置:\n", output)

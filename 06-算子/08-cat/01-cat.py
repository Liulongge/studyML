import numpy as np

# 创建几个多维数组
arry1 = np.array([[[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                 
                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]]])

arry2 = np.array([[[[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],
                 
                   [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]]])


import numpy as np

def cat(arry1, arry2, axis=0):
    if arry1.shape != arry2.shape:
        raise ValueError("数组形状不匹配，无法进行进一步操作")

    batch, channel, height, width = arry1.shape
    print(arry1.shape)
    new_batch = batch
    new_channel= channel
    new_height = height
    new_width = width
    if axis == 0: # batch 方向
        new_batch = 2 * batch
        output = np.zeros((new_batch, new_channel, new_height, new_width))
        output[0, :, :, :] = arry1[0, :, :, :]
        output[1, :, :, :] = arry2[0, :, :, :]
    elif axis == 1: # c方向
        new_channel = 2 * channel
        output = np.zeros((new_batch, new_channel, new_height, new_width))
        for c in range(new_channel):
            if c < channel:
                output[:, c, :, :] = arry1[0, c, :, :]
            else:
                output[:, c, :, :] = arry2[0, c - channel, :, :]
    elif axis == 2: # h方向
        new_height = 2 * height
        output = np.zeros((new_batch, new_channel, new_height, new_width))
        for h in range(new_height):
            if h < height:
                output[:, :, h, :] = arry1[:, :, h, :]
            else:
                output[:, :, h, :] = arry2[:, :, h - height, :]
    elif axis == 3:
        new_width = 2 * width
        output = np.zeros((new_batch, new_channel, new_height, new_width))
        for w in range(new_width):
            if w < width:
                output[:, :, :, w] = arry1[:, :, :, w]
            else:
                output[:, :, :, w] = arry2[:, :, :, w - width]
    else:
        raise ValueError("不支持的cat类型")

    return output


output = cat(arry1, arry2, axis=0)
print("python, n方向, cat:\n{}".format(output))
output = cat(arry1, arry2, axis=1)
print("python, c方向, cat:\n{}".format(output))
output = cat(arry1, arry2, axis=2)
print("python, h方向, cat:\n{}".format(output))
output = cat(arry1, arry2, axis=3)
print("python, w方向, cat:\n{}".format(output))

# 轴
# 0: n方向, 1: c方向, 2: h方向, 3: w方向
# 行方向, 堆叠数组
result = np.concatenate((arry1, arry2), axis=2)
print("numpy, 行方向cat:\n", result)
# [[[[1 2 3]
#    [4 5 6]
#    [7 8 9]
#    [1 1 1]
#    [1 1 1]
#    [1 1 1]]

#   [[1 2 3]
#    [4 5 6]
#    [7 8 9]
#    [1 1 1]
#    [1 1 1]
#    [1 1 1]]]]

# 列方向, 堆叠数组
result = np.concatenate((arry1, arry2), axis=3)
print("numpy, 列方向cat:\n", result)
# [[[[1 2 3 1 1 1]
#    [4 5 6 1 1 1]
#    [7 8 9 1 1 1]]

#   [[1 2 3 1 1 1]
#    [4 5 6 1 1 1]
#    [7 8 9 1 1 1]]]]

from torch import tensor
import torch

tensor1 = tensor(arry1)
tensor2 = tensor(arry2)
# c方向, 将它们连接在一起
result = torch.cat((tensor1, tensor2), dim=1)
print("torch, c方向cat:\n", result)
# tensor([[[[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]],

#          [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]],

#          [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]],

#          [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]]])

result = torch.cat((tensor1, tensor2), dim=2)
print("torch, h方向cat:\n", result)
# tensor([[[[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9],
#           [1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]],

#          [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9],
#           [1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]]])

result = torch.cat((tensor1, tensor2), dim=3)
print("torch, w方向cat:\n", result)
# tensor([[[[1, 2, 3, 1, 1, 1],
#           [4, 5, 6, 1, 1, 1],
#           [7, 8, 9, 1, 1, 1]],

#          [[1, 2, 3, 1, 1, 1],
#           [4, 5, 6, 1, 1, 1],
#           [7, 8, 9, 1, 1, 1]]]])
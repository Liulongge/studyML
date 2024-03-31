import numpy as np

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

def add(arry1, arry2):
    if arry1.shape != arry2.shape:
        raise ValueError("数组形状不匹配，无法进行进一步操作")

    batch, channel, height, width = arry1.shape
    output = np.zeros((batch, channel, height, width))

    for h in range(height):
        for w in range(width):
            for c in range(channel):
                for n in range(batch):
                    # 获取图像信息, n~n+3行, n~n+3列, 全部channel
                    output[n, c, h, w] = arry1[n, c, h, w] + arry2[n, c, h, w]

    return output

output = add(arry1, arry2)

print("python\n{}".format(output))




import numpy as np

# 使用numpy.add()函数进行相加
result = np.add(arry1, arry2)
print("numpy, 使用numpy.add():\n", result)
print("numpy, 直接使用加号操作符:\n", arry1 + arry2)
preallocated = np.zeros((1, 2, 3, 3))
np.add(arry1, arry2, out=preallocated)
print("numpy, 支持out参数, 用于将结果写入预先分配的数组中:\n", preallocated)

import torch
from torch import tensor
tensor1 = tensor(arry1)
tensor2 = tensor(arry2)

# 使用加法运算符
result = tensor1 + tensor2
print("torch, 使用加法运算符:\n", result)

# 使用torch.add()函数
result = torch.add(tensor1, tensor2)
print("torch, 使用torch.add()函数:\n", result)

a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# 原地加法，直接修改a的内容
tensor2.add_(tensor1)
print("torch, 原地加法:\n", tensor2)

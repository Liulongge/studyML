import numpy as np
 
def conv2d(image, kernel):
    # 获取图像和内核的尺寸
    h, w = image.shape
    kh, kw = kernel.shape
    
    # 补零处理，以便内核可以在图像边缘之外“滑动”
    image_pad = np.pad(image, (kh//2, kw//2), 'constant')
    
    # 初始化输出数组
    output = np.zeros((h, w))
    
    # 卷积操作
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = image_pad[y:y+kh, x:x+kw]
            
            # 执行内核与窗口的点积（元素乘积的和）
            output[y, x] = np.sum(window * kernel)
    
    return output
 
# 示例用法
image = np.array([[1, 2, 0],
                  [3, 4, 5],
                  [6, 0, 7]])
kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
 
result = conv2d(image, kernel)
print("input:\n{}".format(image))
print("kernel:\n{}".format(kernel))
print("output:\n{}".format(result))



import torch
import torch.nn as nn

# 创建一个1x1x3x3的输入数据
input_data = torch.tensor([[[[1, 2, 0],
                             [3, 4, 5],
                             [6, 0, 7]]]], dtype=torch.float32)

# 定义一个1x1x3x3的卷积核
kernel = torch.tensor([[[[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]]]], dtype=torch.float32)

# 定义卷积层
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

# 将定义好的卷积核赋值给卷积层的权重
conv_layer.weight.data = kernel

# 进行卷积运算
output_data = conv_layer(input_data)

print("input:\n{}".format(input_data))
print("kernel:\n{}".format(kernel))
print("output:\n{}".format(output_data.data))

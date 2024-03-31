# ReLu(Rectified Linear Unit)

# 线性整流函数(ReLU函数)可以用如下形式表示：
# f(x) = max(0, x)
# 其中x为输入值，f(x)为ReLU函数的输出值。可以看出，当输入值大于等于0时，ReLU函数的输出为输入值本身；而当输入值小于0时，ReLU函数

# 1、python 实现
import numpy as np
 
def relu6(x):  
    return np.minimum(np.maximum(0, x), 6)
  
# 示例使用
x = np.array([-1, 2, -3, 9])
y = relu6(x)
print(y)
# 输出将会是 [0 2 0 6]


# 2、pytorch relu
import torch
import torch.nn as nn
from torch import tensor

# 创建一个ReLU6实例
relu6 = nn.ReLU6(inplace=False)  # inplace=True 会直接在原张量上修改值，而不是创建新张量

# 假设我们有一个张量
input_tensor = tensor([[-0.7143,  0.9522,  1.8683],
                       [-1.1039,  10.5947, -10.4799]])

# 将ReLU6应用于该张量
output_tensor = relu6(input_tensor)
print(output_tensor)
# 输出张量的值现在已经被ReLU6函数约束在[0, 6]区间内
# tensor([[0.0000, 0.9522, 1.8683],
        # [0.0000, 6.0000, 0.0000]])

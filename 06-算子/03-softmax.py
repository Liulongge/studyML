
# 参考:
# https://blog.csdn.net/qq_32642107/article/details/97270994?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171155630416800222823014%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171155630416800222823014&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-97270994-null-null.142^v100^pc_search_result_base7&utm_term=softmax&spm=1018.2226.3001.4187
# https://blog.csdn.net/weixin_40662331/article/details/80648290?ops_request_misc=&request_id=&biz_id=102&utm_term=softmax%E8%AF%A6%E8%A7%A3&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-80648290.142^v100^pc_search_result_base7&spm=1018.2226.3001.4187

# Softmax函数是一种在机器学习和深度学习中广泛使用的激活函数，
# 尤其在处理多类别分类问题时，它常被用作神经网络的最后一层输出层。
# Softmax函数能够将一个包含任意实数值的向量（通常是神经网络的输出）转换为一个概率分布，
# 即所有输出的值都介于0和1之间，并且这些输出值的总和为1。

# 实际应用中，使用 Softmax 需要注意数值溢出的问题。
# 因为有指数运算，如果 V 数值很大，经过指数运算后的数值往往可能有溢出的可能。
# 所以，需要对 V 进行一些数值处理：即 V 中的每个元素减去 V 中的最大值。


import numpy as np

# numpy实现
def softmax(x):
    # 首先计算e的各个元素次方
    e_x = np.exp(x - np.max(x))
    # 然后归一化，求和后取倒数，确保结果是一个概率分布
    return e_x / e_x.sum(axis=-1)


# 假设我们有一个向量
input = np.array([1.5, 2.2, 3.1, 0.9, 1.2, 1.7])
# 应用softmax函数
output = softmax(input)
# 输出概率分布
print("输入:\n", input)
print("输出:\n", output)


# pytorch 实现
from torch import nn
import torch 
softmax_layer = nn.Softmax(dim=-1)
output = softmax_layer(torch.tensor(input))
print("输出:\n", output)
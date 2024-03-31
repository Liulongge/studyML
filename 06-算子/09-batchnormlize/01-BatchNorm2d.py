import numpy as np  
  
def batch_norm(x, gamma, beta, eps=1e-5):  
    """  
    纯Python实现Batch Normalization。  
      
    参数:  
        x: 输入数据，形状为 (batch_size, num_features)  
        gamma: 可学习的缩放参数，形状为 (num_features,)  
        beta: 可学习的偏移参数，形状为 (num_features,)  
        eps: 用于防止除以零的小常数  
          
    返回:  
        y: 经过Batch Normalization处理后的数据，形状与x相同  
    """  
    # 计算mini-batch的均值和方差  
    mean = np.mean(x, axis=0)  
    var = np.var(x, axis=0)  
      
    # 对每个元素进行标准化  
    x_norm = (x - mean) / np.sqrt(var + eps)  
      
    # 应用可学习的缩放和偏移参数  
    y = gamma * x_norm + beta  
      
    return y  
  
# 示例用法  
batch_size = 2  
num_features = 3  
x = np.random.randn(batch_size, num_features)  # 随机生成输入数据  
gamma = np.random.randn(num_features)  # 初始化缩放参数  
beta = np.random.randn(num_features)  # 初始化偏移参数  
  
y = batch_norm(x, gamma, beta)  # 对输入数据进行Batch Normalization处理  
print(y)



import torch  
import torch.nn as nn  
  
# 初始化一个BatchNorm2d层，假设输入有3个通道（比如RGB图像）  
batch_norm = nn.BatchNorm2d(num_features=3)  

# 创建一个随机的输入tensor，形状为(batch_size, num_channels, height, width)  
# 这里我们假设batch_size为4，每个图像有3个通道（RGB），图像大小为32x32  
batch_size = 1
num_channels = 3
height = width = 3
input_tensor = torch.randn(batch_size, num_channels, height, width)  

# 将模型置于评估模式（对于推理）或训练模式（对于训练）  
# BatchNorm层在训练时会更新运行均值和方差，而在评估模式时会使用这些运行统计量  
batch_norm.eval()  # 使用评估模式  
  
# 通过BatchNorm层进行前向传播  
output_tensor = batch_norm(input_tensor)  

# 打印输出tensor的形状和值  
print("put tensor:\n", input_tensor)  
print("output tensor:\n", output_tensor)
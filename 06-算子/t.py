import numpy as np  
  
def max_pooling_2d(input_data, pool_size=(2, 2), stride=2):  
    """  
    2D 最大池化函数  
      
    参数:  
    input_data: 输入的numpy数组，形状应为 (height, width, channels)  
    pool_size: 池化窗口的大小，形如 (height, width)  
    stride: 步长，池化窗口移动的步长  
      
    返回:  
    池化后的numpy数组  
    """  
    # 获取输入数据的维度  
    height, width, channels = input_data.shape  
      
    # 计算输出数据的维度  
    out_height = (height - pool_size[0]) // stride + 1  
    out_width = (width - pool_size[1]) // stride + 1  
      
    # 初始化输出数组  
    pooled_output = np.zeros((out_height, out_width, channels))  
      
    # 遍历每个通道  
    for c in range(channels):  
        # 提取当前通道的数据  
        channel_data = input_data[:, :, c]  
          
        # 对当前通道的数据进行池化  
        for i in range(0, height, stride):  
            for j in range(0, width, stride):  
                # 提取当前池化窗口的数据  
                pool_region = channel_data[i:i+pool_size[0], j:j+pool_size[1]]  
                  
                # 计算当前池化窗口的最大值  
                max_val = np.max(pool_region)  
                  
                # 将最大值放入输出数组的相应位置  
                pooled_output[i//stride, j//stride, c] = max_val  
      
    return pooled_output  
  
# 示例输入数据，形状为 (3, 4, 3)，即高为3，宽为4，通道数为3  
input_data = np.random.rand(3, 4, 3)  
  
# 执行最大池化，假设池化窗口大小为 (2, 2)，步长为 2  
pooled_data = max_pooling_2d(input_data, pool_size=(2, 2), stride=2)  
  
print(pooled_data.shape)  # 输出应为 (1, 2, 3)
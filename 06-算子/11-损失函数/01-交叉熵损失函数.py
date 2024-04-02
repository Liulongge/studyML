import numpy as np

# def categorical_cross_entropy_loss(y_true, y_pred):
#     # 确保预测值在(0, 1)之间，并且所有的预测概率之和为1（softmax输出）
#     y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
#     y_pred /= y_pred.sum(axis=-1, keepdims=True)

#     # 检查y_true是否已经是one-hot编码格式，如果不是则转化为one-hot编码
#     if len(y_true.shape) == 1:
#         n_classes = y_pred.shape[-1]
#         y_true = np.eye(n_classes)[y_true]

#     # 计算交叉熵损失
#     loss = -np.sum(y_true * np.log(y_pred), axis=-1)

#     # 平均损失
#     mean_loss = np.mean(loss)

#     return mean_loss

def cross_entropy_loss(y_pred, y_true):  
    softmax_logits = softmax(y_pred)  
    loss = -np.log(softmax_logits[range(y_true.shape[0]), y_true])  
    return np.mean(loss)  
  
def softmax(x):  
    """Compute softmax values for each sets of scores in x."""  
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=1, keepdims=True)  

# 示例：
y_true = np.array([0, 1, 2, 0])  # 真实标签（假设共有3个类别）
y_pred = np.array([
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5],
    [0.6, 0.1, 0.3],
    [0.9, 0.05, 0.05]
])  # 预测概率向量

print("numpy, 交叉熵损失函数:\n", cross_entropy_loss(y_pred, y_true))


import torch  
import torch.nn as nn 
from torch import tensor

# 初始化交叉熵损失函数  
criterion = nn.CrossEntropyLoss()  

# 计算损失  
loss = criterion(tensor(y_pred), tensor(y_true))  
print("torch, 交叉熵损失函数:\n", loss)  
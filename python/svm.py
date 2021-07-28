import cv2
import numpy as np
import matplotlib.pyplot as plt
#1 准备data
rand1 = np.array([[155,48],[159,50],[164,53],[168,56],[172,60]])
# 女生身高和体重数据
rand2 = np.array([[152,53],[156,55],[160,56],[172,64],[176,65]])
# 男生身高和体重数据

# 2 建立分组标签，0代表女生，1代表男生
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

# 3 合并数据
data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')

# 4 训练
# ml  机器学习模块 SVM_create() 创建
svm = cv2.ml.SVM_create() 

# 属性设置
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) # line
svm.setC(0.01)

# 训练
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)

# 预测
pt_data = np.vstack([[167,55],[162,57]]) #0 女生 1男生
pt_data = np.array(pt_data,dtype='float32')
print(pt_data)
(par1,par2) = svm.predict(pt_data)
print(par2)

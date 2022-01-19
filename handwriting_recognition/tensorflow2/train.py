import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

print(tf.__version__)

##加载数据     60000条训练集   10000条测试集  
(x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()  #此处会去官网加载数据，可能比较慢
print(type(x_train_all))

#print((x_train.shape),(x_test.shape))   #(60000, 28, 28) (10000, 28, 28)
#数据归一化
scaler = StandardScaler()
scaled_x_train_all = scaler.fit_transform(x_train_all.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
scaled_x_test = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)


#划分验证集和训练集
scaled_x_train,scaled_x_valid = scaled_x_train_all[5000:],scaled_x_train_all[:5000]
y_train,y_valid = y_train_all[5000:],y_train_all[:5000]


#采用函数式API
a = Input(shape=(784,))   #单条数据维度，不包括数据总数
b = Dense(10,activation='softmax')(a)
model = Model(a,b)
model.summary()


model.compile(loss="sparse_categorical_crossentropy",optimizer = "sgd",metrics=["accuracy"])
model.fit(scaled_x_train.reshape(-1,784),y_train,epochs=20,validation_data=(scaled_x_valid.reshape(-1,784),y_valid))

#测试集评估
model.evaluate(scaled_x_test.reshape(-1,784),y_test,verbose=0)  #verbose是否打印相关信息

#随机选中图片测试
img_random = scaled_x_test[np.random.randint(0,len(scaled_x_test))]
import matplotlib.pyplot as plt
# %matplotlib inline

plt.imshow(img_random)
plt.show()
plt.savefig('test.jpg')

#模型预测
prob = model.predict(img_random.reshape(-1,784))
print(np.argmax(prob))

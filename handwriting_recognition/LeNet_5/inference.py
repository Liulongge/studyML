import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入数据的模块

mnist=input_data.read_data_sets('mnist_data',one_hot=True)
train_X,train_Y,test_X,test_Y=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

test_image=mnist.test.images[101]  #选取第101张图片
img_size=28
test_image=test_image.reshape(-1,img_size,img_size,1)

with tf.Session() as sess:   #    
    new_saver=tf.train.import_meta_graph('mnist_cnn_model/medel.ckpt.meta') #第二步：导入模型的图结构
    new_saver.restore(sess,'mnist_cnn_model/medel.ckpt')  #第三步：将这个会话绑定到导入的图中
    #new_saver.restore(sess,tf.train.latest_checkpoint('mymodel'))    #第三步也可以是这样操作，因为会从mymodel文件夹中获取checkpoint，而checkpoint中存储了最新存档的文件路径

    print('=======================模型加载完成=============================')
    X=sess.graph.get_tensor_by_name('input:0')                 #从模型中获取输入的那个节点
    p_keep=sess.graph.get_tensor_by_name('p_keep_rate:0')
    model_y=sess.graph.get_tensor_by_name('output_layer/output:0')
    print('=======================输入输出加载完成=============================')
    result=sess.run(model_y,feed_dict={X:test_image,p_keep:1})  #需要的就是模型预测值model_Y，这里存为result
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(result)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    soft_result=tf.nn.softmax(result)
    result1=sess.run(soft_result)
    print(result1)
    
    plt.imshow(test_image.reshape([28,28]),cmap='Greys')
    plt.show()
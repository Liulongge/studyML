import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #导入数据的模块
  
batch_size=100  #表示每一批训练100组数据，因为训练集共有数据55000组，故而训练一个周期需要经过550次迭代
test_size=256   #作为验证数据，验证集有10000组数据，但这里只验证256组，因为数据太多，运算太慢
img_size=28     #手写字图像的大小
num_class=10    #图像的类别

X=tf.placeholder(dtype=tf.float32,shape=[None,img_size,img_size,1],name='input')
Y=tf.placeholder(dtype=tf.float32,shape=[None,num_class])

p_keep=tf.placeholder(tf.float32,name='p_keep_rate') #后面用到dropout层的保留的参数

mnist=input_data.read_data_sets('mnist_data',one_hot=True) #导入数据集
train_X,train_Y,test_X,test_Y=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

train_X=train_X.reshape(-1,img_size,img_size,1)   #训练数据和测试数据都需要重塑一下形状，因为导入的是724长度的
test_X=test_X.reshape(-1,img_size,img_size,1)

#第一个卷积层
with tf.name_scope('cnn_layer_01') as cnn_01:     
    w1=tf.Variable(tf.random_normal(shape=[3,3,1,32],stddev=0.01))
    conv1=tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding="SAME")
    conv_y1=tf.nn.relu(conv1)

#第一个池化层
with tf.name_scope('pool_layer_01') as pool_01:
    pool_y2=tf.nn.max_pool(conv_y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    pool_y2=tf.nn.dropout(pool_y2,p_keep)

#第二个卷积层
with tf.name_scope('cnn_layer_02') as cnn_02:
    w2=tf.Variable(tf.random_normal(shape=[3,3,32,64],stddev=0.01))
    conv2=tf.nn.conv2d(pool_y2,w2,strides=[1,1,1,1],padding="SAME")
    conv_y3=tf.nn.relu(conv2)

#第二个池化层
with tf.name_scope('pool_layer_02') as pool_02:
    pool_y4=tf.nn.max_pool(conv_y3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    pool_y4=tf.nn.dropout(pool_y4,p_keep)

#第三个卷积层
with tf.name_scope('cnn_layer_03') as cnn_03:
    w3=tf.Variable(tf.random_normal(shape=[3,3,64,128],stddev=0.01))
    conv3=tf.nn.conv2d(pool_y4,w3,strides=[1,1,1,1],padding="SAME")
    conv_y5=tf.nn.relu(conv3)

#第三个池化层
with tf.name_scope('pool_layer_03') as pool_03:
    pool_y6=tf.nn.max_pool(conv_y5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#全连接层 
with tf.name_scope('full_layer_01') as full_01:
    w4=tf.Variable(tf.random_normal(shape=[128*4*4,625],stddev=0.01))
    FC_layer=tf.reshape(pool_y6,[-1,w4.get_shape().as_list()[0]])
    FC_layer=tf.nn.dropout(FC_layer,p_keep)
    FC_y7=tf.matmul(FC_layer,w4)
    FC_y7=tf.nn.relu(FC_y7)
    FC_y7=tf.nn.dropout(FC_y7,p_keep)
 
 #输出层，model_Y则为神经网络的预测输出
with tf.name_scope('output_layer') as output_layer:
    w5=tf.Variable(tf.random_normal(shape=[625,num_class]))
    model_Y=tf.matmul(FC_y7,w5,name='output')

#损失函数
Y_=tf.nn.softmax_cross_entropy_with_logits(logits=model_Y,labels=Y)
cost=tf.reduce_mean(Y_)

#准确率
correct_prediction=tf.equal(tf.argmax(model_Y,axis=1),tf.argmax(Y,axis=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#优化方式
optimizer=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)

#将相关的参数写入tensorboard
#-----------------------------------------------------------------------
tf.summary.scalar('loss',cost)
tf.summary.scalar('accuracy',accuracy)
tf.summary.histogram('w1',w1)
tf.summary.histogram('w2',w2)
tf.summary.histogram('w3',w3)
tf.summary.histogram('w4',w4)
tf.summary.histogram('w5',w5)
merge=tf.summary.merge_all()
#--------------------------------------------------------------------------
#构建会话任务
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer=tf.summary.FileWriter('mnist_cnn_summary_01',graph=sess.graph)
    #将索引重组一下，得到【（0,100），（100,200），（200,300）。。。。】的形式
    training_batch=zip(range(0,len(train_X),batch_size),range(batch_size,len(train_X)+1,batch_size))

    run_metadata=tf.RunMetadata()
    #开始训练，此处只训练一个epoch，若想训练多个epoch，可以再添加一个循环
    for start,end in training_batch:
        opti,summary,loss,acc=sess.run([optimizer,merge,cost,accuracy],\
        feed_dict={X:train_X[start:end],Y:train_Y[start:end],p_keep:0.8},\
        run_metadata=run_metadata) #这是最关键的一步，optimizer,summary,loss,accuracy均放在一起进行训练

        writer.add_run_metadata(run_metadata,tag='step{0}'.format(start),global_step=(start/batch_size)+1)
        writer.add_summary(summary,global_step=(start/batch_size)+1)

        print(f'第 {(start/batch_size)+1} 次迭代时，准确度为 {acc},误差为 {loss}',end='\r',flush=True)

    print('===================下面开始进行测试==============================')
    #下面开始对测试集上的数据进行验证
    test_index=np.arange(len(test_X))
    np.random.shuffle(test_index)      #将测试集上的10000组数据顺序随即大乱
    test_index=test_index[0:test_size] #只选择打乱顺序之后的256组样本数据进行测试
    test_acc=sess.run(accuracy,feed_dict={X:test_X[test_index],Y:test_Y[test_index],p_keep:1})
    print(f'测试集上面的准确率为 ：{test_acc}')
    print('===================下面是保存模型================================')
    #保存模型
    saver=tf.train.Saver()
    path=saver.save(sess,'mnist_cnn_model/medel.ckpt')
    print(f'模型欧保存到 {path}')
    print('训练完毕！')
import tensorflow as tf
import timeit

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000, 1000])
    cpu_b = tf.random.normal([1000, 2000])
    print("\033[31m cpu device: \033[0m")
    print(cpu_a.device, cpu_b.device)
# 创建使用GPU运算的2个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    print("\033[31m gpu device: \033[0m")
    print(gpu_a.device, gpu_b.device)

def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c 

def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('\033[31m 设备启动时间:\033[0m')
print('\033[31m warmup:\033[0m', cpu_time, gpu_time)
# 正式计算10次，取平均时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('\033[31m run time:\033[0m', cpu_time, gpu_time)
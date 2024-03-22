import numpy as np
 
def conv2d(image, kernel):
    # 获取图像和内核的尺寸
    h, w = image.shape
    kh, kw = kernel.shape
    
    # 补零处理，以便内核可以在图像边缘之外“滑动”
    image_pad = np.pad(image, (kh//2, kw//2), 'constant')
    
    # 初始化输出数组
    output = np.zeros((h, w))
    
    # 卷积操作
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = image_pad[y:y+kh, x:x+kw]
            
            # 执行内核与窗口的点积（元素乘积的和）
            output[y, x] = np.sum(window * kernel)
    
    return output
 
# 示例用法
image = np.array([[1, 2, 0],
                  [3, 4, 5],
                  [6, 0, 7]])
kernel = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
 
result = conv2d(image, kernel)
print(result)
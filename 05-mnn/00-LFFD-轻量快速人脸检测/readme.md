## 参考
    1. 参考工程: [text](https://github.com/SyGoing/LFFD-MNN/tree/master?tab=readme-ov-file)
    2. 原始工程: [text](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices)
    3. paper: [text](https://arxiv.org/pdf/1904.10633.pdf)

## 编译
    1. 编译mnn生成动态库
    2. 将动态库拷贝至/MNN/lib目录下
    3. $ cmake .. && make -j8

## 运行
### 使用camera
    $ ./test ../models/symbol_10_320_20L_5scales_v2_deploy.mnn
    $ ./test ../models/
### 使用图像
    $ ./test ../models/symbol_10_320_20L_5scales_v2_deploy.mnn ../data/test_5.jpg
    $ ./test ../models/ ../data/selfie.jpg
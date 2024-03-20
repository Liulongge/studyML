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
    $ ./test --run_mode=offline --model_name ../models/symbol_10_560_25L_8scales_v1_deploy.mnn --scale_num=8
    $ ./test --run_mode=online --model_name ../models/symbol_10_320_20L_5scales_v2_deploy.mnn --scale_num=5
### 使用图像
    $ ./test --run_mode=offline --model_name ../models/symbol_10_560_25L_8scales_v1_deploy.mnn --scale_num=8
    $ ./test --run_mode=offline --model_name ../models/symbol_10_320_20L_5scales_v2_deploy.mnn --scale_num=5
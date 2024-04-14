
## 模型转换
### 编译转换工具
    $ cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TORCH=ON
    参数:
        MNN_BUILD_CONVERTER 是否编译模型转换工具
        MNN_BUILD_TORCH 是否支持TorchScript模型转换，MacOS下需要安装pytorch，Linux下会下载libtorch
    产物:
        MNNConvert 模型转换工具
        TestConvertResult 模型转换正确性测试工具，Windows下没有此产物，用MNNConvert对应功能替代
        TestPassManager 模型转换工具测试用例
        MNNDump2Json 模型转换为Json
        MNNRevert2Buffer Json转换为模型
        OnnxClip Onnx模型裁剪工具
### 使用
    $ ./MNNConvert -f CAFFE --modelFile XXX.caffemodel --prototxt XXX.prototxt --MNNModel XXX.mnn --bizCode biz
    demo: 
        $ ./MNNConvert -f CAFFE --modelFile ../../studyML/05-mnn/train_10_560_25L_8scales_v1_iter_1400000.caffemodel --prototxt ../../studyML/05-mnn/symbol_10_560_25L_8scales_v1_deploy.prototxt --MNNModel ../../studyML/05-mnn/symbol_10_560_25L_8scales_v1_deploy.mnn --bizCode biz
        $ ./MNNConvert -f ONNX --modelFile ../../../models/my_model.onnx  --MNNModel ../../../models/my_model.mnn --bizCode biz
## 编译benchmark工具
#### Linux / macOS / Ubuntu
    $ cmake .. -DMNN_BUILD_BENCHMARK=ON
### 使用
    $ ./benchmark.out models_folder [loop_count] [warmup] [forwardtype] [numberThread] [precision] [weightSparsity] [testQuantizedModel]
    参数如下:
        models_folder: benchmark models文件夹，benchmark models。
        loop_count: 可选，默认是10
        warm_up_count: 预热次数
        forwardtype: 可选，默认是0，即CPU，forwardtype有0->CPU，1->Metal，3->OpenCL，6->OpenGL，7->Vulkan
        numberThread: 可选，默认是4，为 CPU 线程数或者 GPU 的运行模式
        precision: 可选，默认是2，有效输入为：0(Normal), 1(High), 2(Low_FP16), 3(Low_BF16)
        weightSparsity: 可选，默认是 0.0 ，在 weightSparsity > 0.5 时且后端支持时，开启稀疏计算
        weightSparseBlockNumber: 可选，默认是 1 ，仅当 weightSparsity > 0.5 时生效，为稀疏计算 block 大小，越大越有利于稀疏计算的加速，一般选择 1, 4, 8, 16
        testQuantizedModel 可选，默认是0，即只测试浮点模型；取1时，会在测试浮点模型后进行量化模型的测试

    demo: $ ./benchmark.out ../benchmark/models/ 10
    结果: 
    [ - ] resnet-v2-50.mnn            max =   23.577 ms  min =   21.529 ms  avg =   22.095 ms
    [ - ] symbol_10_320_20L_5scales_v2_deploy.mnn    max =    5.290 ms  min =    5.119 ms  avg =    5.149 ms
    [ - ] MobileNetV2_224.mnn         max =    2.703 ms  min =    2.444 ms  avg =    2.511 ms
    [ - ] mobilenet-v1-1.0.mnn        max =   10.339 ms  min =    3.774 ms  avg =    5.603 ms
    [ - ] nasnet.mnn                  max =   14.011 ms  min =    8.022 ms  avg =    9.186 ms
    [ - ] SqueezeNetV1.0.mnn          max =    4.179 ms  min =    4.148 ms  avg =    4.157 ms
    [ - ] squeezenetv1.1.mnn          max =    2.585 ms  min =    2.505 ms  avg =    2.520 ms
    [ - ] mobilenetV3.mnn             max =    1.508 ms  min =    0.959 ms  avg =    1.081 ms
    [ - ] inception-v3.mnn            max =   29.295 ms  min =   28.982 ms  avg =   29.050 ms

## 模型量化
    $ ./quantized.out ../../../models/my_model.mnn  ../../../models/model_quan.mnn ../../../models/imageInputConfig.json
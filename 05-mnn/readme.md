
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

    $ python pymnn/examples/MNNQuant/test_mnn_offline_quant.py --mnn_model demo/model/MobileNet/mobilenet_v1.mnn --quant_imgs ../../../../Desktop/coco/val2017 --quant_model ./quant_model10.mnn

    $ python .//pymnn/examples/MNNExpr/mobilenet_demo.py ../../models/quant_model10.mnn ./demo/model/MobileNet/ILSVRC2012_val_00049999.JPEG

    $ python pymnn/examples/MNNQuant/test_mnn_offline_quant.py --mnn_model ../../models/mobilenet_v1.mnn --quant_imgs ../../../../../../Desktop/coco/val2017 --quant_model ./mobilenet_v1_quant.mnn

    $ python .//pymnn/examples/MNNExpr/mobilenet_demo.py ../../models/mobilenet_v1_quant.mnn ../../models/ILSVRC2012_val_00049999.JPEG

    $ ./quantized.out ../../../studyML/05-mnn/models/mobilenet_v1.mnn   ./model_quan.mnn   ../../../studyML/05-mnn/models/imageInputConfig.json

## benchmark
### yolov8n
    metal:
        [ - ] yolov8n.mnn                 max =   13.801 ms  min =   12.322 ms  avg =   12.843 ms
        [ - ] yolov8n.mnn                 max =   15.707 ms  min =   13.392 ms  avg =   14.354 ms
        [ - ] yolov8n.mnn                 max =   13.078 ms  min =   11.953 ms  avg =   12.285 ms
        [ - ] yolov8n.mnn                 max =   14.989 ms  min =   13.596 ms  avg =   14.137 ms
    opencl:
        total kernel time = 439  us, conv time = 0 us, while time = 1 us
        kernel time = 1    us outputFormatTransform
        total kernel time = 440  us, conv time = 0 us, while time = 0 us
        copyFromDevice, 773, cost time: 0.382000 ms
        [ - ] yolov8n.mnn                 max =   32.706 ms  min =   31.747 ms  avg =   32.072 ms

        total kernel time = 433  us, conv time = 0 us, while time = 1 us
        kernel time = 2    us outputFormatTransform
        total kernel time = 435  us, conv time = 0 us, while time = 0 us
        copyFromDevice, 773, cost time: 0.442000 ms
        [ - ] yolov8n.mnn                 max =   32.609 ms  min =   31.684 ms  avg =   32.252 ms

        total kernel time = 447  us, conv time = 0 us, while time = 1 us
        kernel time = 1    us outputFormatTransform
        total kernel time = 448  us, conv time = 0 us, while time = 0 us
        copyFromDevice, 773, cost time: 0.380000 ms
        [ - ] yolov8n.mnn                 max =   32.598 ms  min =   31.628 ms  avg =   32.279 ms

### yolov8s
    metal:
        [ - ] yolov8s.mnn                 max =   31.506 ms  min =   29.855 ms  avg =   31.038 ms
        [ - ] yolov8s.mnn                 max =   32.434 ms  min =   29.683 ms  avg =   31.653 ms
        [ - ] yolov8s.mnn                 max =   32.013 ms  min =   30.671 ms  avg =   31.577 ms
    opencl:
        total kernel time = 982  us, conv time = 0 us, while time = 1 us
        kernel time = 1    us outputFormatTransform
        total kernel time = 983  us, conv time = 0 us, while time = 0 us
        copyFromDevice, 773, cost time: 0.403000 ms
        [ - ] yolov8s.mnn                 max =   63.790 ms  min =   58.906 ms  avg =   60.852 ms

        total kernel time = 942  us, conv time = 0 us, while time = 1 us
        kernel time = 1    us outputFormatTransform
        total kernel time = 943  us, conv time = 0 us, while time = 0 us
        copyFromDevice, 773, cost time: 0.419000 ms
        [ - ] yolov8s.mnn                 max =   62.201 ms  min =   57.954 ms  avg =   60.528 ms

        
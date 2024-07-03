# TensorRT介绍
## TensorRT做的工作
### 构建期(推理优化器)
    1. 模型解析/建立：加载Onnx等其他格式的模型/使用原声API搭建模型
    2. 计算图融合：横向层融合(Conv)，纵向图融合(Conv+add+ReLU)，......
    3. 结点消除：去除无用层，节点变换(Pad, Slice, Concat, Shuffle), ......
    4. 多精度支持：FP32/PF16/INT8/TF32(可插入reformat节点)
    5. 优选kernel/format：硬件有关优化
    6. 导入plugin：实现自定义操作
    7. 显存优化：此案存池复用
### 运行期(运行时环境)
    1. 运行时环境：对象生命周期管理，内存显存管理，异常处理
    2. 序列化反序列化：推理引擎保存为文件或从文件中加载 


# 开发辅助工具
## trtexec
    1. TensorRT命令行工具
       1. 随TensorRT安装，位于tensorrt-XX/bin/trtexec 
    2. 功能：
       1. 由.onnx模型文件生成TensorRT引擎并序列化为.plan
       2. 查看.onnx会.plan文件的网络逐层信息
       3. 模型性能测试(测试TensorRT引擎基于随机输入或给定输入下的性能)
    3. 范例代码：
       1. 08-Tool/trtexec，运行./command.sh
    4. 常用选项
       1. 构建阶段
          1. --onnx=./xxx.onnx     指定输入模型文件名
          2. --workspace=1024      优化工程中可使用显存最大值(MiB)
          3. --fp16, --int8, --noTF32, --best, --sparsity=...      指定引擎精度和稀疏等属性
          4. --saveEngine=xxx.plan     指定输出引擎文件名
          5. --buidOnly            只创建引擎不运行
          6. --verbose             打印详细日志
          7. --timingCacheFile=timing.cache        指定输出优化计时缓存文件名
          8. --profilingVerbosity=detailed     构建期保留更多的逐层信息
          9.  --dumpLayerInfo, --exportLayerInfo=layerInfo.txt  导出引擎逐层信息，可与profilingVerbosity合用
       2. 运行阶段
          1. --loadEngine=xxx.plan      读取engine文件，而不是输入的onnx文件
          2. --shapes=1x1x28x28     指定输入张量大小
          3. --warmUp=1000      热身阶段最短运行时间(ms)
          4. --duration=10      测试阶段最短运行时间(s)
          5. --iterations=100   指定测试阶段运行的最小迭代次数
          6. --useCudaGraph     使用CUDAGraph来捕获和执行推理过程
          7. --noDataTransfers  关闭Host与Device之间的数据传输
          8. --streams=2        使用多个stream来运行推理
          9. --verbose          打印详细信息
          10. --dumpProfile, --exportProfile=layerProfile.txt   保存逐层性能数据信息    

## onnx-graphsurgeon
      需要手工修改网络的情行？冗余结点、阻碍TensorRT融合的结点组合、可以手工模块化的结点
      onnx-graphsurgeon：onnx模型编辑器，包含python API(简称ogs)
      功能：
      1. 修改计算图：图属性/结点/张量/结点和张量的链接/权重
      2. 修改子图：添加/删除/替/换隔离
      3. 优化计算图：常量折叠/拓扑排序/去除无用层
      4. 功能和API上有别与onnx
      安装：pip install nvidia-pyidex onnx-graphsurgen
## polygraphy
      如何检测TensorRT上计算结果正确性/精度？
      怎么找出计算错误/精度不足的层？
      怎么进行简单的计算图优化？
      polygraphy：深度学习模型调试器，包含python API和命令行工具
      功能：
      1. 使用多种后端运行推理计算，包含TensorRT，onnxruntime，TensorFlow
      2. 比较不同后端的逐层计算结果
      3. 由模型文件生成TensorRT引擎序列化为.plan
      4. 查看模型网络的逐层信息
      5. 修改onnx模型，如提取子图，计算图化简
      6. 分析onnx转TensorRT失败原因，将原计算图中可以/不可以转TensorRT的子图分割保存
      7. 隔离TensorRT中错误tatic
      安装：pip install polygraphy
## nsight systems
      性能调试器：
      1. 随cuda安装或独立安装，位于/user/local/cuda/bin/下的nsy和nsys-ui
      2. 替代旧工具nvprof和nvvp
      3. 首先命令行运行nsys profile XXX，获得.qdrep或.qdrep-nsys文件
      4. 然后打开nsys-ui，将上述文件拖入即可观察timeline
   
# Plugin
      功能：
      1. 以.so的形式插入到网络中实现某些算子
      2. 实现TensorRT不原生支持的层或结构
      3. 替换性能不足的层或结构
      4. 手动合并TensorRT没有自动融合的层或结构
      限制条件：
      1. 自己实现CUDA C++ kernel，为结果精度和性能负责
      2. Plugin与其他Layer之间无法fusing
      3. 可能在Plugin结点前后插入reformatting结点增加开销
      建议：
      1. 先尝试原声Layer的组合保证计算正确性
      2. 尝试TensorRT自带的Plugin是否满足要求
      3. 还是不满意，自己写

# 参考
    https://www.bilibili.com/video/BV15Y4y1W73E/?spm_id_from=333.999.0.0&vd_source=6b48595092f05a0fc1d129f83872951f
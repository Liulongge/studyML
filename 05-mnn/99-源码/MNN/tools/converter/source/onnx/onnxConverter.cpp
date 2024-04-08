//
//  onnxConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <queue>

#include "MNN_generated.h"
#include "OnnxUtils.hpp"
#include "logkit.h"

#include "OnnxTmpGraph.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/util.h"
#include "onnx.pb.h"
#include "onnxConverter.hpp"
#include "onnxOpConverter.hpp"

int onnx2MNNNet(const std::string inputModel, const std::string bizCode,
                std::unique_ptr<MNN::NetT>& netT) 
{
    // 获取onnx模型路径
    std::string modelDir;
    size_t pos = inputModel.find_last_of("\\/");
    if (pos != std::string::npos) 
    {
        modelDir = inputModel.substr(0, pos + 1);
    }

    // 实例化，从protobuf中读取onnx模型
    onnx::ModelProto onnxModel;
    // read ONNX Model
    bool success = onnx_read_proto_from_binary(inputModel.c_str(), &onnxModel);
    DCHECK(success) << "read onnx model failed: " << inputModel;
    if (!success) 
    {
        MNN_ERROR("[ERROR] Model file is not onnx model.\n");
        return 1;
    }

    // 从模型中获取ir与op版本信息
    int opsetVersion = 13;
    auto opsetInfo = onnxModel.opset_import();
    if (!opsetInfo.empty()) 
    {
        opsetVersion = static_cast<int>(opsetInfo.begin()->version());
    }
    LOG(INFO) << "ONNX Model ir version: " << onnxModel.ir_version();
    LOG(INFO) << "ONNX Model opset version: " << opsetVersion;

    // 获取graph与node(算子)数量信息
    const auto& onnxGraph = onnxModel.graph();
    const int nodeCount   = onnxGraph.node_size();
    printf("nodeCount: %d\n", nodeCount);

    // 根据onnxGraph与netT原始指针创建OnnxScope
    std::unique_ptr<OnnxScope> scope(new OnnxScope(&onnxGraph, netT.get()));
    scope->mOpsetVersion = opsetVersion;
    // find the inputs which do not have initializer
    const auto& initializers         = scope->mInitializers;
    const auto& inputs               = scope->mInputs; // onnx模型(graph)输入
    const auto& outputs              = scope->mOutputs;
    // set input node to MNN net
    for (const auto& iter : inputs) 
    {
        // 初始化mnn输入node(op)
        bool notHaveInitializer = initializers.find(iter.first) == initializers.end();
        if (notHaveInitializer) 
        {
            MNN::OpT* MNNOp  = new MNN::OpT;
            MNNOp->name      = iter.first;
            MNNOp->type      = MNN::OpType_Input;
            MNNOp->main.type = MNN::OpParameter_Input;
            auto inputParam  = new MNN::InputT;
            const auto it    = inputs.find(iter.first);
            //FUNC_PRINT_ALL(iter.first.c_str(), s);
            DCHECK(it != inputs.end()) << "Input Paramter ERROR ==> " << iter.first;
            const auto& tensorInfo = (it->second)->type().tensor_type();
            const int inputDimSize = tensorInfo.shape().dim_size();
            inputParam->dims.resize(inputDimSize);
            // 获取onnx输入node dim信息，并初始化mnn node
            for (int i = 0; i < inputDimSize; ++i) 
            {
                const auto& dim = tensorInfo.shape().dim(i);
                if (dim.has_dim_value()) 
                {
                    inputParam->dims[i] = static_cast<int32_t>(dim.dim_value());
                } 
                else 
                {
                    inputParam->dims[i] = -1;
                }
            }
            inputParam->dtype   = onnxOpConverter::convertDataType(tensorInfo.elem_type());
            inputParam->dformat = MNN::MNN_DATA_FORMAT_NCHW;
            MNNOp->outputIndexes.push_back(scope->declareTensor(iter.first));
            MNNOp->main.value = inputParam;
            netT->oplists.emplace_back(MNNOp);
        }
    }

    // onnx model not all topo sort graph, sort it
    // onnx模型可能不是拓扑排序好的，进行拓扑排序
    std::vector<int> idxMap = OnnxScope::topoSort(onnxGraph);

    // lambda表达式，
    auto makeConst = [&](const std::string& inputName) 
    {
        // ONNX模型中的权重和参数信息实际上是存储在模型的初始化器（initializers）中的。
        // 在ONNX模型中，初始化器是一组预先定义的常量值，用于在模型推理过程中作为输入提供给操作节点。
        // 这些初始化器通常包含模型的权重和偏置等参数。
        const auto it         = initializers.find(inputName);
        if (it != initializers.end() && scope->lookupTensor(it->first) == -1) 
        {
            static int a =0 ;
            // Create const Op
            // 创建常量op
            MNN::OpT* constOp   = new MNN::OpT;
            constOp->type       = MNN::OpType_Const;
            constOp->main.type  = MNN::OpParameter_Blob;
            constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second, modelDir);
            constOp->name    = it->first;
            constOp->outputIndexes.push_back(scope->declareTensor(it->first));
            netT->oplists.emplace_back(constOp);
        }
    };

    // 模型(graph)输出node
    for (int i=0; i<onnxGraph.output_size(); ++i) 
    {
        // lambda表达式调用, 创建输出op并初始化
        makeConst(onnxGraph.output(i).name());
    }

    // Declare all outputs
    for (int idx = 0; idx < nodeCount; ++idx) 
    {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        const auto& onnxNode = onnxGraph.node(i);
        for (int k = 0; k < onnxNode.output_size(); k++) 
        {
            scope->declareTensor(onnxNode.output(k));
        }
    }

    // onnx node ==> MNN node
    // onnx node转MNN node
    for (int idx = 0; idx < nodeCount; ++idx) 
    {
        int i = idxMap.size() == nodeCount ? idxMap[idx] : idx;
        // onnx node
        const auto& onnxNode = onnxGraph.node(i);
        // 算子类型(string): Conv, ReLu, Add, AveragePool等
        const auto& opType   = onnxNode.op_type();

        // name maybe null, use the first output name as node-name
        // node名字可能是空的，使用第一个输出的名字作为node名字
        // node名字(string类型): input.4, onnx::Pad_188, onnx::Conv_128等
        const auto& name = onnxNode.output(0);
        // onnx op转换器套装, 查找并返回对应onnx算子类型的算子转换器
        auto opConverter = onnxOpConverterSuit::get()->search(opType);

        // 创建opT并初始化
        MNN::OpT* MNNOp  = new MNN::OpT;
        MNNOp->name      = name; // op名字
        MNNOp->type      = opConverter->opType(); // op类型, see OpType
        MNNOp->main.type = opConverter->type(); // op param类型, see OpParameter

        // convert initializer to be Constant node(op)
        // 输入node
        for (int k = 0; k < onnxNode.input_size(); ++k) 
        {
            static int a = 0;
            const auto& inputName = onnxNode.input(k);
            makeConst(inputName);
        }

        // build input and output
        for (int k = 0; k < onnxNode.input_size(); k++) 
        {
            int inputIdx = scope->lookupTensor(onnxNode.input(k));
            if (inputIdx < 0) 
            {
                LOG(INFO) << "Check it out ==> " << MNNOp->name << " has empty input, the index is " << k;
            }
            MNNOp->inputIndexes.push_back(inputIdx);
        }
        for (int k = onnxNode.input_size() - 1; k >= 0 && MNNOp->inputIndexes[k] < 0; --k) 
        {
            MNNOp->inputIndexes.pop_back();
        }
        for (int k = 0; k < onnxNode.output_size(); k++) 
        {
            MNNOp->outputIndexes.push_back(scope->declareTensor(onnxNode.output(k)));
        }
        // build op
        // 调用不同的子类进行构建(例如：MNN/tools/converter/source/onnx/ReluOnnx.cpp)
        opConverter->run(MNNOp, &onnxNode, scope.get());
        // 将构建好的op送入vector保存
        netT->oplists.emplace_back(MNNOp);
    }
    netT->tensorNumber = netT->tensorName.size();
    // set MNN net output name
    for (const auto& iter : outputs) 
    {
        netT->outputName.push_back(iter.first);
    }

    netT->sourceType = MNN::NetSource_ONNX;
    netT->bizCode    = bizCode;

    return 0;
}

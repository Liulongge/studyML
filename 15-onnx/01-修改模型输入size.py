import onnx
import onnx_tool

model = onnx.load('yolov8s.onnx')


# 检查模型的有效性
onnx.checker.check_model(model)

# 获取模型的图结构
graph = model.graph

# 修改输入尺寸
for input in graph.input:
    if input.name == 'images':  # 假设输入的名字是 'input'
        # 更新输入的形状信息
        input.type.tensor_type.shape.dim[0].dim_value = 1
        input.type.tensor_type.shape.dim[1].dim_value = 3
        input.type.tensor_type.shape.dim[2].dim_value = 1280
        input.type.tensor_type.shape.dim[3].dim_value = 1280
        
        print(f"Updated input shape to {input.type.tensor_type.shape}")

# 保存修改后的模型
output_model_path = 'modified_model.onnx'
onnx.save(model, output_model_path)

# 再次检查模型的有效性
onnx.checker.check_model(output_model_path)
print("Modified model is valid.")
onnx.save(model, 'modified_model.onnx')

onnx.checker.check_model('modified_model.onnx')
onnx_tool.model_profile('modified_model.onnx')
onnx_tool.model_profile('yolov8s.onnx')
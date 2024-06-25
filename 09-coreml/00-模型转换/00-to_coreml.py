# https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
import torch  
import torchvision.models as models
import coremltools as ct  
  
# 1. 加载预训练的PyTorch模型  
model = models.resnet18(pretrained=True)  
model.eval()  
  
# 2. 创建一个随机的输入张量  
x = torch.randn(1, 3, 224, 224)  # 例如，对于ResNet18，输入应为3x224x224  
  
# 3. 使用torch.jit.trace将模型转换为TorchScript（或者你可以使用torch.jit.script，但trace通常更简单）  
traced_script_module = torch.jit.trace(model, x)  
  
# 4. 将TorchScript模型保存为.pt文件（可选，但有助于后续调试）  
traced_script_module.save("resnet18.pt")  
  

scripted_model = torch.jit.load("resnet18.pt") 

mlmodel = ct.converters.convert(
    scripted_model,
    inputs=[ct.TensorType(shape=(1, 3, 64, 64))],
)

# Save the converted model.
mlmodel.save("resnet18.mlpackage")
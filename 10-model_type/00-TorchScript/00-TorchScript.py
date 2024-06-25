# https://blog.csdn.net/weixin_43603658/article/details/129858853
# TorchScript简介
# TorchScript是PyTorch模型推理部署的中间表示，可以在高性能环境libtorch（C ++）中直接加载，
# 实现模型推理，而无需Pytorch训练框架依赖。 torch.jit是torchscript Python语言包支持，
# 支持pytorch模型快速，高效，无缝对接到libtorch运行时，实现高效推理。

# 在 PyTorch 中，.pth, .pt, 和 .pkl 文件通常用来保存模型的状态字典（state_dict）或者整个模型（包括模型架构和状态字典）。
# 而 TorchScript 则是另一种完全不同的概念，它涉及到模型的序列化和优化，通常保存为 .pt 或者 .ts 文件。

# ### `.pth` 和 `.pt` 文件
# 这两种文件格式在 PyTorch 中几乎可以互换使用，它们主要用于保存以下内容：
# - 模型的状态字典：这包含了模型的所有可学习参数（权重和偏置）。
# - 完整的模型：除了状态字典，还可以保存模型的架构以及其他相关数据如优化器状态、训练指标等。
# `.pth` 和 `.pt` 文件通常通过 `torch.save()` 函数来创建，使用 `torch.load()` 来加载。
# 加载时，如果是状态字典，你需要先定义好模型架构，然后使用 `model.load_state_dict()` 方法来加载参数；如果是完整的模型，直接加载即可。

# ### `.pkl` 文件
# `.pkl` 文件使用 Python 的 `pickle` 库来序列化和反序列化对象。
# 在 PyTorch 中，`.pkl` 也可以用来保存模型和其他数据结构。
# 虽然 `.pkl` 文件可以保存任何 Python 对象，但在 PyTorch 的上下文中，
# `.pkl` 和 `.pth`/`.pt` 文件通常可以互换使用，保存的内容也类似。

# ### `TorchScript` 文件
# `TorchScript` 是 PyTorch 的一种静态类型系统，它提供了两种方式来表示 PyTorch 模型：
# - 脚本化（Scripting）：将模型转换为 TorchScript 代码。
# - 追踪（Tracing）：记录模型的前向传播过程，然后将这些操作序列化为 TorchScript 图。

# TorchScript 模型可以被优化并部署在没有 Python 解释器的环境中，例如移动设备或嵌入式系统。
# TorchScript 文件通常也保存为 `.pt` 或者 `.ts` 文件。
# 它们可以通过 `torch.jit.script()` 或 `torch.jit.trace()` 创建，使用 `torch.jit.load()` 来加载。

# 总结来说，`.pth`, `.pt`, 和 `.pkl` 主要是用于保存 PyTorch 模型和数据的通用文件格式，
# 而 `TorchScript` 是一种专门的格式，用于序列化和优化模型，以便在不同的环境中高效运行。

import torch  
import torch.nn as nn  
  
# 定义一个简单的模型  
class SimpleModel(nn.Module):  
    def __init__(self):  
        super(SimpleModel, self).__init__()  
        self.fc = nn.Linear(10, 1)  
  
    def forward(self, x):  
        x = self.fc(x)  
        return x  
  
# 实例化模型  
model = SimpleModel()  
  
# 示例输入数据  
example = torch.randn(1, 10)  
  
# 使用 torch.jit.trace 或 torch.jit.script 转换模型  
# 这里我们使用 torch.jit.script 因为它会编译整个模型为TorchScript  
traced_script_module = torch.jit.script(model)  
  
# 保存 TorchScript 模型  
traced_script_module.save("model.pt")

# 加载 TorchScript 模型  
loaded_model = torch.jit.load("model.pt")  
  
# 验证模型是否加载成功  
# 使用相同的示例输入数据  
output = loaded_model(example)  
print(output)
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch 

dummy_input = Variable(torch.randn(1, 3, 32, 32))
model = torch.load('./my_model.pth')
torch.onnx.export(model, dummy_input, "my_model.onnx")


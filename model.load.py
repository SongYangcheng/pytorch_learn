import torch
import torchvision
from torch import nn

#方式一保存方式，加载模型

model = torch.load("vgg16_methodl.pth")
print(model)

#方式2，加载模式
vgg16 = torchvision.models.vgg16(prentrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(model)
# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        def forward(self, x):
            x= self.conv1(x)
            return x


tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
model = torch.load("tudui_methodl.pth")
print(model)

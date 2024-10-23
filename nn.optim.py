from torch import nn
import torch
from torch.nn import Conv2d, MaxPool2d, Linear,Flatten,Sequential
from torch.utils.tensorboard import SummaryWriter
# C10数据集
dataset = torchvision.datasets.CIFAR10("data_1", train=False, tranform=torchvision.transforms.ToTensor())
datloader = Dataloader(dataset, batch_size=64)
# 网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential( 
          Conv2d(3, 32, 5, padding=2),
          MaxPool2d(2),
           Conv2d(32, 32, 5, padding=2),
           MaxPool2d(2),
           Conv2d(32, 64, 5, padding=2),
           MaxPool2d(2),
           Flatten(),
           Linear(1024, 64),
           Linear(64, 10),
        )
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x
loss = nn.CrossEntropyLoss() #交叉商
tudui = Tudui()
# 优化器引入
optim = torch.optim.SGD(tudui.parameters(), lr=0.01, )
for epoch in range(20):
    ruuinf_logs = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        print(outputs)
        print(targets)
        print(result_loss)
        #反向传播
        result_loss = loss(outputs, targets)
        # result_loss.backward()
        optim.zero_grad() #对每个函数梯度进行清零
        result_loss.backward()
        optim.step()
        # print(result_loss)
        # print("ok")
    
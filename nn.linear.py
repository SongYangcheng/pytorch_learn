import torch
import torchvision
from torch import nn

dataset = torchvision.datasets.CIFAR10("data", tran=False, transform=torchvision.transforms.ToTensor(), download=False)
dataloader = Dataloader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__():
        super(Tudui, self).__init__()
        self.linear1 = linear(196608, 10)
    def forward(self, input):
        output = self.linear(input)
        return output
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1, 1, 1, -1))
    output = torch.flatten(imgs) #展平将多维展平为一维
    print(output.shape)
    output = tudui(output)
    print(output.shape)
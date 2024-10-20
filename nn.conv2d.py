import torch
import torchvision 
dataset = torchvision.datasets.CIFAR10("data/", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        supper(Tudui,self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()
writer = Summarywriter("../logs")
for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)
    print(imgs.shape)

    print(output.shape)
    writer.add_imges("input", imgs, step)


    torch.reshape(output, (-1, 3, 30 , 30))
    writer.add_images("output", output, step)
    step  += 1
    
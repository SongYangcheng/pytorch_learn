# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from PIL import Image
# # help(SummaryWriter)
# writer = SummaryWriter("logs")
# image_path = "hymenoptera_data/train/ants/0013035.jpg"
# img_PIL = Image.open(image_path)
# img_array = np.array(img_PIL)
# writer.add_image("test", img_array, 1, dataformats='HWC')
# print(img_array.shape)
# for i in range(0, 99):

#     writer.add_scalar("y = x", i, i)

# writer.close()

from PIL import Image
import torchvision

from torch import nn
import torch
image_path = 'hymenoptera_data/train/bees/16838648_415acd9e3f.jpg'
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()]) 

image = transform(image)
print(image.shape)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
           nn.Conv2d(3, 32, 5, 1, 2),#卷积
           nn.MaxPool2d(2), #最大池化
           nn.Conv2d(32, 32, 5, 1, 2),
           nn.MaxPool2d(2),
           nn.Conv2d(32, 64, 5, 1, 2),
           nn.MaxPool2d(2),
           nn.Flatten(), #进行展平
           nn.Linear(64*10*10, 64),
           nn.Linear(64, 10)

        )
    def forward(self, x):
        x = self.model(x)
        return x
model = torch.load("tudui_0.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
output = model(image)
print(output)
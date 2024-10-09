from torchvision import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
# python 的用法，tensor数据类型
#transforms.Totensor解决两个问题
#1. transforms的使用
#2. 为什么我们需要Tensor数据类型
img_path= "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)
writer.close()

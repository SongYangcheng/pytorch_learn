from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
import torch

writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data/train/ants/0013035.jpg").convert('RGB')
print(img)
img_np = np.asarray(img) 
# ToTensor 使用
# trans_totensor = transforms.ToTensor()
# img_tensor = trans_totensor(img)
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0 
writer.add_image("ToTensor", img_tensor)

# Normallize 使用
print(img_tensor[0][0][0])
trans_norm = transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize 使用
print(img.size)
tran_resize = tranforms.Resize((512, 5122))
img_resize = tran_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_imge("Resize", img_resize, 0)
print(img_resize)
# Compose 使用
trans_resize  = transforms.Resize(512)
tran_resize_2 = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
# Randomcrop 随机裁剪
trans_random  = rranforms.RandomCrop(512)
trans_compose_2 = trandorms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()

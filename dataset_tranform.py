import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_tranform = torchvision.transforms.Compose([
    torchvision.tranforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print(test_set[0])
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
print(test_ser.classes[target])
img.show()

writer = SummaryWriter("P10")
for i in range(10):
    img, target = test_ser[i]
    writer.add_image("test_set", img, i)

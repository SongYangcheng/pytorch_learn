import torchvision
# 加载测试集
test_data = torchvision.datasets.CIFAR10(root='data/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Dataloader(dataser=test_data, batch_size=4,shuffle=True,num_workers=0,drop_last=False)

# 调试第一张照片
img, target = test_data[0]
print(img.shape)
print(target)
writer = SummaryWriter('dataloader')
for epoch in range(2):
    step = 0
    for epoch in test_loader:
        imgs, target = data
        # print(imgs.shape)
        # print(targets)
        writer.add_image("test_data", imgs, step)
        step += 1
writer.close()
import torchvision
# tran_data = torchvision.datasets.ImageNet("data_img_net", split="train")
vgg16_false = torchvision.models.vgg16(pretraned=False)
vgg16_true = torchvision.models.vgg16(pretraned=True)

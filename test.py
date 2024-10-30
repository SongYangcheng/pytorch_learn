import torch
outputs = torch.tensor([[0.1, 0.2],
                          [0.3, 0.4],
                          [0.6, 0.5],
                          [0.7, 0.6]])  # 当outputs.argmax(1)时每个列表内进行比较，后面大返回1，否则返回0
print(outputs.argmax(1))
preds = outputs.argmax(1)
targets = torch.tensor([0, 1, 0, 1])
print((preds == targets).sum())
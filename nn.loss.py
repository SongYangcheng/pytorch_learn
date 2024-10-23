import torch
from torch.nn import L1Loss
from torch import nn
input = torch.tensor([1, 2, 3], dtype=torch.float32)

targets = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(input, targets)

# 平均差 MELoss
loss_mse = nn.MSELoss()
result_mse = loss_mse(input, targets)

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))
loss_core = nn.CrossEntropyLoss()
result_cross = loss_core(x, y)
print(result_cross)

import torch.nn as nn
import torch
class Todui(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = input + 1
        return output

tudui = Todui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)


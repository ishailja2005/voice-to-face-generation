import torch

from models.generator import Generator

model = Generator()

x = torch.randn(8,256)

output = model(x)

print(output.shape)
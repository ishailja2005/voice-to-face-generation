import torch

from training.loss import criterion

generated = torch.randn(8, 3, 160, 160)

real = torch.randn(8, 3, 160, 160)

loss = criterion(generated, real)

print(loss)

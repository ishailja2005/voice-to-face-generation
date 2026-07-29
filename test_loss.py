import torch

from training.loss import Speech2FaceLoss

predicted_features = torch.randn(8, 512)
target_features = torch.randn(8, 512)
generated = torch.randn(8, 3, 160, 160)
real = torch.randn(8, 3, 160, 160)
loss, _, _ = Speech2FaceLoss()(predicted_features, target_features, generated, real)

print(loss)

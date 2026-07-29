"""Face feature encoder and decoder used for the two-stage Speech2Face setup."""

import torch.nn as nn


class FaceEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(512 * 5 * 5, latent_dim))

    def forward(self, image):
        return self.to_latent(self.features(image))


class FaceDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 512 * 5 * 5), nn.ReLU(inplace=True))
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, latent):
        features = self.from_latent(latent).view(-1, 512, 5, 5)
        return self.decode(features)

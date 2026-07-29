"""Maps a 256-D speech embedding to the learned face-feature space."""

import torch.nn as nn


class VoiceEncoder(nn.Module):
    def __init__(self, input_dim=256, face_feature_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, face_feature_dim),
        )

    def forward(self, embedding):
        return self.network(embedding)

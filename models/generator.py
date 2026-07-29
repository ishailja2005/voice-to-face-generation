"""Compatibility wrapper: voice encoder followed by the frozen face decoder."""

import torch.nn as nn

from .face_autoencoder import FaceDecoder
from .voice_encoder import VoiceEncoder


class Generator(nn.Module):
    def __init__(self, input_dim=256, face_feature_dim=512):
        super().__init__()
        self.voice_encoder = VoiceEncoder(input_dim, face_feature_dim)
        self.face_decoder = FaceDecoder(face_feature_dim)

    def forward(self, embedding):
        return self.face_decoder(self.voice_encoder(embedding))

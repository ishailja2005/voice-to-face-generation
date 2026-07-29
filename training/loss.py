"""Losses for the feature-regression stage of Speech2Face."""

import torch.nn as nn


class Speech2FaceLoss(nn.Module):
    """Regress face features and retain a small canonical-image reconstruction term."""

    def __init__(self, feature_weight=1.0, image_weight=0.1):
        super().__init__()
        self.feature_weight = feature_weight
        self.image_weight = image_weight
        self.feature_loss = nn.SmoothL1Loss()
        self.image_loss = nn.L1Loss()

    def forward(self, predicted_features, target_features, reconstructed_image, target_image):
        feature = self.feature_loss(predicted_features, target_features)
        image = self.image_loss(reconstructed_image, target_image)
        return self.feature_weight * feature + self.image_weight * image, feature, image


criterion = Speech2FaceLoss()

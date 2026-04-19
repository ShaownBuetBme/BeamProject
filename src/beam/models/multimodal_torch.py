from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class MultimodalRegressor(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        output_dim: int = 2,
        image_embedding_dim: int = 128,
        numeric_embedding_dim: int = 32,
        fusion_hidden_dim: int = 64,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.image_projection = nn.Sequential(
            nn.Linear(in_features, image_embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.numeric_encoder = nn.Sequential(
            nn.Linear(num_numeric_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.1),
            nn.Linear(32, numeric_embedding_dim),
            nn.ReLU(),
        )

        fusion_in = image_embedding_dim + numeric_embedding_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        img_features = self.image_backbone(image)
        img_features = self.image_projection(img_features)

        num_features = self.numeric_encoder(numeric)

        fused = torch.cat([img_features, num_features], dim=1)
        return self.regression_head(fused)

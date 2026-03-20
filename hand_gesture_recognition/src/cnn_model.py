"""models/cnn_model.py — Transfer-learning CNN classifier for gesture images."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


BackboneType = Literal["mobilenet_v3", "resnet18", "resnet50", "efficientnet_b0"]


class GestureCNN(nn.Module):
    """
    Pretrained CNN backbone with a custom classification head.

    MobileNetV3-Small is the default — fast on CPU, suitable for
    real-time inference on edge devices. Swap to EfficientNet-B0
    or ResNet-50 for higher accuracy on more classes.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: BackboneType = "mobilenet_v3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        weights = "DEFAULT" if pretrained else None

        if backbone == "mobilenet_v3":
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[3].in_features
            base.classifier[3] = nn.Identity()
            self.feature_dim = in_features

        elif backbone == "resnet18":
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.feature_dim = in_features

        elif backbone == "resnet50":
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.feature_dim = in_features

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier[1] = nn.Identity()
            self.feature_dim = in_features

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, feature_dim)
        return self.classifier(features)     # (B, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (useful for visualization / t-SNE)."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone for fine-tuning only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

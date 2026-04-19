import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

IDX_TO_VIS  = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES = ["Clear", "Light", "Medium", "Dense"]

class FogDetector_EfficientNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        backbone    = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.backbone     = backbone
        self.confidence   = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._in_features = in_features

    def forward(self, x):
        features   = self.backbone.features(x)
        features   = self.backbone.avgpool(features)
        features   = features.view(features.size(0), -1)
        logits     = self.backbone.classifier(features)
        confidence = self.confidence(features)
        return logits, confidence
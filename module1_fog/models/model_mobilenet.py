import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

IDX_TO_VIS  = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES = ["Clear", "Light", "Medium", "Dense"]

class FogDetector_MobileNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)

        self.backbone     = backbone
        self.confidence   = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._in_features = in_features

    def forward(self, x):
        features   = self.backbone.features(x)
        features   = self.backbone.avgpool(features)
        features   = features.view(features.size(0), -1)
        features   = self.backbone.classifier[:3](features)
        logits     = self.backbone.classifier[3](features)
        confidence = self.confidence(features)
        return logits, confidence
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

IDX_TO_VIS  = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES = ["Clear", "Light", "Medium", "Dense"]

class FogDetector_ConvNeXt(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        backbone    = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = 768   # ConvNeXt-Tiny feature dim after global avg pool

        # Replace final linear
        backbone.classifier[2] = nn.Linear(in_features, num_classes)

        self.features   = backbone.features      # spatial feature extractor
        self.norm       = backbone.classifier[0] # LayerNorm2d(768)
        self.pool       = nn.AdaptiveAvgPool2d(1) # (B,768,H,W) → (B,768,1,1)
        self.classifier = backbone.classifier[2] # Linear(768, 4)

        self.confidence = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)          # (B, 768, 7, 7)
        x = self.norm(x)              # (B, 768, 7, 7)  LayerNorm2d
        x = self.pool(x)              # (B, 768, 1, 1)  global avg pool
        x = x.view(x.size(0), -1)    # (B, 768)         flatten

        logits     = self.classifier(x)
        confidence = self.confidence(x)
        return logits, confidence
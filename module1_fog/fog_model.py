import torch
import torch.nn as nn
from torchvision import models

IDX_TO_VIS  = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES = ['Clear', 'Light', 'Medium', 'Dense']

class FogDetector(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super(FogDetector, self).__init__()

        # Pretrained ResNet18 backbone
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final classification layer — keep feature extractor
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Classification head: 512 -> 128 -> 4 classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # Confidence head: 512 -> 64 -> 1 (sigmoid 0-1)
        self.confidence = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features   = self.features(x)
        features   = features.view(features.size(0), -1)  # flatten
        logits     = self.classifier(features)
        confidence = self.confidence(features)
        return logits, confidence


def get_visibility_score(logits):
    probs      = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)
    vis_scores = torch.tensor([IDX_TO_VIS[c.item()] for c in pred_class])
    return pred_class, vis_scores, probs.max(dim=1).values

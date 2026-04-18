"""
Standalone inference for Module 1 — Fog Detection.
Loads model once; predict_fog() is the fusion layer interface.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --weights path/to/fog_best.pth
"""

import os
import argparse
import torch
from PIL import Image

from fog_model import FogDetector
from fog_dataset import val_transform
from config_loader import load_config

cfg    = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_VIS      = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES     = ["Clear", "Light", "Medium", "Dense"]
VIS_WEIGHTS     = torch.tensor([1.0, 0.7, 0.4, 0.1])          # Koschmieder-grounded
DEFAULT_WEIGHTS = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["weights_file"])


def load_model(weights_path: str = DEFAULT_WEIGHTS,
               device: torch.device = DEVICE) -> FogDetector:
    """Load FogDetector from checkpoint. Call once at startup."""
    model = FogDetector(num_classes=4).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[fog] Model loaded from {weights_path} on {device}")
    return model


def predict_fog(image_path: str,
                model: FogDetector,
                device: torch.device = DEVICE) -> dict:
    """
    Run fog classification on a single image.

    Args:
        image_path : path to input image
        model      : loaded FogDetector (call load_model() once)
        device     : torch device

    Returns dict with keys:
        fog_class             : str   — Clear / Light / Medium / Dense
        visibility_score      : float — discrete Koschmieder score (1.0/0.7/0.4/0.1)
        continuous_visibility : float — probability-weighted score (smooth, 0.1–1.0)
        confidence            : float — softmax probability of predicted class (0–1)
        class_probs           : list  — softmax over all 4 classes
    """
    image  = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs      = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

    weights  = VIS_WEIGHTS.to(device)
    cont_vis = float((probs * weights).sum().item())

    return {
        "fog_class":             CLASS_NAMES[pred_class],
        "visibility_score":      IDX_TO_VIS[pred_class],
        "continuous_visibility": round(cont_vis, 4),
        "confidence":            round(float(probs.max().item()), 4),
        "class_probs":           [round(p, 4) for p in probs.squeeze().tolist()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fog Module Inference")
    parser.add_argument("--image",   required=True,           help="Path to input image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pth weights")
    args = parser.parse_args()

    fog_model = load_model(args.weights)
    result    = predict_fog(args.image, fog_model)

    print("\n── Fog Detection Result ──")
    for k, v in result.items():
        print(f"  {k:<25}: {v}")
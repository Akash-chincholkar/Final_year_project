"""
Ablation study evaluation — loads all trained weights and evaluates
each on the test set. Prints a comparison table with inference time.

Run after train_ablation.py has completed.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fog_dataset import FogDataset, val_transform, CLASS_NAMES
from config_loader import load_config

from models.fog_model          import FogDetector
from models.model_mobilenet    import FogDetector_MobileNet
from models.model_efficientnet import FogDetector_EfficientNet
from models.model_convnext     import FogDetector_ConvNeXt

cfg    = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOG_DIR  = cfg["paths"]["dataset_dir"]
SAVE_DIR = os.path.join(cfg["paths"]["save_dir"], "ablation")

MODELS = {
    "resnet18":     FogDetector,
    "mobilenet":    FogDetector_MobileNet,
    "efficientnet": FogDetector_EfficientNet,
    "convnext":     FogDetector_ConvNeXt,
}


def measure_inference_time(model, device, n=100):
    """Average inference time per image in milliseconds."""
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(n):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / n * 1000
    return round(elapsed, 2)


def evaluate_one(name, model_cls):
    if name == "resnet18":
        weights_path = os.path.join(
            cfg["paths"]["save_dir"], cfg["paths"]["weights_file"]
        )
    else:
        weights_path = os.path.join(SAVE_DIR, f"{name}_best.pth")

    if not os.path.exists(weights_path):
        print(f"  [{name}] weights not found at {weights_path} — skipping")
        return None

    model = model_cls(num_classes=4).to(DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    inf_time = measure_inference_time(model, DEVICE)

    test_ds     = FogDataset(FOG_DIR, "test", val_transform)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc    = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1_mac = f1_score(all_labels, all_preds, average="macro")
    f1_per = f1_score(all_labels, all_preds, average=None)

    return {
        "accuracy":  acc,
        "f1_macro":  f1_mac,
        "f1_clear":  f1_per[0],
        "f1_light":  f1_per[1],
        "f1_medium": f1_per[2],
        "f1_dense":  f1_per[3],
        "inf_ms":    inf_time,
    }


def plot_accuracy_vs_speed(all_results, save_dir):
    """Scatter plot — accuracy vs inference time per architecture."""
    names  = list(all_results.keys())
    accs   = [all_results[n]["accuracy"] for n in names]
    speeds = [all_results[n]["inf_ms"]   for n in names]

    colors = {
        "resnet18":     "#1D9E75",
        "mobilenet":    "#888780",
        "efficientnet": "#888780",
        "convnext":     "#888780",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for n, a, s in zip(names, accs, speeds):
        ax.scatter(s, a, s=120,
                   color=colors.get(n, "#888780"),
                   zorder=3)
        ax.annotate(n, (s, a),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=10)

    # Annotate selected model
    resnet_spd = all_results["resnet18"]["inf_ms"]
    resnet_acc = all_results["resnet18"]["accuracy"]
    ax.annotate("Selected ✓",
                xy=(resnet_spd, resnet_acc),
                xytext=(8, -18),
                textcoords="offset points",
                fontsize=9,
                color="#1D9E75")

    ax.set_xlabel("Inference time (ms/image)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Inference Time — Architecture Comparison",
                 fontsize=12)
    ax.grid(True, alpha=0.3)

    # Dynamic y limits with padding
    min_acc = min(accs)
    max_acc = max(accs)
    ax.set_ylim(min_acc - 0.5, max_acc + 0.5)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ablation_accuracy_vs_speed.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Plot saved to {save_path}")


if __name__ == "__main__":
    print(f"\n{'='*85}")
    print("ABLATION STUDY — TEST SET RESULTS")
    print(f"{'='*85}")
    print(f"  {'Model':<15} {'Acc%':>6} {'F1-mac':>8} "
          f"{'F1-Clear':>10} {'F1-Light':>10} {'F1-Med':>8} "
          f"{'F1-Dense':>9} {'ms/img':>8}")
    print(f"  {'-'*83}")

    all_results = {}
    for name, cls in MODELS.items():
        r = evaluate_one(name, cls)
        if r is None:
            continue
        all_results[name] = r
        marker = " ←" if name == "resnet18" else ""
        print(f"  {name:<15} {r['accuracy']:>6.2f} {r['f1_macro']:>8.4f} "
              f"{r['f1_clear']:>10.4f} {r['f1_light']:>10.4f} "
              f"{r['f1_medium']:>8.4f} {r['f1_dense']:>9.4f} "
              f"{r['inf_ms']:>7.1f}ms{marker}")

    print(f"  {'='*83}")

    if all_results:
        best_acc = max(all_results, key=lambda k: all_results[k]["accuracy"])
        best_spd = min(all_results, key=lambda k: all_results[k]["inf_ms"])

        print(f"\n  Best accuracy : {best_acc.upper()} "
              f"({all_results[best_acc]['accuracy']:.2f}%)")
        print(f"  Fastest model : {best_spd.upper()} "
              f"({all_results[best_spd]['inf_ms']:.1f} ms/image)")
        print(f"\n  Selected      : RESNET18 — best accuracy/speed tradeoff")
        print(f"{'='*85}\n")

        # Generate plot — only runs if at least one model evaluated
        plot_accuracy_vs_speed(all_results, SAVE_DIR)
"""
Ablation study training
Trains MobileNet, EfficientNet, ConvNeXt
using existing fog_dataset.py

Uses:
- Same hyperparameters for all models
- Best model saving
- ReduceLROnPlateau
- Early stopping
"""

import os
import sys
import time
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# -------------------------------------------------
# PROJECT IMPORTS
# -------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config

from fog_dataset import (
    FogDataset,
    train_transform,
    val_transform
)

from models.model_mobilenet import FogDetector_MobileNet
from models.model_efficientnet import FogDetector_EfficientNet
from models.model_convnext import FogDetector_ConvNeXt


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
cfg = load_config()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

FOG_DIR = cfg["paths"]["dataset_dir"]
SAVE_DIR = os.path.join(cfg["paths"]["save_dir"], "ablation")
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = cfg["training"]["epochs"]
LR = cfg["training"]["lr"]
BATCH_SIZE = cfg["training"]["batch_size"]

EARLY_STOPPING_PATIENCE = 7


# -------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------
MODELS = {
    "convnext": FogDetector_ConvNeXt,
}


# -------------------------------------------------
# DATALOADERS
# -------------------------------------------------
def get_dataloaders(root_dir, batch_size):

    num_workers = min(4, os.cpu_count())

    train_dataset = FogDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform
    )

    val_dataset = FogDataset(
        root_dir=root_dir,
        split="val",
        transform=val_transform
    )

    test_dataset = FogDataset(
        root_dir=root_dir,
        split="test",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------
# TRAIN ONE MODEL
# -------------------------------------------------
def train_one(name, model_cls):

    print(f"\n{'=' * 60}")
    print(f"Training: {name.upper()}")
    print(f"{'=' * 60}")

    train_loader, val_loader, _ = get_dataloaders(
        FOG_DIR,
        BATCH_SIZE
    )

    model = model_cls(num_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    early_counter = 0

    save_path = os.path.join(
        SAVE_DIR,
        f"{name}_best.pth"
    )

    start_time = time.time()

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()

        train_correct = 0
        train_total = 0

        for images, labels, _ in train_loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits, _ = model(images)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # ---------------- VALIDATION ----------------
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():

            for images, labels, _ in val_loader:

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits, _ = model(images)

                loss = criterion(logits, labels)

                val_loss += loss.item()

                preds = logits.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train: {train_acc:.2f}% | "
            f"Val: {val_acc:.2f}% | "
            f"Loss: {val_loss:.4f}",
            end=""
        )

        # ---------- SAVE BEST ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("  <-- best saved", end="")

        # ---------- EARLY STOPPING ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
        else:
            early_counter += 1

        if early_counter >= EARLY_STOPPING_PATIENCE:
            print("  <-- early stopped")
            break

        print()

    elapsed = (time.time() - start_time) / 60

    print(
        f"\n{name} done | "
        f"Best Val Acc: {best_val_acc:.2f}% | "
        f"{elapsed:.1f} min"
    )

    return best_val_acc


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    results = {}

    for name, cls in MODELS.items():
        results[name] = train_one(name, cls)

    print(f"\n{'=' * 60}")
    print("ABLATION TRAINING COMPLETE")
    print(f"{'=' * 60}")

    for name, acc in results.items():
        print(f"{name:<15} {acc:.2f}%")
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fog_model import FogDetector
from fog_dataset import FogDataset, train_transform, val_transform
from config_loader import load_config

def main():
    cfg    = load_config()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FOG_DATASET_DIR = cfg["paths"]["dataset_dir"]
    SAVE_PATH       = cfg["paths"]["save_dir"]
    EPOCHS          = cfg["training"]["epochs"]
    LR              = cfg["training"]["lr"]
    BATCH_SIZE      = cfg["training"]["batch_size"]

    print(f"Device    : {DEVICE}")
    print(f"Dataset   : {FOG_DATASET_DIR}")
    print(f"Save to   : {SAVE_PATH}")

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ── DATASETS ────────────────────────────────────────
    train_ds = FogDataset(FOG_DATASET_DIR, "train", train_transform)
    val_ds   = FogDataset(FOG_DATASET_DIR, "val",   val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)  # 0 = safe on Windows
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── MODEL ───────────────────────────────────────────
    model     = FogDetector(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0

    # ── TRAINING LOOP ───────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # ── VALIDATION ──────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits, _ = model(images)
                loss = criterion(logits, labels)

                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_PATH, "fog_best.pth"))
            print(f"  -> Best saved ({val_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from fog_model import FogDetector
from fog_dataset import FogDataset, val_transform, CLASS_NAMES
from config_loader import load_config

def main():
    cfg       = load_config()
    SAVE_PATH = cfg["paths"]["save_dir"]
    FOG_DIR   = cfg["paths"]["dataset_dir"]
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FogDetector(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(
        f'{SAVE_PATH}/fog_best.pth', map_location=DEVICE))
    model.eval()

    # Test dataloader
    test_ds     = FogDataset(FOG_DIR, 'test', val_transform)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=4)  # workers fine inside main()

    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            logits, confidence = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            confs = confidence.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_confs.extend(confs)

    # Classification report
    print('── Classification Report ──')
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Fog Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/confusion_matrix.png', dpi=150)
    plt.show()

    # Confidence stats
    print(f'Mean confidence: {np.mean(all_confs):.4f}')
    print(f'Std  confidence: {np.std(all_confs):.4f}')


if __name__ == '__main__':
    main()
# Module 1 — Fog Detection

Classifies road images into 4 fog density levels and outputs a
continuous visibility score grounded in Koschmieder's atmospheric
scattering model.

## Architecture

- **Backbone:** ResNet-18 (ImageNet pretrained)
- **Classification head:** 512 → 128 → 4 classes
- **Confidence head:** 512 → 64 → 1 (sigmoid)
- **Dataset:** Foggy Cityscapes (clear + β = 0.005 / 0.01 / 0.02)

## Visibility Score Mapping

| Class  | β value | MOR      | Visibility Score |
|--------|---------|----------|-----------------|
| Clear  | —       | > 1000m  | 1.0             |
| Light  | 0.005   | ~600m    | 0.7             |
| Medium | 0.01    | ~300m    | 0.4             |
| Dense  | 0.02    | ~150m    | 0.1             |

## Results

| Class  | Precision | Recall | F1   |
|--------|-----------|--------|------|
| Clear  | 1.00      | 1.00   | 1.00 |
| Light  | 0.97      | 0.96   | 0.96 |
| Medium | 0.97      | 0.98   | 0.97 |
| Dense  | 0.99      | 0.99   | 0.99 |
| **Overall** | — | — | **0.98** |

## Setup

```bash
pip install -r ../requirements.txt
```

Edit `configs/config.yaml` → set `paths.dataset_dir` to your
local FogDataset folder.

## Run Order

```bash
# 1. Prepare dataset (run once)
python fog_segregator.py

# 2. Train
python train.py

# 3. Evaluate
python evaluate.py

# 4. Grad-CAM visualization
python gradcam_viz.py --image path/to/image.jpg

# 5. Single image inference
python inference.py --image path/to/image.jpg
```

## Output

```json
{
  "fog_class": "Dense",
  "visibility_score": 0.1,
  "confidence": 0.923,
  "class_probs": [0.02, 0.03, 0.05, 0.90]
}
```

Pretrained weights available via the project Google Drive link
(see root README).
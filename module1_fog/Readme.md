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

| Class  | β value | MOR     | Visibility Score |
|--------|---------|---------|-----------------|
| Clear  | —       | > 1000m | 1.0             |
| Light  | 0.005   | ~600m   | 0.7             |
| Medium | 0.01    | ~300m   | 0.4             |
| Dense  | 0.02    | ~150m   | 0.1             |

## Results — ResNet-18 (Selected Model)

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Clear      | 0.99      | 1.00   | 0.99 |
| Light      | 0.96      | 0.97   | 0.97 |
| Medium     | 0.96      | 0.96   | 0.96 |
| Dense      | 0.99      | 0.98   | 0.99 |
| **Overall**| —         | —      | **0.98** |

Test set: 6,100 images (1,525 per class)

## Ablation Study

All architectures trained with identical hyperparameters (30 epochs,
lr=1e-4, batch=32) on the same dataset split.

| Model           | Accuracy | F1-macro | F1-Light | F1-Medium | Inference |
|-----------------|----------|----------|----------|-----------|-----------|
| MobileNetV3     | 96.38%   | 0.9638   | 0.9541   | 0.9332    | 24.8ms    |
| EfficientNet-B0 | 97.28%   | 0.9728   | 0.9632   | 0.9502    | 49.1ms    |
| **ResNet-18**   | **97.66%**| **0.9765**| **0.9657**| **0.9589**| **13.1ms**|
| ConvNeXt-Tiny   | 98.07%   | 0.9807   | 0.9734   | 0.9638    | 30.8ms    |

**ResNet-18 selected** — fastest inference (13.1ms) with competitive
accuracy. ConvNeXt-Tiny achieves 0.41% higher accuracy but is 2.4×
slower. F1 on Light and Medium (the hardest classes due to visual
similarity) remains strong at 0.9657 and 0.9589.

## Setup

```bash
pip install -r ../requirements.txt

Copy .env.example to .env in the repo root and set your local paths:

FOG_DATASET_DIR=D:/path/to/your/FogDataset
SAVE_DIR=D:/path/to/your/fog_module_results
## Run Order

```bash
# 1. Verify config loads correctly
python config_loader.py

# 2. Train ResNet-18 (primary model)
python fog_train.py

# 3. Evaluate on test set
python fog_evaluate.py

# 4. Run ablation study (train all architectures)
python train_ablation.py

# 5. Evaluate ablation (comparison table + plot)
python evaluate_ablation.py

# 6. Grad-CAM visualization
python gradcam_viz.py --image path/to/image.jpg

# 7. Single image inference
python fog_inference.py --image path/to/image.jpg
## Output

```json
{
  "fog_class": "Dense",
  "visibility_score": 0.1,
  "continuous_visibility": 0.1,
  "confidence": 1.0,
  "class_probs": [0.0, 0.0, 0.0, 1.0]
}
```

## Pretrained Weights

| Model                | Accuracy | Link                                                                                              |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| ResNet-18 (selected) | 97.66%   | [Download](https://drive.google.com/file/d/1L822eoUH2-fynyxT6bh7qR3Z9Pvr7q61/view?usp=drive_link) |
| MobileNetV3          | 96.38%   | [Download](https://drive.google.com/file/d/1bdHK3qug_jH8dTmb0qwzoty6zE32qxf8/view?usp=sharing)                                                                      |
| EfficientNet-B0      | 97.28%   | [Download](https://drive.google.com/file/d/1-7s_T5FJ8XZ33U7yRsQsprJnJihEXi7X/view?usp=sharing)                                                                      |
| ConvNeXt-Tiny        | 98.07%   | [Download](https://drive.google.com/file/d/1Ab2gGs_HVTf6y24I8ycxDooBhP1UzfMG/view?usp=sharing)                                                                      |

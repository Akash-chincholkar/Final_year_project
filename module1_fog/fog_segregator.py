import os
import shutil
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────
CLEAR_ROOT = r"D:\Fog_Dataset\leftImg8bit_trainvaltest\leftImg8bit"
FOGGY_ROOT = r"D:\Fog_Dataset\leftImg8bit_foggy"
OUTPUT_DIR = r"D:\Fog_Dataset\FogDataset"

# Beta value → class label → visibility score
FOG_CLASSES = {
    "clear":  {"label": 0, "beta": None,  "vis_score": 1.0},
    "light":  {"label": 1, "beta": "0.005", "vis_score": 0.7},
    "medium": {"label": 2, "beta": "0.01",  "vis_score": 0.4},
    "dense":  {"label": 3, "beta": "0.02",  "vis_score": 0.1},
}

SPLITS = ["train", "val", "test"]
# ────────────────────────────────────────────────────────

# Create output folders
for split in SPLITS:
    for fog_class in FOG_CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, fog_class),
            exist_ok=True
        )

counts = defaultdict(int)

# ── COPY CLEAR IMAGES ────────────────────────────────────
print("Processing clear images...")
for split in SPLITS:
    split_dir = os.path.join(CLEAR_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    for city in os.listdir(split_dir):
        city_dir = os.path.join(split_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for img_file in os.listdir(city_dir):
            if not img_file.endswith(".png"):
                continue
            src = os.path.join(city_dir, img_file)
            dst = os.path.join(OUTPUT_DIR, split, "clear", img_file)
            shutil.copy2(src, dst)
            counts[f"{split}_clear"] += 1

# ── COPY FOGGY IMAGES ────────────────────────────────────
print("Processing foggy images...")
beta_to_class = {
    "0.005": "light",
    "0.01":  "medium",
    "0.02":  "dense",
}

for split in SPLITS:
    split_dir = os.path.join(FOGGY_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    for city in os.listdir(split_dir):
        city_dir = os.path.join(split_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for img_file in os.listdir(city_dir):
            if not img_file.endswith(".png"):
                continue
            # Extract beta from filename
            # e.g. aachen_000000_000019_leftImg8bit_foggy_beta_0.005.png
            for beta, fog_class in beta_to_class.items():
                if f"beta_{beta}" in img_file:
                    src = os.path.join(city_dir, img_file)
                    dst = os.path.join(
                        OUTPUT_DIR, split, fog_class, img_file
                    )
                    shutil.copy2(src, dst)
                    counts[f"{split}_{fog_class}"] += 1
                    break

# ── PRINT SUMMARY ────────────────────────────────────────
print(f"\n{'='*55}")
print("FOG DATASET SUMMARY")
print(f"{'='*55}")
for split in SPLITS:
    print(f"\n{split}:")
    for fog_class in FOG_CLASSES:
        key = f"{split}_{fog_class}"
        print(f"  {fog_class:<10} {counts[key]}")
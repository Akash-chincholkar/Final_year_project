import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CLASS_TO_IDX = {'clear': 0, 'light': 1, 'medium': 2, 'dense': 3}
IDX_TO_VIS   = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
CLASS_NAMES  = ['Clear', 'Light', 'Medium', 'Dense']

class FogDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir  = root_dir
        self.split     = split
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root_dir, split)
        for fog_class, label in CLASS_TO_IDX.items():
            class_dir = os.path.join(split_dir, fog_class)
            if not os.path.exists(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png','.jpg','.jpeg')):
                    self.samples.append({
                        'path':      os.path.join(class_dir, img_file),
                        'label':     label,
                        'fog_class': fog_class,
                        'vis_score': IDX_TO_VIS[label],
                    })
        print(f'  {split}: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image  = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, sample['label'], sample['vis_score']


# ── TRANSFORMS ──────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

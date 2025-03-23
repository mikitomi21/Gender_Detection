from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path

from model.classification import ClassificationType


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ClassificationType.get_names_of_all_types()
        self.class_to_idx = ClassificationType.get_class_to_idx()
        self.images = self._load_images()

    def _load_images(self) -> (Path, str):
        images = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for file in class_dir.iterdir():
                if str(file).endswith(".gitkeep"):
                    continue
                images.append((file, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")  # po co?
        if self.transform:
            image = self.transform(image)
        return image, label

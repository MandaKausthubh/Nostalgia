# datasets/base_dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split


class BaseDataset(Dataset):
    """
    Unified dataset base class for Nostalgia experiments.
    Provides consistent output and metadata structure.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        name: str = "BaseDataset",
        split: Optional[str] = None,
        train_split_ratio: float = 0.9
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.train_split_ratio = train_split_ratio
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.metadata: Dict[str, Any] = {
            "name": name,
            "num_classes": None
        }

        self.samples: list = []   # list of (path, label)
        self.classes: list = []   # all class names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.long),
            "paths": path
        }

    def _set_classes(self, classes: list):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.metadata["num_classes"] = len(classes)

    def _train_val_split(self, samples):
        """Split samples into train/val using ratio."""
        train, val = train_test_split(samples, test_size=1 - self.train_split_ratio, stratify=[s[1] for s in samples])
        return train, val

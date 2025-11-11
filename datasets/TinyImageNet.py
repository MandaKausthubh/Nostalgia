import os
import glob
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from .baseDataset import BaseDataset

class TinyImageNetDataset(BaseDataset):
    """
    TinyImageNet dataset loader.
    Compatible with the standard directory structure:
        tiny-imagenet-200/train/
        tiny-imagenet-200/val/
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        preload: bool = False,
    ):
        assert split in ("train", "val"), f"Invalid split: {split}"
        self.split = split
        super().__init__(
            data_root=os.path.join(root, split),
            transform=transform,
            target_transform=target_transform,
            preload=preload,
        )

    def _load_metadata(self):
        if self.split == "train":
            self._load_train_metadata()
        else:
            self._load_val_metadata()

    def _load_train_metadata(self):
        """
        Loads paths and labels for the training split.
        """
        class_dirs = sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}

        for cls_name in class_dirs:
            img_dir = os.path.join(self.data_root, cls_name, "images")
            for img_path in glob.glob(os.path.join(img_dir, "*.JPEG")):
                self.samples.append((img_path, class_to_idx[cls_name]))

        self.metadata["classes"] = class_dirs
        self.metadata["class_to_idx"] = class_to_idx

    def _load_val_metadata(self):
        """
        Loads paths and labels for the validation split.
        The labels are in val_annotations.txt.
        """
        ann_file = os.path.join(self.data_root, "val_annotations.txt")
        img_dir = os.path.join(self.data_root, "images")

        # Parse annotation file
        img_to_cls = {}
        with open(ann_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                img_name, cls_name = parts[0], parts[1]
                img_to_cls[img_name] = cls_name

        # Create mapping
        classes = sorted(set(img_to_cls.values()))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # Build sample list
        for img_name, cls_name in img_to_cls.items():
            img_path = os.path.join(img_dir, img_name)
            self.samples.append((img_path, class_to_idx[cls_name]))

        self.metadata["classes"] = classes
        self.metadata["class_to_idx"] = class_to_idx

    def _load_sample(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, label

import os
from glob import glob
from .baseDataset import BaseDataset

class ImageNetV2Dataset(BaseDataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__(root=root, transform=transform, name=f"ImageNetV2-{split}", split=split)
        self._load_split()

    def _get_split_dir(self):
        split_dirs = {
            "train": "ImageNetV2-matched-frequency",
            "val": "ImageNetV2-threshold0.7-val",
            "test": "ImageNetV2-top-0.7"
        }
        if self.split not in split_dirs:
            raise ValueError(f"Unknown split '{self.split}'. Expected one of {list(split_dirs.keys())}")
        return os.path.join(self.root, split_dirs[self.split])

    def _load_split(self):
        split_dir = self._get_split_dir()
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        self._set_classes(class_dirs)

        samples = []
        for cls in class_dirs:
            img_dir = os.path.join(split_dir, cls)
            imgs = glob(os.path.join(img_dir, "*.jpeg")) + glob(os.path.join(img_dir, "*.JPEG"))
            label = self.class_to_idx[cls]
            samples.extend([(img, label) for img in imgs])

        self.samples = samples

import os
from PIL import Image
from typing import Any, Tuple, List
from .baseDataset import BaseDataset

class ImageNetODataset(BaseDataset):
    """
    Dataset wrapper for ImageNet-O (out-of-distribution images for ImageNet-1k models).
    Assumes you have downloaded the images and a CSV (or list) specifying image filenames.
    Since this is a pure test/OOD set, there is no ‘train’ split.
    """

    def __init__(
        self,
        data_root: str,
        list_file: str,
        transform = None,
        target_transform = None,
        preload: bool = False,
    ):
        """
        Args:
            data_root: Root directory with the images.
            list_file: Path to a text/CSV file listing the image filenames (or image + dummy label).
            transform: Optional transform applied to the image.
            target_transform: Optional transform applied to the target (if you provide one).
            preload: If True, load all images into memory upfront.
        """
        self.list_file = list_file
        super().__init__(
            data_root = data_root,
            transform = transform,
            target_transform = target_transform,
            preload = preload,
        )

    def _load_metadata(self):
        # Read the list file
        self.samples: List[Tuple[str, int]] = []
        with open(self.list_file, 'r') as f:
            for line in f:
                fn = line.strip()
                label = -1
                self.samples.append((os.path.join(self.data_root, fn), label))
        # store metadata
        self.metadata["num_samples"] = len(self.samples)
        self.metadata["ood_label"] = label  # pyright: ignore[reportPossiblyUnboundVariable]

    def _load_sample(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return img, label



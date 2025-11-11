from typing import Any, Tuple
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset as IMNetV2Raw
from .baseDataset import BaseDataset


class ImageNetV2Dataset(BaseDataset):
    """
    ImageNetV2 dataset loader that wraps around `imagenet_v2` library.
    Supports standard variants like:
        - matched-frequency
        - threshold0.7
        - top-images

    Args:
        root: Path to cache directory or dataset root.
        variant: One of 'matched-frequency', 'threshold0.7', or 'top-images'.
        transform: Optional torchvision-style transform.
        target_transform: Optional label transform.
        preload: Whether to preload all images into memory.
    """

    def __init__(
        self,
        root: str,
        variant: str = "matched-frequency",
        transform=None,
        target_transform=None,
        preload: bool = False,
    ):
        self.variant = variant
        super().__init__(
            data_root=root,
            transform=transform,
            target_transform=target_transform,
            preload=preload,
        )

    def _load_metadata(self):
        """
        Loads the metadata using the official imagenet_v2 library.
        """
        # use the official imagenet_v2 pytorch wrapper
        self.raw_dataset = IMNetV2Raw(
            location=self.data_root,
            variant=self.variant,
            transform=self.transform
        )

        # store basic metadata
        self.samples = list(range(len(self.raw_dataset)))
        self.metadata["variant"] = self.variant
        self.metadata["num_classes"] = 1000
        self.metadata["class_names"] = None  # ImageNetV2 does not ship with labels by default

    def _load_sample(self, idx: int) -> Tuple[Any, int]:
        """
        Fetch a single sample from the wrapped ImageNetV2 dataset.
        """
        img, label = self.raw_dataset[idx]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        return img, label

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        data, label = self.samples[idx] if self.preload else self._load_sample(idx)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label


from imagenetv2_pytorch import ImageNetValDataset, ImageNetV2Dataset
from .baseDataset import BaseDataset
from sklearn.model_selection import train_test_split
from PIL import Image

# Create an index mapping for ImageNetV2 dataset splits (90:10 for train and validation)
def create_imagenetv2_split_indices(samples, train_ratio=0.9):
    train_samples, val_samples = train_test_split(
        samples, test_size=1 - train_ratio, stratify=[s[1] for s in samples]
    )
    return train_samples, val_samples

class ImageNetV2Wrapper(BaseDataset):
    def __init__(self, root: str, split: str = "train", transform=None, train_split_ratio: float = 0.9):
        """
        Wrapper around ImageNetValDataset and ImageNetV2Dataset from imagenetv2_pytorch.

        Args:
            root (str): Root directory for the dataset.
            split (str): Dataset split to use ("train", "val", or "test").
            transform: Transformations to apply to the dataset.
            train_split_ratio (float): Ratio for splitting ImageNetV2 into train and validation.
        """
        super().__init__(root=root, transform=transform, name=f"ImageNetV2-{split}", split=split, train_split_ratio=train_split_ratio)

        if split == "train" or split == "val":
            full_dataset = ImageNetV2Dataset(root=root, transform=self.transform)
            train_samples, val_samples = create_imagenetv2_split_indices(full_dataset.samples, train_ratio=train_split_ratio)
            self.samples = train_samples if split == "train" else val_samples
            self.classes = full_dataset.classes
        elif split == "test":
            test_dataset = ImageNetValDataset(root=root, transform=self.transform)
            self.samples = test_dataset.samples
            self.classes = test_dataset.classes
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

        self.metadata["num_classes"] = len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return {
            "images": self.transform(Image.open(path).convert("RGB")),
            "labels": label,
            "paths": path
        }

from imagenetv2_pytorch import ImageNetValDataset, ImageNetV2Dataset
from .baseDataset import BaseDataset
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import os

# Create and save randomized indices for train and validation splits
def save_split_indices(root, train_ratio=0.9, split_file="imagenetv2_splits.json"):
    full_dataset = ImageNetV2Dataset(location=root)
    full_samples = [(i, full_dataset[i][2]) for i in range(len(full_dataset))]
    train_samples, val_samples = train_test_split(
        full_samples, test_size=1 - train_ratio, stratify=[s[1] for s in full_samples]
    )
    split_indices = {
        "train": [s[0] for s in train_samples],
        "val": [s[0] for s in val_samples]
    }
    split_path = os.path.join(root, split_file)
    with open(split_path, "w") as f:
        json.dump(split_indices, f)
    print(f"Split indices saved to {split_path}")
    return split_indices

# Load split indices from file
def load_split_indices(root, split_file="imagenetv2_splits.json"):
    split_path = os.path.join(root, split_file)
    with open(split_path, "r") as f:
        return json.load(f)


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
            split_indices = load_split_indices(root)
            full_dataset = ImageNetV2Dataset(location=root, transform=self.transform)
            selected_indices = split_indices[split]
            self.samples = [(full_dataset[i]["paths"], full_dataset[i]["labels"]) for i in selected_indices]
            self.classes = full_dataset.classes
        elif split == "test":
            test_dataset = ImageNetValDataset(location=root, transform=self.transform)
            self.samples = [(test_dataset[i]["paths"], test_dataset[i]["labels"]) for i in range(len(test_dataset))]
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

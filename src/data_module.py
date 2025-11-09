"""
PyTorch Lightning DataModule for loading datasets.
"""
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any
import torchvision.transforms as transforms

# Import dataset classes
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from datasets import (
    ImageNet, ImageNetV2, ImageNetA, ImageNetR, 
    ImageNetSketch, ObjectNet
)


class ImageNetDataModule(pl.LightningDataModule):
    """DataModule for ImageNet and related datasets."""
    
    def __init__(
        self,
        dataset_name: str = "imagenet1k",
        data_dir: str = "~/data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        distributed: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.distributed = distributed
        self.kwargs = kwargs
        
        # Create preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        dataset_class = self._get_dataset_class()
        
        if stage == "fit" or stage is None:
            # Training dataset - these classes create their own loaders
            self.train_dataset_obj = dataset_class(
                preprocess=self.preprocess,
                location=self.data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                distributed=self.distributed,
                **self.kwargs
            )
            # Use the dataloader created by the dataset class
            self.train_loader = self.train_dataset_obj.train_loader
            
            # Validation dataset (use test split as val)
            self.val_dataset_obj = dataset_class(
                preprocess=self.preprocess,
                location=self.data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                distributed=self.distributed,
                **self.kwargs
            )
            self.val_loader = self.val_dataset_obj.test_loader
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset_obj = dataset_class(
                preprocess=self.preprocess,
                location=self.data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                distributed=self.distributed,
                **self.kwargs
            )
            self.test_loader = self.test_dataset_obj.test_loader
    
    def _get_dataset_class(self):
        """Get dataset class based on dataset_name."""
        dataset_map = {
            "imagenet1k": ImageNet,
            "imagenet": ImageNet,
            "imagenetv2": ImageNetV2,
            "imagenet-a": ImageNetA,
            "imagenet-r": ImageNetR,
            "imagenet-sketch": ImageNetSketch,
            "objectnet": ObjectNet,
        }
        
        if self.dataset_name.lower() not in dataset_map:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return dataset_map[self.dataset_name.lower()]
    
    def train_dataloader(self):
        """Return training dataloader."""
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            return self.train_loader
        # Fallback if dataset class doesn't create loader
        return DataLoader(
            self.train_dataset_obj.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if hasattr(self, 'val_loader') and self.val_loader is not None:
            return self.val_loader
        # Fallback if dataset class doesn't create loader
        return DataLoader(
            self.val_dataset_obj.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            return self.test_loader
        # Fallback if dataset class doesn't create loader
        return DataLoader(
            self.test_dataset_obj.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


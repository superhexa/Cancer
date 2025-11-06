from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


class ImagePreprocessor:
    
    def __init__(self, config):
        self._config = config
        self._train_transform = self._build_train_pipeline()
        self._eval_transform = self._build_eval_pipeline()
    
    def _build_train_pipeline(self) -> transforms.Compose:
        augmentation_config = self._config.get('augmentation', default={})
        
        pipeline = [
            transforms.RandomResizedCrop(
                self._config.get('dataset', 'image_size'),
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(
                p=augmentation_config.get('horizontal_flip', 0.5)
            ),
            transforms.RandomVerticalFlip(
                p=augmentation_config.get('vertical_flip', 0.3)
            ),
            transforms.RandomRotation(
                degrees=augmentation_config.get('rotation_degrees', 15)
            ),
            transforms.ColorJitter(
                brightness=augmentation_config.get('color_jitter', 0.2),
                contrast=augmentation_config.get('color_jitter', 0.2),
                saturation=augmentation_config.get('color_jitter', 0.2)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._config.get('normalization', 'mean'),
                std=self._config.get('normalization', 'std')
            )
        ]
        
        return transforms.Compose(pipeline)
    
    def _build_eval_pipeline(self) -> transforms.Compose:
        pipeline = [
            transforms.Resize(256),
            transforms.CenterCrop(self._config.get('dataset', 'image_size')),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._config.get('normalization', 'mean'),
                std=self._config.get('normalization', 'std')
            )
        ]
        
        return transforms.Compose(pipeline)
    
    def get_train_transform(self) -> transforms.Compose:
        return self._train_transform
    
    def get_eval_transform(self) -> transforms.Compose:
        return self._eval_transform


class DatasetFactory:
    
    @staticmethod
    def create_datasets(config) -> Tuple[Dataset, Dataset]:
        preprocessor = ImagePreprocessor(config)
        root_path = Path(config.get('dataset', 'root_path'))
        
        full_dataset = ImageFolder(
            root=root_path,
            transform=None
        )
        
        split_ratio = config.get('dataset', 'split_ratio')
        val_size = int(len(full_dataset) * split_ratio)
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(config.seed)
        train_indices, val_indices = random_split(
            range(len(full_dataset)),
            [train_size, val_size],
            generator=generator
        )
        
        train_dataset = DatasetSubset(
            full_dataset,
            train_indices.indices,
            preprocessor.get_train_transform()
        )
        
        val_dataset = DatasetSubset(
            full_dataset,
            val_indices.indices,
            preprocessor.get_eval_transform()
        )
        
        return train_dataset, val_dataset
    
    @staticmethod
    def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
        train_dataset, val_dataset = DatasetFactory.create_datasets(config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader


class DatasetSubset(Dataset):
    
    def __init__(self, base_dataset: Dataset, indices: list, transform: Optional[transforms.Compose]):
        self._base_dataset = base_dataset
        self._indices = indices
        self._transform = transform
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self._indices[idx]
        image, label = self._base_dataset.samples[original_idx]
        
        image = Image.open(image).convert('RGB')
        
        if self._transform:
            image = self._transform(image)
        
        return image, label


class SingleImageLoader:
    
    def __init__(self, config):
        self._preprocessor = ImagePreprocessor(config)
        self._transform = self._preprocessor.get_eval_transform()
    
    def load(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return self._transform(image).unsqueeze(0)

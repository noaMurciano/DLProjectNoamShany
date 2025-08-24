"""
Personal Food Taste Dataset Module

Data handling for personal food taste classification using HuggingFace Food101 dataset.

Author: Noam Shani
Course: 046211 Deep Learning, Technion

Attribution:
- HuggingFace Datasets: https://github.com/huggingface/datasets
- Food101 Dataset: Bossard, L., et al. (2014). Food-101–mining discriminative components with random forests. ECCV.
- PyTorch Dataset: https://pytorch.org/docs/stable/data.html
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Dict, Optional
import json
from datasets import load_dataset


class PersonalTasteDataset(Dataset):
    """
    Custom dataset for personal food taste preferences.
    
    Integrates personal taste labels with HuggingFace Food101 dataset images.
    
    Classes:
    - 0 (disgusting): Personal taste rating = disgusting
    - 1 (neutral): Personal taste rating = neutral  
    - 2 (tasty): Personal taste rating = tasty
    
    Attribution:
    - Based on PyTorch Dataset: https://pytorch.org/docs/stable/data.html
    - Uses HuggingFace Food101: https://huggingface.co/datasets/food101
    """
    
    def __init__(
        self, 
        csv_path: str,
        split: str = 'train',
        data_dir: str = 'data/food101_cache',
        transform: Optional[transforms.Compose] = None,
        include_neutral: bool = True
    ):
        """
        Args:
            csv_path: Path to personal labels CSV
            split: 'train', 'val', or 'test'
            data_dir: HuggingFace dataset cache directory
            transform: Image transforms
            include_neutral: Whether to include neutral class (for curriculum learning)
        """
        self.csv_path = csv_path
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.include_neutral = include_neutral
        
        # Load the HuggingFace Food101 dataset
        # Attribution: HuggingFace Datasets library
        self.hf_dataset = load_dataset("food101", cache_dir=data_dir)
        
        # Load our personal labels data
        self.df = pd.read_csv(csv_path)
        
        # Apply curriculum learning filter if needed
        if not include_neutral:
            # Stage A: Only extreme classes (disgusting vs tasty)
            self.df = self.df[self.df['class_id'] != 1].reset_index(drop=True)
            print(f"Curriculum Stage A: {len(self.df)} samples (no neutral class)")
        else:
            print(f"Full dataset: {len(self.df)} samples (all classes)")
            
        # Class distribution
        self.class_counts = self.df['class_id'].value_counts().sort_index()
        print(f"Class distribution: {dict(self.class_counts)}")
        
        # Store mappings
        self.class_names = {0: 'disgusting', 1: 'neutral', 2: 'tasty'}
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict, str]:
        """
        Get sample by index.
        
        Returns:
            image: Transformed image tensor [3, H, W]
            class_id: Class label (0, 1, or 2)
            metadata: Additional information dict
            image_id: Image identifier for debugging
        """
        row = self.df.iloc[idx]
        
        try:
            # Get HuggingFace dataset index
            hf_idx = int(row['hf_idx'])
            hf_split = row.get('hf_split', 'train')
            
            # Load image from HuggingFace dataset
            # Attribution: HuggingFace Food101 dataset access
            hf_sample = self.hf_dataset[hf_split][hf_idx]
            image = hf_sample['image'].convert('RGB')
            image_id = f"{hf_split}_{hf_idx}"
            
        except Exception as e:
            print(f"Error loading HF dataset sample {idx}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            image_id = f"dummy_{idx}"
        
        # Apply transforms
        # Attribution: torchvision transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare metadata
        metadata = {
            'original_label': row.get('label', 'unknown'),
            'food_category': row.get('food_category', 'unknown'),
            'confidence': row.get('confidence', 1.0)
        }
        
        return (
            image,
            int(row['class_id']),
            metadata,
            image_id
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate inverse frequency class weights for handling class imbalance.
        
        Attribution: Standard inverse frequency weighting approach
        Reference: He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE TKDE.
        """
        total = len(self.df)
        weights = []
        
        for class_id in [0, 1, 2]:
            count = self.class_counts.get(class_id, 0)
            if count > 0:
                weight = total / (3 * count)  # Inverse frequency
            else:
                weight = 0.0
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_weighted_sampler(self, oversample_minority: bool = True) -> WeightedRandomSampler:
        """
        Create weighted sampler to handle class imbalance during training.
        
        Attribution: PyTorch WeightedRandomSampler
        Reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        """
        class_weights = self.get_class_weights()
        
        # Create sample weights
        sample_weights = []
        for _, row in self.df.iterrows():
            class_id = int(row['class_id'])
            weight = class_weights[class_id].item()
            
            # Additional upsampling for minority classes
            if oversample_minority and class_id in [0, 2]:  # disgusting, tasty
                weight *= 1.5
                
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.df),
            replacement=True
        )


def get_personal_transforms(split: str, image_size: int = 224, augmentation_strength: str = 'medium') -> transforms.Compose:
    """
    Get appropriate transforms for personal dataset.
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size
        augmentation_strength: 'light', 'medium', or 'strong'
        
    Attribution:
    - torchvision transforms: https://pytorch.org/vision/stable/transforms.html
    - ImageNet normalization: Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database.
    """
    if split == 'train':
        # Training augmentations based on strength
        if augmentation_strength == 'light':
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        elif augmentation_strength == 'medium':
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:  # strong
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    else:  # val/test - no augmentation
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_personal_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 16,
    num_workers: int = 2,
    data_dir: str = 'data/food101_cache'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders for personal dataset.
    
    Args:
        train_csv, val_csv, test_csv: Paths to split CSV files
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        data_dir: HuggingFace dataset cache directory
        
    Returns:
        train_loader, val_loader, test_loader
        
    Attribution:
    - PyTorch DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    # Create datasets
    train_dataset = PersonalTasteDataset(
        csv_path=train_csv,
        split='train',
        data_dir=data_dir,
        transform=get_personal_transforms('train', augmentation_strength='medium')
    )
    
    val_dataset = PersonalTasteDataset(
        csv_path=val_csv,
        split='val',
        data_dir=data_dir,
        transform=get_personal_transforms('val')
    )
    
    test_dataset = PersonalTasteDataset(
        csv_path=test_csv,
        split='test',
        data_dir=data_dir,
        transform=get_personal_transforms('test')
    )
    
    # Create weighted sampler for training to handle class imbalance
    train_sampler = train_dataset.get_weighted_sampler(oversample_minority=True)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def create_3way_dataloaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience function to create dataloaders from config dictionary.
    
    Attribution: Custom implementation for 046211 project
    """
    return create_personal_dataloaders(
        train_csv=config['train_csv'],
        val_csv=config['val_csv'],
        test_csv=config['test_csv'],
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 2),
        data_dir=config.get('data_dir', 'data/food101_cache')
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing PersonalTasteDataset...")
    
    # Test dataset creation
    try:
        # This would need actual CSV files to work
        dataset = PersonalTasteDataset(
            csv_path="data/personal_train_3way.csv",
            transform=get_personal_transforms('train')
        )
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        print(f"Class weights: {dataset.get_class_weights()}")
        
    except Exception as e:
        print(f"⚠️ Test failed (expected without actual data): {e}")
        print("✅ Module imports successful")

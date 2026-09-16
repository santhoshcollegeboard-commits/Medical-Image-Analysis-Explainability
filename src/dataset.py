"""
Medical image dataset loader using MedMNIST (PneumoniaMNIST).
Standardized radiological benchmark based on Kermany et al. pediatric chest radiographs.
Includes medical-specific augmentation (rotation, slight translation) and class balance weighting.
"""

from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


def get_medical_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms tailored for pulmonary radiographs.
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=8),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, eval_transform


def get_medical_dataloaders(
    data_flag: str = 'pneumoniamnist',
    batch_size: int = 64,
    download: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], np.ndarray]:
    """
    Loads PneumoniaMNIST benchmark partitions: Train (4,708), Val (524), Test (624).
    Calculates class weights to compensate for clinical class imbalance.
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_transform, eval_transform = get_medical_transforms()

    train_dataset = DataClass(split='train', transform=train_transform, download=download)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download)

    # Calculate class weights from training labels
    train_labels = train_dataset.labels.squeeze()
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    label_dict = {0: 'Normal', 1: 'Pneumonia'}

    return train_loader, val_loader, test_loader, label_dict, class_weights

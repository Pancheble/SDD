"""
data/datasets.py
CIFAR-10 / ImageNet 데이터셋 및 DataLoader 빌더.
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def build_transforms(cfg, split: str = "train"):
    """
    학습/검증 데이터 증강 파이프라인 빌드.
    논문 Section 3.3.1: 두 뷰는 독립적 증강을 거침 (랜덤 크롭, 컬러 지터).
    """
    size = cfg.data.image_size

    if split == "train":
        transforms = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])
    else:
        transforms = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms


class TwoViewDataset(torch.utils.data.Dataset):
    """
    동일 이미지에 서로 다른 증강을 두 번 적용하여 (view1, view2) 쌍 반환.
    Student / Teacher에 각각 다른 뷰를 입력.
    """

    def __init__(self, base_dataset, transform):
        self.dataset   = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # PIL Image가 아닐 경우 대비 (이미 tensor인 경우 재변환 불필요)
        if not isinstance(img, torch.Tensor):
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            v1, v2 = img, img  # 이미 변환된 경우 그대로 사용
        return v1, v2, label


def build_dataloaders(cfg):
    """
    cfg에 따라 DataLoader 반환.
    Returns:
        train_loader, val_loader
    """
    dataset_name = cfg.data.dataset.lower()
    data_path    = cfg.data.data_path

    if dataset_name == "cifar10":
        base_train = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True
        )
        base_val = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True
        )
    elif dataset_name == "imagenet":
        base_train = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, "train")
        )
        base_val = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, "val")
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_tf = build_transforms(cfg, split="train")
    val_tf   = build_transforms(cfg, split="val")

    train_dataset = TwoViewDataset(base_train, train_tf)
    val_dataset   = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True,
        transform=val_tf
    ) if dataset_name == "cifar10" else torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, "val"), transform=val_tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader

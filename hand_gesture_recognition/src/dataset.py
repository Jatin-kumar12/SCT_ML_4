"""data/dataset.py — PyTorch datasets for image-based and landmark-based gesture data."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Image augmentation pipeline ──────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Image dataset (for CNN models) ───────────────────────────────────────────

class GestureImageDataset(Dataset):
    """
    Expects directory layout:
        dataset_path/
            thumbs_up/  image1.jpg  image2.jpg ...
            peace/      ...
            ...
    """

    def __init__(
        self,
        dataset_path: str,
        transform: Optional[A.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = Path(dataset_path)
        self.transform = transform

        # Auto-discover classes if not provided
        if class_names is None:
            class_names = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )
        self.class_names = class_names
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(class_names)
        }

        self.samples: List[Tuple[Path, int]] = []
        for cls_dir in self.root.iterdir():
            if cls_dir.is_dir() and cls_dir.name in self.class_to_idx:
                label = self.class_to_idx[cls_dir.name]
                for img_path in cls_dir.glob("*.[jp][pn]g"):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ── Landmark dataset (for MLP / LSTM models) ─────────────────────────────────

class GestureLandmarkDataset(Dataset):
    """
    Loads pre-extracted landmark feature vectors saved as a JSON file.

    Expected JSON format:
        [
          {"features": [0.1, 0.2, ...], "label": 0},
          ...
        ]

    Feature vector: 63-dim (21 landmarks × 3 coords), pre-normalized.
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.features = torch.tensor(
            [d["features"] for d in data], dtype=torch.float32
        )
        self.labels = torch.tensor(
            [d["label"] for d in data], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ── Sequence dataset (for LSTM models on video) ───────────────────────────────

class GestureSequenceDataset(Dataset):
    """
    Each sample is a sequence of landmark frames [T, 63].
    Useful for dynamic gestures (wave, swipe, circle).

    Directory layout:
        dataset_path/
            wave/   seq_001.npy  seq_002.npy ...
            swipe/  ...
    """

    def __init__(
        self,
        dataset_path: str,
        seq_length: int = 30,
        class_names: Optional[List[str]] = None,
    ):
        self.seq_length = seq_length
        self.root = Path(dataset_path)

        if class_names is None:
            class_names = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls_dir in self.root.iterdir():
            if cls_dir.is_dir() and cls_dir.name in self.class_to_idx:
                label = self.class_to_idx[cls_dir.name]
                for seq_path in cls_dir.glob("*.npy"):
                    self.samples.append((seq_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq_path, label = self.samples[idx]
        seq = np.load(seq_path).astype(np.float32)  # (T, 63)

        # Pad or truncate to fixed length
        if len(seq) >= self.seq_length:
            seq = seq[: self.seq_length]
        else:
            pad = np.zeros((self.seq_length - len(seq), seq.shape[1]), np.float32)
            seq = np.vstack([seq, pad])

        return torch.from_numpy(seq), label


# ── Data loaders factory ──────────────────────────────────────────────────────

def build_image_loaders(
    dataset_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for image data."""

    full_ds = GestureImageDataset(
        dataset_path, transform=get_train_transforms(image_size)
    )
    n = len(full_ds)
    n_train = int(n * train_split)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test]
    )
    # Apply val/test transforms (no augmentation)
    val_ds.dataset = GestureImageDataset(
        dataset_path, transform=get_val_transforms(image_size)
    )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, **kwargs),
        DataLoader(test_ds, **kwargs),
    )

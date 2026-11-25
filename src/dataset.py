"""
Brain MRI Dataset Pipeline for LGG Brain Tumor Segmentation
Pairs MRI slices (.tif) with corresponding ground truth masks (_mask.tif)
and normalizes images to 256x256 using Albumentations.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def find_mri_pairs(data_dir: str = "./data/kaggle_3m", cache_file: str = "./data/pairs_cache.csv") -> pd.DataFrame:
    """
    Scans the dataset directory and pairs each MRI slice with its corresponding mask.
    Computes whether each slice contains tumor pixels and caches results to CSV.
    """
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file)
            if len(cached_df) > 0 and {"image_path", "mask_path", "has_tumor"}.issubset(cached_df.columns):
                # Verify first file exists
                if os.path.exists(cached_df.iloc[0]["image_path"]):
                    return cached_df
        except Exception:
            pass

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    records = []
    for file_path in data_path.rglob("*.tif"):
        file_name = file_path.name
        if file_name.endswith("_mask.tif"):
            continue

        mask_name = file_name.replace(".tif", "_mask.tif")
        mask_path = file_path.parent / mask_name

        if mask_path.exists():
            patient_id = file_path.parent.name
            # Check if mask has tumor pixels
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            has_tumor = int(mask_img.max() > 0) if mask_img is not None else 0

            records.append({
                "patient_id": patient_id,
                "image_path": str(file_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "filename": file_name,
                "has_tumor": has_tumor,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise ValueError(f"No valid MRI slice and mask pairs found in '{data_dir}'.")

    df = df.sort_values(by=["patient_id", "filename"]).reset_index(drop=True)
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_csv(cache_file, index=False)
    except Exception:
        pass
    return df


def get_transforms(img_size: int = 256, is_train: bool = True) -> A.Compose:
    """
    Returns data transformation & normalization pipeline.
    Resizes images and masks to (img_size, img_size) and applies standard normalization.
    """
    if is_train:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for paired Brain MRI slices and segmentation masks.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        # Read RGB MRI image
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read Grayscale Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask at {mask_path}")
        # Binarize mask: positive pixels = 1.0, background = 0.0
        mask = (mask > 0).astype(np.float32)

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).float()

        # Ensure mask is [1, H, W]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": img_path,
            "mask_path": mask_path,
            "patient_id": row["patient_id"],
        }


def get_dataloaders(
    data_dir: str = "./data/kaggle_3m",
    batch_size: int = 16,
    val_split: float = 0.2,
    img_size: int = 256,
    random_state: int = 42,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """
    Builds paired DataFrame, splits into train/validation, and returns PyTorch DataLoaders.
    """
    df = find_mri_pairs(data_dir=data_dir)

    # Balanced sampling when max_samples is specified
    if max_samples is not None and max_samples < len(df):
        pos_df = df[df["has_tumor"] == 1]
        neg_df = df[df["has_tumor"] == 0]
        n_pos = min(len(pos_df), max_samples // 2)
        n_neg = min(len(neg_df), max_samples - n_pos)
        sampled_pos = pos_df.sample(n=n_pos, random_state=random_state)
        sampled_neg = neg_df.sample(n=n_neg, random_state=random_state)
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Patient-stratified or random train/val split
    patients = np.array(df["patient_id"].unique())
    rng = np.random.default_rng(random_state)
    rng.shuffle(patients)
    split_idx = int(len(patients) * (1.0 - val_split))
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)

    # Fallback if split leaves val_df too small
    if len(val_df) == 0:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        split_idx = int(len(df) * (1.0 - val_split))
        train_df = df.iloc[indices[:split_idx]].reset_index(drop=True)
        val_df = df.iloc[indices[split_idx:]].reset_index(drop=True)

    train_transform = get_transforms(img_size=img_size, is_train=True)
    val_transform = get_transforms(img_size=img_size, is_train=False)

    train_dataset = BrainMRIDataset(train_df, transform=train_transform)
    val_dataset = BrainMRIDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, train_df, val_df

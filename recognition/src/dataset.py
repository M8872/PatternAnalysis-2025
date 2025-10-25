"""
Dataset and DataLoader utilities for simple Alzheimer's (AD vs CN) classification.

This file converts 3D MRI NIfTI volumes into a small set of 2D axial slices
that we can feed into a 2D image model (ConvNeXt-Tiny). The goal is to keep
everything very simple and very commented so beginners can follow.

High-level steps:
1) scan_adni(root): walk the directory and find NIfTI files with AD/CN labels
2) MRISliceDataset: read a specific axial slice from each volume, turn into PIL
3) create_dataloaders(...): split into train/val/test and wrap with DataLoader

We use ImageNet normalization because the model is pretrained on ImageNet.
We also replicate the single grayscale slice to 3 channels (RGB) to match
the expected input shape for ConvNeXt/ResNet-like backbones.
"""

# ========= IMPORTS =========
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


# ========= CONSTANTS AND SIMPLE HELPERS =========

AD_LABEL = 1  # Alzheimer's Disease (positive class)
CN_LABEL = 0  # Cognitively Normal (control class)

# ImageNet normalization stats for 3-channel inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_ad_path(path: str) -> bool:
    """Return True if a directory path indicates the AD class.

    We look for a whole path token named 'AD' (case-insensitive).
    """
    return bool(re.search(r"(^|/)(AD)(/|$)", path, flags=re.IGNORECASE))


def _is_cn_path(path: str) -> bool:
    """Return True if the path corresponds to the cognitively normal class.

    Supports both 'CN' and 'NC' folder tokens because some datasets may name
    the control group 'NC'. We only match whole path tokens (delimited by '/'
    or start/end) to avoid accidental matches like 'scan'.
    """
    if re.search(r"(^|/)(CN)(/|$)", path, flags=re.IGNORECASE):
        return True
    if re.search(r"(^|/)(NC)(/|$)", path, flags=re.IGNORECASE):
        return True
    return False


def scan_adni(root: str) -> List[Tuple[str, int]]:
    """Recursively scan a root folder and build (path, label) pairs.

    We assume a very simple folder structure where the path contains either
    'AD' or 'CN' somewhere in the directory names. All .nii and .nii.gz
    files are considered MRI volumes. This function does not load data; it
    just collects paths and their inferred labels.
    """
    pairs: List[Tuple[str, int]] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith((".nii", ".nii.gz")):
                fpath = os.path.join(dirpath, fname)
                if _is_ad_path(dirpath):
                    pairs.append((fpath, AD_LABEL))
                elif _is_cn_path(dirpath):
                    pairs.append((fpath, CN_LABEL))
    return pairs


def _normalize_volume_to_unit_range(volume: np.ndarray) -> np.ndarray:
    """Normalize a 3D volume to the [0, 1] range.

    We use 1st and 99th percentiles to be robust to outliers, then clip.
    This keeps intensities consistent across scans and helps training.
    """
    v = volume.astype(np.float32)
    v = np.nan_to_num(v)
    v_min, v_max = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min)
    v = np.clip(v, 0.0, 1.0)
    return v


def _slice_indices_centered(num_slices: int, k: int) -> List[int]:
    """Return k axial slice indices centered around the middle of the volume.

    If the volume has fewer than k slices, we pad by repeating the last index
    so that every volume contributes the same number of slices.
    """
    if num_slices <= 0:
        return []
    center = num_slices // 2
    half = max(k // 2, 1)
    indices = list(range(max(0, center - half), min(num_slices, center - half + k)))
    # Pad if volume has fewer than k slices
    while len(indices) < k and indices:
        indices.append(indices[-1])
    return indices[:k]


@dataclass
class MRISliceSpec:
    """Tiny struct that describes exactly one (volume, slice) example.

    - volume_path: where the NIfTI file lives on disk
    - label: integer class (AD=1 or CN=0)
    - slice_index: which axial slice to take from the 3D volume
    """
    volume_path: str
    label: int
    slice_index: int


class MRISliceDataset(Dataset):
    """A very simple 2D slice dataset built from 3D NIfTI MRI volumes.

    What it does for each item:
    1) Loads the NIfTI file
    2) Picks one axial slice (by slice_index)
    3) Converts it to a PIL grayscale image (so torchvision can process it)
    4) Applies basic transforms (resize, to tensor)
    5) Replicates channels to get 3xHxW (ConvNeXt expects 3 channels)
    6) Normalizes using ImageNet mean/std (common for pretrained models)
    """

    def __init__(
        self,
        file_label_pairs: Sequence[Tuple[str, int]],
        slices_per_volume: int = 8,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.file_label_pairs = list(file_label_pairs)
        self.slices_per_volume = max(1, int(slices_per_volume))
        # Default transform: resize + to tensor only. We normalize AFTER we
        # replicate to 3 channels using ImageNet stats (see __getitem__).
        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),  # Grayscale -> [1, H, W], values in [0,1]
            ]
        )

        # Precompute which slices we will use for each volume for determinism.
        # This makes __getitem__ simple and ensures we pick the same slices
        # every run (important for reproducibility).
        self.index_map: List[MRISliceSpec] = []
        for path, label in self.file_label_pairs:
            try:
                img = nib.load(path)
                data = img.get_fdata()
                num_slices = data.shape[2] if data.ndim >= 3 else 0
                indices = _slice_indices_centered(num_slices, self.slices_per_volume)
                for s in indices:
                    self.index_map.append(MRISliceSpec(path, label, s))
            except Exception:
                # Skip corrupted volumes quietly (student-friendly). In a
                # production system we would log or raise, but here we want
                # the script to keep going for learning purposes.
                continue

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Look up which (volume, slice) we need for this index.
        spec = self.index_map[idx]

        # Load the NIfTI volume from disk and get the data as a numpy array.
        img = nib.load(spec.volume_path)
        vol = img.get_fdata()

        # Normalize intensities to [0, 1] so the network sees consistent values.
        vol = _normalize_volume_to_unit_range(vol)

        # Use axial plane (H, W, D), select a specific slice.
        if vol.ndim == 3:
            slice_2d = vol[:, :, spec.slice_index]
        else:
            # Unexpected extra channels/time dimension – take the first channel.
            slice_2d = vol[:, :, spec.slice_index, 0]

        # Convert to PIL grayscale image for torchvision transforms.
        slice_uint8 = (slice_2d * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(slice_uint8, mode="L")

        x = self.transform(pil_img)  # [1, H, W]
        # Replicate to 3 channels for ConvNeXt/ResNet-style backbones.
        x = x.repeat(3, 1, 1)  # [3, H, W]
        # Apply ImageNet normalization after replication.
        x = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(x)
        y = int(spec.label)
        return x, y


def _stratified_split(
    pairs: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Stratified volume-level split by label (AD vs CN).

    We split AD and CN groups separately, then combine them. This keeps the
    class balance similar across train/val/test. We also shuffle with a fixed
    seed so that runs are deterministic.
    """
    rng = np.random.RandomState(seed)
    ad = [p for p in pairs if p[1] == AD_LABEL]
    cn = [p for p in pairs if p[1] == CN_LABEL]

    def split_one(group: List[Tuple[str, int]]):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return [group[i] for i in train_idx], [group[i] for i in val_idx], [group[i] for i in test_idx]

    ad_tr, ad_va, ad_te = split_one(ad)
    cn_tr, cn_va, cn_te = split_one(cn)

    train = ad_tr + cn_tr
    val = ad_va + cn_va
    test = ad_te + cn_te
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def create_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    slices_per_volume: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Scan ADNI-like folders, split, build datasets, and return DataLoaders.

    The loaders include common DataLoader options like pin_memory and
    persistent_workers to be nice on GPU systems. The batch sizes and slice
    counts are all configurable via function arguments (or CLI in train.py).
    """
    pairs = scan_adni(root)
    if len(pairs) == 0:
        raise FileNotFoundError(
            f"No NIfTI files found under '{root}'. Expected 'AD' and 'CN' folders."
        )

    train_pairs, val_pairs, test_pairs = _stratified_split(pairs, seed=seed)

    # Keep transforms simple; normalization is applied in __getitem__ after we
    # convert to 3 channels, so we omit normalization here.
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    train_ds = MRISliceDataset(train_pairs, slices_per_volume=slices_per_volume, transform=transform)
    val_ds = MRISliceDataset(val_pairs, slices_per_volume=max(2, slices_per_volume // 2), transform=transform)
    test_ds = MRISliceDataset(test_pairs, slices_per_volume=max(2, slices_per_volume // 2), transform=transform)

    # CUDA pin_memory speeds up host->device transfers; persistent_workers keeps
    # worker processes alive between epochs for a small speed boost.
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent
    )
    return train_loader, val_loader, test_loader



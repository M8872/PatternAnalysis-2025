"""
Subject-level dataset utilities for AD vs NC MRI slice classification.

This module scans JPEG slices, groups them by subject ID (prefix before the
first underscore), performs subject-level train/val/test splits, and builds
PyTorch Dataset/DataLoader objects with safe transforms (augment train only).
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Label constants (kept consistent with the rest of the project)
AD_LABEL = 1
NC_LABEL = 0

# Folder names we recognise for each label
LABELS_BY_DIRNAME = {"AD": AD_LABEL, "NC": NC_LABEL, "CN": NC_LABEL}
LABEL_TO_NAME = {AD_LABEL: "AD", NC_LABEL: "NC"}

# Image extensions we will consider when scanning the folders.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Normalisation constants (ImageNet-style, works well for RGB CNN backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class SubjectRecord:
    """All slices belonging to a single subject."""

    subject_id: str
    label: int
    image_paths: List[str]


@dataclass(frozen=True)
class ImageSample:
    """Single slice and its metadata."""

    path: str
    label: int
    subject_id: str


def _iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield image files under a directory (recursively)."""
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _extract_subject_id(filename: str) -> str:
    """Return the subject prefix before the first underscore."""
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _index_subjects(source_dir: Path) -> List[SubjectRecord]:
    """Group all slices under source_dir by subject and label."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {source_dir}")

    subjects_per_label: Dict[int, Dict[str, List[str]]] = {}
    subject_label_map: Dict[str, int] = {}
    found_any = False

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = LABELS_BY_DIRNAME.get(class_dir.name.upper())
        if label is None:
            continue
        found_any = True
        subject_bucket = subjects_per_label.setdefault(label, {})
        for img_path in _iter_image_files(class_dir):
            subject_id = _extract_subject_id(img_path.name)
            prev_label = subject_label_map.get(subject_id)
            if prev_label is not None and prev_label != label:
                raise ValueError(
                    f"Subject '{subject_id}' appears under multiple labels "
                    f"('{LABEL_TO_NAME.get(prev_label, prev_label)}' and '{class_dir.name}')."
                )
            subject_label_map.setdefault(subject_id, label)
            paths = subject_bucket.setdefault(subject_id, [])
            paths.append(str(img_path.resolve()))

    if not found_any or not subjects_per_label:
        raise FileNotFoundError(
            f"No AD/NC folders with JPEG slices found under {source_dir}."
        )

    records: List[SubjectRecord] = []
    for label, subjects in subjects_per_label.items():
        for subject_id, image_paths in subjects.items():
            image_paths.sort()
            records.append(SubjectRecord(subject_id=subject_id, label=label, image_paths=image_paths))

    records.sort(key=lambda r: (r.label, r.subject_id))
    return records


def _compute_split_sizes(num_subjects: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    """Convert ratios into exact subject counts."""
    if num_subjects == 0:
        return 0, 0, 0

    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("train_ratio must be in (0, 1].")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError("val_ratio must be in [0, 1].")
    if train_ratio + val_ratio > 1.0 + 1e-6:
        raise ValueError("train_ratio + val_ratio must be <= 1.0.")

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    ratios = [train_ratio, val_ratio, test_ratio]
    counts = [int(round(r * num_subjects)) for r in ratios]
    diff = num_subjects - sum(counts)

    # Balance rounding errors by distributing the remainder.
    while diff != 0:
        if diff > 0:
            idx = min(range(3), key=lambda i: counts[i])
            counts[idx] += 1
            diff -= 1
        else:
            idx = max(range(3), key=lambda i: counts[i])
            if counts[idx] == 0:
                break
            counts[idx] -= 1
            diff += 1

    # Ensure we have at least one subject for training when data exists.
    if counts[0] == 0:
        idx = 1 if counts[1] >= counts[2] else 2
        if counts[idx] > 0:
            counts[idx] -= 1
            counts[0] = 1
        else:
            counts[0] = 1

    counts = [max(0, c) for c in counts]
    total = sum(counts)
    if total != num_subjects:
        counts[0] += num_subjects - total
    return counts[0], counts[1], counts[2]


def _split_subjects(
    records: Sequence[SubjectRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Split subjects stratified by label."""
    rng = random.Random(seed)
    by_label: Dict[int, List[SubjectRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    train_set: List[SubjectRecord] = []
    val_set: List[SubjectRecord] = []
    test_set: List[SubjectRecord] = []

    for label, subjects in by_label.items():
        subjects_copy = subjects.copy()
        rng.shuffle(subjects_copy)
        n_train, n_val, n_test = _compute_split_sizes(len(subjects_copy), train_ratio, val_ratio)
        train_set.extend(subjects_copy[:n_train])
        val_set.extend(subjects_copy[n_train : n_train + n_val])
        test_set.extend(subjects_copy[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return train_set, val_set, test_set


def _expand_records(records: Sequence[SubjectRecord]) -> List[ImageSample]:
    """Flatten SubjectRecord objects into per-slice ImageSample entries."""
    samples: List[ImageSample] = []
    for record in records:
        for path in record.image_paths:
            samples.append(ImageSample(path=path, label=record.label, subject_id=record.subject_id))
    samples.sort(key=lambda s: (s.subject_id, s.path))
    return samples


def _summarise_split(split: Sequence[ImageSample]) -> Tuple[int, int]:
    """Return number of unique subjects and total slices inside a split."""
    unique_subjects = {sample.subject_id for sample in split}
    return len(unique_subjects), len(split)


def _print_split_summary(splits: Dict[str, Sequence[ImageSample]], seed: int) -> None:
    """Log a short human-readable summary to confirm subject-level splitting."""
    print(f"Subject-level split summary (seed={seed}):")
    for name in ("train", "val", "test"):
        split = splits.get(name, ())
        n_subjects, n_slices = _summarise_split(split)
        label_subjects: Dict[int, set] = {}
        for sample in split:
            label_subjects.setdefault(sample.label, set()).add(sample.subject_id)
        if label_subjects:
            label_summary = ", ".join(
                f"{LABEL_TO_NAME.get(label, str(label))}:{len(subjects)}"
                for label, subjects in sorted(label_subjects.items())
            )
        else:
            label_summary = "none"
        print(
            f"  {name:<5} -> {n_subjects:3d} unique subjects ({label_summary}); "
            f"{n_slices:4d} slices"
        )


class SubjectImageDataset(Dataset):
    """Torch Dataset for subject-level image samples."""

    def __init__(self, samples: Sequence[ImageSample], transform: Optional[T.Compose] = None) -> None:
        self.samples: List[ImageSample] = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        try:
            with Image.open(sample.path) as img:
                image = img.convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing image file: {sample.path}") from exc

        if self.transform:
            image = self.transform(image)
        return image, sample.label


def materialise_split_folders(
    splits: Dict[str, Sequence[ImageSample]],
    output_root: Union[str, Path],
    copy_files: bool = False,
) -> None:
    """Optional helper to mirror the split into dataset_split/train|val|test folders."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        for sample in samples:
            class_dir = LABEL_TO_NAME.get(sample.label, str(sample.label))
            dest_dir = root / split_name / class_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(sample.path)
            filename = src_path.name
            dest_path = dest_dir / filename
            if dest_path.exists():
                dest_path = dest_dir / f"{sample.subject_id}_{filename}"
                if dest_path.exists():
                    # Skip duplicates if they already exist.
                    continue

            if copy_files:
                shutil.copy2(src_path, dest_path)
            else:
                try:
                    os.symlink(src_path, dest_path)
                except (FileExistsError, OSError):
                    # Fallback to copy if symlinks are unsupported (Windows without perms).
                    shutil.copy2(src_path, dest_path)


def create_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_output_dir: Optional[Union[str, Path]] = None,
    copy_split: bool = False,
    return_splits: bool = False,
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]]],
]:
    """
    Build DataLoaders with subject-level splits to avoid data leakage.

    Parameters
    ----------
    data_root:
        Path to the dataset root (expects train/AD/ and train/NC/ inside).
    train_ratio / val_ratio:
        Fractions of subjects assigned to train and validation splits.
        The remainder is used for the test split.
    split_output_dir:
        Optional folder to populate with the split (symlinks by default).
    copy_split:
        When True, copy files instead of creating symlinks.
    return_splits:
        When True, also return {split_name: [file paths]} for inspection.
    """
    root = Path(data_root)
    train_dir = root / "train"
    records = _index_subjects(train_dir)

    train_records, val_records, test_records = _split_subjects(records, train_ratio, val_ratio, seed)
    train_samples = _expand_records(train_records)
    val_samples = _expand_records(val_records)
    test_samples = _expand_records(test_records)

    if not train_samples:
        raise ValueError("Subject split produced an empty training set. Adjust the ratios or check the data.")

    split_samples: Dict[str, List[ImageSample]] = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    _print_split_summary(split_samples, seed=seed)

    if split_output_dir is not None:
        materialise_split_folders(split_samples, split_output_dir, copy_files=copy_split)

    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = SubjectImageDataset(train_samples, transform=train_transform)
    val_dataset = SubjectImageDataset(val_samples, transform=eval_transform)
    test_dataset = SubjectImageDataset(test_samples, transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    if return_splits:
        split_paths = {
            split_name: [sample.path for sample in samples]
            for split_name, samples in split_samples.items()
        }
        return train_loader, val_loader, test_loader, split_paths

    return train_loader, val_loader, test_loader


__all__ = [
    "AD_LABEL",
    "NC_LABEL",
    "ImageSample",
    "SubjectImageDataset",
    "create_dataloaders",
    "materialise_split_folders",
]


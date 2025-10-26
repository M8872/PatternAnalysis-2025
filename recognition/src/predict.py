"""
Simple prediction/evaluation script for the test set.

This file loads the latest checkpoint from runs/checkpoints, builds the same
model used in training, and runs it on the test DataLoader. Results are saved
as a very simple CSV-like text file so it is easy to open and inspect.

We deliberately keep this minimal and very commented.
"""

# ========= IMPORTS =========
import argparse
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import create_dataloaders  # type: ignore
from .modules import build_convnext_tiny  # type: ignore
from .utils import (  # type: ignore
    ensure_dir,
    find_latest_checkpoint,
    get_device,
    load_checkpoint,
    set_seed,
)


# ========= MAIN: LOAD CHECKPOINT, RUN TEST, SAVE PREDICTIONS =========

def main() -> None:
    # --------- ARG PARSER ---------
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-Tiny on ADNI test set")
    default_data = os.environ.get("DATA_DIR", "data")
    parser.add_argument("--data-root", type=str, default=default_data)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--slices-per-volume", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=str, default="runs/checkpoints")
    parser.add_argument("--predictions-dir", type=str, default="runs/predictions")
    args = parser.parse_args()

    # --------- SETUP SEED + DEVICE ---------
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.predictions_dir)

    # --------- BUILD ONLY THE TEST DATALOADER ---------
    # We re-use the same helper so transforms/data config match training.
    _, _, test_loader = create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slices_per_volume=args.slices_per_volume,
        seed=args.seed,
    )

    # --------- BUILD MODEL AND LOAD LATEST CHECKPOINT ---------
    model = build_convnext_tiny(num_classes=2).to(device)
    latest = find_latest_checkpoint(args.checkpoints_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoints_dir}. Train first.")
    load_checkpoint(latest, model, optimizer=None)
    print(f"🕒 Loaded checkpoint: {latest}")

    # --------- RUN INFERENCE ON THE TEST SET ---------
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="🔮 Predict", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    # --------- SAVE A VERY SIMPLE TEXT FILE WITH PREDICTIONS ---------
    # This is intentionally a tiny format so you can open it with any editor.
    save_path = os.path.join(args.predictions_dir, "predictions.txt")
    with open(save_path, "w") as f:
        f.write("index,target,pred\n")
        for i, (t, p) in enumerate(zip(all_targets, all_preds)):
            f.write(f"{i},{t},{p}\n")
    print(f"📝 Saved predictions to: {save_path}")


if __name__ == "__main__":
    main()



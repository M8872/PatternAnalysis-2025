"""
Simple training script for Alzheimer's detection (AD vs CN) using ConvNeXt-Tiny.

This file is intentionally written in a very beginner-friendly style:
- Lots of comments explaining each step in plain English
- Clear section headers that show the high-level flow
- No clever tricks; we keep everything straightforward

Overall flow of this script:
1) Parse command-line arguments (like where the data is and how many epochs)
2) Set random seeds and pick the best device (GPU if available)
3) Build the DataLoaders (train/val/test) from ADNI-like folder structure
4) Build the model, choose the loss, and set up the optimizer
5) (Optional) Resume from a previous checkpoint
6) Train for N epochs while tracking loss and accuracy
7) Save checkpoints and simple plots/metrics after every epoch
"""

# ========= IMPORTS (standard libs first, then third-party, then local) =========
import argparse
import os
import sys
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ========= LOCAL IMPORTS =========
from .dataset import create_dataloaders  # type: ignore
from .modules import build_convnext_tiny  # type: ignore
from .utils import (  # type: ignore
    ensure_dir,
    find_latest_checkpoint,
    get_device,
    load_checkpoint,
    set_seed,
    save_checkpoints,
    update_plots_csv,
    write_metrics,
)


# ========= HELPER: ONE EPOCH OF TRAIN OR VAL (no side effects) =========

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
    capture_examples_max: int = 0,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Run one full pass (epoch) over a DataLoader.

    We use the same function for training and validation. The only difference
    is whether gradients are turned on or off. This keeps the code short.

    Returns a tuple: (avg_loss, accuracy_percent, examples)
    - avg_loss: simple mean loss over all samples in this epoch
    - accuracy_percent: accuracy in percent (0.0 to 100.0)
    - examples: a few {"target": int, "pred": int} pairs (for metrics.json),
      only captured during validation when capture_examples_max > 0
    """
    # Put the model in the correct mode: training enables dropout/BN updates,
    # evaluation turns them off. This is important for proper behavior.
    if train:
        model.train()
    else:
        model.eval()

    # We will keep track of running sums so we can compute averages at the end.
    running_loss: float = 0.0
    running_correct: int = 0
    running_total: int = 0
    captured_examples: List[Dict[str, Any]] = []

    # tqdm creates a nice progress bar in the terminal.
    pbar = tqdm(loader, desc=("🛠️ Train" if train else "🧪 Val"), leave=False)
    for images, targets in pbar:
        # Move tensors to the selected device (GPU if available; else CPU).
        images = images.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)

        # Only compute gradients during training; during validation we save time
        # and memory by turning them off.
        with torch.set_grad_enabled(train):
            # Forward pass: model takes in images and returns raw logits.
            logits = model(images)
            # Criterion is CrossEntropy for 2-class classification (AD vs CN).
            loss = criterion(logits, targets)

        if train:
            # Standard training step: clear old grads -> backprop -> update.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # --------- RUNNING STATS (accumulate for averages) ---------
        # Multiply loss by batch size so we can divide by total samples later.
        running_loss += float(loss.item()) * images.size(0)
        # Count how many predictions matched the targets in this batch.
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        # Track total number of seen samples.
        running_total += images.size(0)

        # Optionally capture a handful of example predictions so we can write
        # them into metrics.json for easy inspection later.
        if (not train) and capture_examples_max > 0 and len(captured_examples) < capture_examples_max:
            preds = logits.argmax(dim=1).detach().cpu().tolist()
            targs = targets.detach().cpu().tolist()
            for t, p in zip(targs, preds):
                if len(captured_examples) >= capture_examples_max:
                    break
                captured_examples.append({"target": int(t), "pred": int(p)})

        # Update the progress bar so we can observe loss/acc as the epoch runs.
        pbar.set_postfix({
            "loss": f"{(running_loss / max(running_total,1)):.4f}",
            "acc": f"{(100.0 * running_correct / max(running_total,1)):.2f}%",
        })

    # Compute final averages for this epoch.
    avg_loss = running_loss / max(running_total, 1)
    acc = 100.0 * running_correct / max(running_total, 1)
    return avg_loss, acc, captured_examples


# ========= MAIN: ARGUMENTS, SETUP, TRAINING LOOP, SAVING =========
def main() -> None:
    # --------- ARG PARSER (read settings from command line) ---------
    parser = argparse.ArgumentParser(description="Alzheimer's AD vs CN classification with ConvNeXt-Tiny")
    default_data = os.environ.get("DATA_DIR", "data")
    parser.add_argument("--data-root", type=str, default=default_data, help="Path with AD/ and CN/ folders (or set DATA_DIR)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--slices-per-volume", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoints-dir", type=str, default="runs/checkpoints")
    parser.add_argument("--plots-dir", type=str, default="runs/metrics")
    args = parser.parse_args()

    # ========= SETUP SEED + DEVICE (determinism and GPU/CPU choice) =========
    # Set all seeds so training is reproducible (same results every run).
    set_seed(args.seed)
    # Pick the best device and tell the user (cuda if available, else cpu).
    device = get_device()

    # ========= CREATE OUTPUT FOLDERS (for checkpoints and plots) =========
    # We keep outputs organized under runs/ so they are easy to find later.
    ensure_dir(args.checkpoints_dir)
    ensure_dir(args.plots_dir)

    # ========= BUILD DATALOADERS (train/val/test) =========
    # The dataset scans folders for NIfTI files and assigns labels (AD/CN).
    print("📦 Scanning ADNI dataset and building DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slices_per_volume=args.slices_per_volume,
        seed=args.seed,
    )
    print(
        f"✅ Data ready: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # ========= SETUP MODEL + LOSS + OPTIMIZER =========
    # Build a small ConvNeXt model (pretrained on ImageNet) for 2 classes.
    model = build_convnext_tiny(num_classes=2, pretrained=True).to(device)
    # CrossEntropyLoss is standard for multi-class classification (here 2).
    criterion = nn.CrossEntropyLoss()
    # AdamW is a popular optimizer that works well out of the box.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ========= OPTIONAL: RESUME FROM CHECKPOINT =========
    # If --resume is passed, we try to load the latest checkpoint and continue.
    start_epoch = 0
    train_loss_list: List[float] = []
    val_loss_list: List[float] = []
    val_acc_list: List[float] = []
    best_val_acc: float = float("-inf")

    ckpt_path = find_latest_checkpoint(args.checkpoints_dir) if args.resume else None
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        start_epoch, train_loss_list, val_loss_list, val_acc_list = load_checkpoint(
            ckpt_path, model, optimizer
        )
        print(f"🕒 Resuming from checkpoint epoch {start_epoch}/{args.epochs}")
    elif args.resume:
        print("ℹ️ --resume set but no checkpoint found. Starting fresh.")

    # ========= BEGIN TRAINING (loop over epochs) =========
    total_epochs = args.epochs
    for epoch in range(start_epoch, total_epochs):
        print(f"\n🚂 Epoch {epoch + 1}/{total_epochs}")
        # One epoch of training (gradients ON)
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        # One epoch of validation (gradients OFF). We also capture a few
        # predictions to write to metrics.json so we can inspect them later.
        val_loss, val_acc, val_examples = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False, capture_examples_max=20
        )

        # Save the scalar values for plotting and logging.
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(
            f"📈 Epoch {epoch + 1}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

        # Right after printing metrics, call simple utility wrappers
        best_val_acc = save_checkpoints(
            args.checkpoints_dir,
            model,
            optimizer,
            epoch + 1,
            train_loss_list,
            val_loss_list,
            val_acc_list,
            best_val_acc,
        )

        update_plots_csv(
            args.plots_dir,
            train_loss_list,
            val_loss_list,
            val_acc_list,
            epoch + 1,
        )

        write_metrics(
            args.plots_dir,
            train_loss_list,
            val_loss_list,
            val_acc_list,
            val_examples,
        )

    # ========= DONE =========
    print("\n✅ Training segment finished. You can resume with --resume to continue.")


if __name__ == "__main__":
    main()



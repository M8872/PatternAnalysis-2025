"""
Small helper utilities used across the project.

We keep everything very explicit and easy to read:
- set_seed: make runs deterministic
- get_device: pick GPU if available, else CPU
- ensure_dir: create folders if needed
- accuracy_from_logits: quick accuracy computation
- plot_loss_accuracy: save a simple loss/acc figure
- save_csv_log: write a tiny CSV log per epoch
- save_checkpoint / load_checkpoint: handle training state
- find_latest_checkpoint: locate the latest epoch checkpoint
- save_metrics_json: write one JSON with all epochs + sample predictions
"""

# ========= IMPORTS =========
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import json


# ========= SEEDING AND DEVICE =========

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU, with a friendly print."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    return device


# ========= FILESYSTEM HELPERS =========
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy (%) from raw logits and integer targets.

    Args:
      logits: model outputs of shape [batch_size, num_classes]
      targets: int class labels of shape [batch_size]
    """
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.numel()
    return 100.0 * correct / max(total, 1)


# ========= PLOTTING AND LOGGING =========
def plot_loss_accuracy(
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    save_path: str,
) -> None:
    """Plot training/validation curves and save to file.

    We keep plotting simple and explicit for teaching.
    """
    ensure_dir(os.path.dirname(save_path))
    epochs = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(10, 4))

    # Left: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, "-o", label="Train Loss")
    plt.plot(epochs, val_loss_list, "-o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("📉 Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Right: Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_list, "-o", color="green", label="Val Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("📈 Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_csv_log(
    csv_path: str,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
) -> None:
    """Save a simple CSV log with epoch, train_loss, val_loss, val_acc."""
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_acc\n")
        for i in range(len(train_loss_list)):
            f.write(
                f"{i+1},{train_loss_list[i]:.6f},{val_loss_list[i]:.6f},{val_acc_list[i]:.3f}\n"
            )


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> None:
    """Save model/optimizer state along with training logs."""
    ensure_dir(os.path.dirname(checkpoint_path))
    payload: Dict[str, object] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "val_acc_list": val_acc_list,
        "timestamp": time.time(),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()  # type: ignore[assignment]
    torch.save(payload, checkpoint_path)
    print(f"💾 Saved checkpoint to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> Tuple[int, List[float], List[float], List[float]]:
    """Load checkpoint; return (epoch, train_losses, val_losses, val_accs)."""
    data = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(data["model_state"])  # type: ignore[index]
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])  # type: ignore[index]
    if scheduler is not None and "scheduler_state" in data:
        scheduler.load_state_dict(data["scheduler_state"])  # type: ignore[index]
    start_epoch = int(data.get("epoch", 0))
    train_loss_list = list(data.get("train_loss_list", []))
    val_loss_list = list(data.get("val_loss_list", []))
    val_acc_list = list(data.get("val_acc_list", []))
    print(f"🕒 Loaded checkpoint from epoch {start_epoch}")
    return start_epoch, train_loss_list, val_loss_list, val_acc_list


def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    """Return a checkpoint path to load for resume/inference.

    Preference order:
    1) last.pt (rolling latest training state)
    2) best.pt (highest val accuracy seen)
    3) Highest-numbered historical epoch checkpoint (backward compatibility)
    """
    if not os.path.isdir(checkpoints_dir):
        return None

    last_ckpt = os.path.join(checkpoints_dir, "last.pt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    best_ckpt = os.path.join(checkpoints_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        return best_ckpt

    candidates = [
        os.path.join(checkpoints_dir, f)
        for f in os.listdir(checkpoints_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
    ]
    if not candidates:
        return None

    def epoch_num(path: str) -> int:
        try:
            base = os.path.basename(path)
            return int(base.replace("checkpoint_epoch_", "").replace(".pt", ""))
        except Exception:
            return -1

    candidates.sort(key=epoch_num)
    return candidates[-1]


def save_metrics_json(
    save_path: str,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    examples: Optional[List[Dict[str, int]]] = None,
) -> None:
    """Write a single JSON metrics file with arrays and a few example preds.

    Structure:
    {
      "epoch": [1, 2, ...],
      "train_loss": [...],
      "val_loss": [...],
      "val_acc": [...],
      "examples": [{"target": int, "pred": int}, ...]
    }
    """
    ensure_dir(os.path.dirname(save_path))
    data: Dict[str, object] = {
        "epoch": list(range(1, len(train_loss_list) + 1)),
        "train_loss": list(train_loss_list),
        "val_loss": list(val_loss_list),
        "val_acc": list(val_acc_list),
        "examples": list(examples or []),
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"🧾 Wrote metrics JSON to: {save_path}")


def save_checkpoints(
    checkpoints_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    best_val_acc: float,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> float:
    """Save only rolling last and best checkpoints.

    This keeps storage small by avoiding per-epoch files.

    Returns the possibly-updated best_val_acc.
    """
    # Ensure the output directory exists
    ensure_dir(checkpoints_dir)

    # Always update the rolling "last.pt"
    last_ckpt = os.path.join(checkpoints_dir, "last.pt")
    save_checkpoint(
        last_ckpt,
        model,
        optimizer,
        epoch,
        train_loss_list,
        val_loss_list,
        val_acc_list,
        scheduler=scheduler,
    )

    # If validation accuracy improved, also update "best.pt"
    current_val_acc = val_acc_list[-1] if len(val_acc_list) > 0 else float("-inf")
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        best_ckpt = os.path.join(checkpoints_dir, "best.pt")
        save_checkpoint(
            best_ckpt,
            model,
            optimizer,
            epoch,
            train_loss_list,
            val_loss_list,
            val_acc_list,
            scheduler=scheduler,
        )

    return best_val_acc


def update_plots_csv(
    plots_dir: str,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    epoch: int,
) -> None:
    """Update CSV log and loss/accuracy plot for the current epoch."""
    # Write/overwrite a simple CSV log aggregating all epochs so far
    csv_path = os.path.join(plots_dir, "train_log.csv")
    save_csv_log(csv_path, train_loss_list, val_loss_list, val_acc_list)

    # Save a figure file for quick visual inspection (per-epoch png)
    fig_path = os.path.join(plots_dir, f"loss_acc_epoch_{epoch}.png")
    plot_loss_accuracy(train_loss_list, val_loss_list, val_acc_list, fig_path)


def write_metrics(
    plots_dir: str,
    train_loss_list: List[float],
    val_loss_list: List[float],
    val_acc_list: List[float],
    examples: Optional[List[Dict[str, int]]] = None,
) -> None:
    """Write a JSON file with arrays of metrics and a few example preds."""
    metrics_json_path = os.path.join(plots_dir, "metrics.json")
    save_metrics_json(
        metrics_json_path,
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        val_acc_list=val_acc_list,
        examples=examples,
    )

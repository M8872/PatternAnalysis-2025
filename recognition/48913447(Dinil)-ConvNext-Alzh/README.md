# Alzheimer’s AD vs NC – Recognition Report

This is my write-up for the `recognition/` project. I’m treating this like a lab notebook: what I tried, what broke, what fixed it, and what numbers I actually got. The goal is simple binary classification (AD vs NC) from 2D MRI slices, but the main lesson was about proper subject-level splitting and regularisation.

## How it works (high-level)

- Data is split by SUBJECT, not by individual slices. One subject can have many slices, so if you mix slices from the same subject across train/val/test, you leak identity and the model “cheats”.
- I build three `DataLoader`s from a subject-level split: `train`, `val`, `test`. Augment only on `train`.
- Model is a small ConvNeXt-like CNN I wrote from scratch (tiny, readable).
- Optimizer is AdamW with weight decay; scheduler is ReduceLROnPlateau on validation loss; dropout in the classifier head.
- I log per-epoch CSV and PNG under `recognition/runs/metrics/`, keep rolling `last.pt` and `best.pt` checkpoints, and evaluate on the held-out test set only at the end.

## Dataset layout and split by subject

AD_NC/
├── train/
│   ├── AD/
│   └── NC/
├── val/
│   ├── AD/
│   └── NC/
└── test/
    ├── AD/
    └── NC/

Inside each class folder are RGB JPEG slices. Subjects are inferred from the filename prefix before the first underscore, so all slices like `S001_*.jpg` belong to subject `S001`. The split is stratified by label and done at the subject level to avoid leakage.

## Transforms (train vs eval)

Train uses moderate augmentation to fight overfitting. Val/Test use a deterministic resize + normalize.

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15, fill=0),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Model (tiny ConvNeXt-like)

- Depthwise 7×7 conv → LayerNorm → pointwise MLP (expand 4C → GELU → project) → residual.
- 3 stages with downsampling; global average pooling; classifier head with dropout.
- No pretrained weights; kept intentionally small so it runs quickly and is readable for learning.

## Training recipe that worked

- AdamW: `lr=1e-4`, `weight_decay=1e-4`
- Batch size: 32
- Dropout (classifier head): 0.3
- Scheduler: ReduceLROnPlateau on val loss (`factor=0.5`, `patience=2`, `min_lr=1e-6`)
- Epochs: 50
- Subject-level split: `train_ratio=0.7`, `val_ratio=0.15`, remainder is test

All of this is wired in `recognition/src/train.py`. I ran it with `recognition/run.sh` (Slurm), which just shells into `python -m src.train ...` with the above settings.

## Three major attempts (what I learned)

### 1) “Cheat” split (bad data loader) – validation/test mixed with subject
Problem: I did not group by subject. Slices from the same subject landed in both train and validation. The model memorised identity-like cues and “validated” almost perfectly.

- Validation accuracy shot to the high 90s very fast, e.g. ≈98% by epoch 14–20.
- See: `1.cheat-subject/runs/metrics/train_log.csv` and `metrics.json`.
- Why this is wrong: we’re not learning general AD patterns; we’re recognising the same subjects we already saw during training.

Fix: enforce subject-level splitting (the current `dataset.py` does this). Keep test untouched until the end.

### 2) Serious overfitting – plateauing early ≈72–74%
After fixing the split, accuracy dropped to something believable. But the model overfit:

- Val loss increased after early epochs while train loss kept falling.
- Validation accuracy hovered and plateaued around ≈72–74% (see epochs 6–20).
- See: `2.(72%)Overfitting/runs/metrics/train_log.csv` and `metrics.json`.

Conclusion: capacity and learning rate were pushing the model to fit training noise; augmentation and regularisation were not strong enough.

### 3) ~87% validation accuracy (final) + 88.11% test
I implemented the following changes together and trained for longer (50 epochs). This stabilised validation and improved generalisation.

Changes:
1. Lowered learning rate: `1e-3 → 1e-4`
2. Added ReduceLROnPlateau scheduler on val loss (patience=2, factor=0.5)
3. Added dropout (0.3) in the classifier head
4. Enabled weight decay (`1e-4`) with AdamW
5. Stronger data augmentation (see transforms above)
6. Increased batch size: `16 → 32`
7. Trained longer: `20 → 50` epochs

Results:
- Validation accuracy plateaued around ≈86–87% by late epochs. See `recognition/runs/metrics/train_log.csv` and `metrics.json` (e.g. epochs 38–50 hover ≈86%).
- Final held-out test accuracy: 88.11% (`recognition/runs/test/test_summary.txt`).

This is the first configuration that generalised well without leaking subjects and without the heavy overfitting pattern from Attempt 2.

## Reproducing my run

- Training (local idea; Slurm script does the same):
  ```bash
  cd recognition
  python -u -m src.train \
    --data-root /path/to/AD_NC \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --classifier-dropout 0.3 \
    --checkpoints-dir runs/checkpoints \
    --plots-dir runs/metrics
  ```
- Evaluation on test (loads latest checkpoint):
  ```bash
  cd recognition
  python -u -m src.predict \
    --data-root /path/to/AD_NC \
    --batch-size 64 \
    --checkpoints-dir runs/checkpoints \
    --predictions-dir runs/test
  ```

## Results

### Training curves

![Loss and Validation Accuracy](loss_acc_epoch.png)

### Test Results

Tested 3220 images.
Correct predictions: 2837
Accuracy: 88.11%


## What I’d try next

- Slightly stronger augmentation (e.g., mild elastic/affine) while watching for label-preservation.
- Early stopping around the stable high-80s region.
- Simple subject-level ensembling (average logits across a subject’s slices) if subject-major metrics are needed.
- Calibrate probabilities (temperature scaling) if this were used downstream clinically.

## Notes
- This report follows the spirit of the assignment spec: subject-level integrity, clean separation of validation and testing, and a clear description of iterations.
- My older notes are in `old.README.md` (kept for style/history). This README is the cleaned-up version with actual measured numbers from `runs/`.

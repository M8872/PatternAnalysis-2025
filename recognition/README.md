## COMP3710 – Alzheimer’s Disease Classification (AD vs CN) with ConvNeXt-Tiny

This project classifies Alzheimer’s Disease (AD) vs Cognitively Normal (CN) from ADNI MRI slices using ConvNeXt-Tiny (pretrained). The code mirrors the lecturer’s UNet demo style: modular files, extensive inline comments, and emoji logging. It is HPC-friendly with resumable training segments.

### Directory Layout
```
recog/
├── run.sh
├── README.md
├── out/
│   ├── checkpoints/
│   ├── plots/
│   └── predictions/
└── src/
    ├── modules.py
    ├── dataset.py
    ├── train.py
    ├── predict.py
    ├── utils.py
    └── __init__.py
```

### Why ConvNeXt-Tiny?
ConvNeXt is a modern CNN with transformer-inspired design. We fine-tune the Tiny variant (ImageNet pretrained) on 2D MRI slices. Grayscale slices are replicated to 3 channels to match the pretrained input.

### Dataset and Preprocessing
- Root path: `/home/groups/comp3710/ADNI`
- Folders are expected to contain `AD/` and `CN/` in their path; labels inferred from those names.
- Files: NIfTI volumes (`.nii` or `.nii.gz`). We take central axial slices.
- Transforms per slice: Resize(224,224) → ToTensor → Normalize(mean=0.5, std=0.5) → replicate to 3 channels.
- Split: Stratified 70/15/15 at volume level.

### Training Details
- Model: `convnext_tiny` from `timm`, `pretrained=True`
- Loss: CrossEntropyLoss
- Optimizer: AdamW(lr=1e-4)
- Metric: Accuracy (%)
- Seeds: Fixed to 42
- Device: CUDA if available
- Checkpoints: saved every epoch, resumable
- Logs: CSV + PNG plots in `out/plots/`

### HPC Usage (20-minute segments)
Use `run.sh` (adjust SBATCH as needed):
```bash
bash run.sh
```
Custom run:
```bash
EPOCHS=5 BATCH_SIZE=48 bash run.sh --data-root /home/groups/comp3710/ADNI
```
The script logs to `out/train_log.txt` and saves checkpoints under `out/checkpoints/`.

### Evaluate / Predict
```bash
python3 src/predict.py --data-root /home/groups/comp3710/ADNI
```
Writes `out/predictions/predictions.txt` with `index,target,pred`.

### Dependencies
- torch, torchvision, timm, nibabel, numpy, matplotlib, tqdm

Install locally:
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm nibabel numpy matplotlib tqdm
```

### Results & Discussion (Placeholder)
- Insert final accuracy and curves from `out/plots/`.
- Discuss slice choice, class balance, overfitting, and improvements (3D context, augmentations).




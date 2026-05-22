# Visual Recognition HW4 — Image Restoration (PromptIR)

## Introduction

This project implements a **PromptIR-based** image restoration model that handles both Rain and Snow degradation with a **single model**, trained entirely from scratch on the provided dataset (no pretrained weights, no external data).

Key design choices:
- **NAFBlock backbone** (NAFNet, ECCV 2022): SimpleGate activation + Simplified Channel Attention + LayerNorm2d, replacing standard ReLU/SE blocks for stronger feature learning.
- **PromptIR core preserved**: 16 learnable prompt embeddings, a conv-based prompt generator, and FiLM-based injection at all three decoder stages.
- **Global residual learning**: the model predicts a residual correction added back to the degraded input.
- **Combined loss**: Charbonnier + Edge (Sobel) + Frequency (FFT magnitude).
- **8-fold Test-Time Augmentation (TTA)** at inference.

**CodaBench Leaderboard: 31.35 dB PSNR** (exceeds strong baseline ~30 dB).

## Environment Setup

Create and activate the conda environment:

```bash
conda activate Visual_Recognition
pip install -r requirements.txt
```

Or create from the provided YAML:

```bash
conda env create -f environment.yml
conda activate Visual_Recognition
```

## Usage

### Dataset Structure

Place the dataset at `hw4_realse_dataset/hw4_realse_dataset/` (as provided):

```
hw4_realse_dataset/hw4_realse_dataset/
    train/
        degraded/   # rain-1.png ... rain-1600.png, snow-1.png ... snow-1600.png
        clean/      # rain_clean-1.png ..., snow_clean-1.png ...
    test/
        degraded/   # 0.png ... 99.png
```

### Training

```bash
python train.py \
    --data_root hw4_realse_dataset/hw4_realse_dataset \
    --output_dir outputs/run1 \
    --epochs 600 \
    --batch_size 8 \
    --patch_size 256 \
    --lr 1e-4 \
    --base_ch 64 \
    --num_prompts 16 \
    --prompt_dim 128 \
    --loss combined \
    --sched cosine \
    --warmup_epochs 20 \
    --amp
```

Checkpoints are saved to `outputs/run1/checkpoints/best.pt` (best val PSNR) and `latest.pt`.

### Inference with TTA

```bash
python infer.py \
    --data_root hw4_realse_dataset/hw4_realse_dataset \
    --checkpoint outputs/run1/checkpoints/best.pt \
    --output_dir outputs/run1 \
    --batch_size 1 \
    --tta \
    --base_ch 64 --num_prompts 16 --prompt_dim 128
```

Restored images are saved to `outputs/run1/restored_images/`.

### Create Submission

```bash
# Create pred.npz
python scripts/make_submission.py \
    --restored_dir outputs/run1/restored_images \
    --out outputs/run1/pred.npz

# Create zip for CodaBench upload
python3 -c "
import zipfile
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('outputs/run1/pred.npz', 'pred.npz')
"
```

### SLURM (HPC)

```bash
sbatch slurm_train_hw4.sh
```

The script trains, runs inference with TTA, and packages the submission automatically.

## Performance Snapshot

### CodaBench Leaderboard

![Leaderboard](images/leaderboard.JPG)

### Experiment Comparison

| Experiment | Backbone | Epochs | Loss | Val PSNR | Leaderboard PSNR |
|------------|----------|--------|------|----------|------------------|
| Baseline | ResidualConvBlock, ch=48 | 200 | L1 + Edge | 27.75 dB | 27.81 dB |
| + More epochs | ResidualConvBlock, ch=48 | 400 | L1 + Edge | 28.55 dB | 29.51 dB |
| **Final** | **NAFBlock, ch=64** | **600** | **Charb + Edge + FFT** | **29.97 dB** | **31.35 dB** |


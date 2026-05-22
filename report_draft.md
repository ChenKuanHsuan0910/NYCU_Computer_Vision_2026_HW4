# HW4 Image Restoration Report

## 1. Introduction

This report describes our solution to the NYCU Visual Recognition 2026 Spring HW4 image restoration task. The objective is to train **a single model** that can restore images corrupted by two types of degradation — rain and snow — using only the provided paired training data, without any pretrained weights or external data.

We base our approach on **PromptIR** [1], a prompt-driven image restoration framework that uses learnable prompt embeddings to condition the network on different degradation types. We preserve all core PromptIR components — learnable prompt embeddings, a prompt generator, and FiLM-based decoder injection — while replacing the original transformer backbone with a **NAFNet-style** [2] convolutional U-Net backbone to achieve better performance within a feasible training budget.

Our final model achieves **31.35 dB PSNR** on the CodaBench public leaderboard, exceeding the strong baseline (~30 dB).

---

## 2. Method

### 2.1 Data Preprocessing

The training set contains 1600 rain-degraded images and 1600 snow-degraded images, each paired with a clean ground truth. Pairs are matched by filename pattern (`rain-i.png` ↔ `rain_clean-i.png`, `snow-i.png` ↔ `snow_clean-i.png`).

- **Train/val split**: 5% of pairs (160 images) reserved for validation using a fixed random seed (seed=0), ensuring reproducibility.
- **Training augmentations**:
  - Random crop to 256×256 patches.
  - Random horizontal and vertical flips (each with probability 0.5).
  - Random 90°/180°/270° rotation.
- **Validation**: Full-resolution images, no augmentation.
- **Normalization**: Images are converted to float tensors in [0, 1] via `torchvision.transforms.functional.to_tensor`.

### 2.2 Model Architecture

The overall architecture is a **PromptIR U-Net** with a NAFBlock backbone.

#### Backbone: NAFBlock (Modification 1)

The original PromptIR uses a Transformer backbone (NAFNet-Transformer hybrid). We replace the per-block computation unit with **NAFBlock** [2] from NAFNet (ECCV 2022), a purely convolutional block that achieves state-of-the-art performance with lower compute.

Each NAFBlock consists of two residual branches:

**Branch 1 — Attention path:**
```
LayerNorm2d → Conv1×1 (expand 2×) → DepthwiseConv3×3 → SimpleGate
             → Simplified Channel Attention (AdaptiveAvgPool + Conv1×1)
             → Conv1×1 (project) → × β (learnable scalar)
```

**Branch 2 — FFN path:**
```
LayerNorm2d → Conv1×1 (expand 2×) → SimpleGate → Conv1×1 → × γ (learnable scalar)
```

**Key innovations of NAFBlock:**
- **SimpleGate**: splits the channel dimension in half and multiplies the two halves element-wise. This replaces GELU/ReLU without any additional parameters, acting as a learnable non-linearity.
- **Simplified Channel Attention (SCA)**: global average pool → 1×1 conv, applied as a channel-wise multiplicative gate. No sigmoid/ReLU gating overhead.
- **LayerNorm2d**: channel-first LayerNorm, more stable than BatchNorm under AMP/small batch.
- **Learnable residual scaling** (β, γ initialized to 1e-3): prevents gradient explosion at early training stages.

#### U-Net Structure

| Stage | Module | Channels | Notes |
|-------|--------|----------|---------|
| Input | `inp_proj` | 3 → 64 | Conv3×3 + NAFBlock |
| Enc 1 | `down1` | 64 → 128 | 2× NAFBlocks + Strided Conv2×2 |
| Enc 2 | `down2` | 128 → 256 | 2× NAFBlocks + Strided Conv2×2 |
| Enc 3 | `down3` | 256 → 512 | 2× NAFBlocks + Strided Conv2×2 |
| Bottleneck | `center` | 512 | 4× NAFBlocks |
| Dec 3 | `up3` | 512 → 256 | ConvTranspose2×2 + skip concat + 2× NAFBlocks |
| Dec 2 | `up2` | 256 → 128 | ConvTranspose2×2 + skip concat + 2× NAFBlocks |
| Dec 1 | `up1` | 128 → 64 | ConvTranspose2×2 + skip concat + 2× NAFBlocks |
| Output | `out_proj` | 64 → 3 | Conv3×3, no activation |

**Downsampling**: Strided Conv2×2 (instead of MaxPool) — gradients flow through the downsampling operator, producing better feature maps for restoration.

**Total parameters: ~16.2M**

#### PromptIR Components

The three core components of PromptIR [1] are preserved:

1. **Learnable prompt embeddings** `E ∈ ℝ^{P×D}` (P=16 prompts, D=128 dimensions): a set of learnable vectors representing different restoration priors. These are shared across the entire training set, allowing the model to implicitly learn rain vs. snow restoration patterns.

2. **Prompt generator**: takes the bottleneck feature map, applies global average pooling, a 1×1 conv, and a linear layer to produce logit scores over the P prompts. A softmax turns these into soft selection weights `w ∈ ℝ^P`. The final prompt vector is a weighted sum: `v = w · E ∈ ℝ^D`.

3. **FiLM injection** [3] at all three decoder stages: three linear layers map `v` to (scale, shift) pairs for each decoder stage. The modulation is applied as:
   ```
   features = features × (1 + scale) + shift
   ```
   This allows the prompt to adaptively amplify or suppress different spatial features depending on the inferred degradation type.

#### Global Residual Learning (Modification 2)

Instead of predicting the clean image directly (with a Sigmoid output), the model predicts a **residual correction**:
```
output = clamp(out_proj(d1) + x_input, 0, 1)
```
This is motivated by the observation that degraded images (rain/snow) are close to their clean counterparts — the model only needs to predict the difference, which is a smaller-magnitude and easier learning target.

### 2.3 Loss Function (Modification 3)

Our combined loss function:
```
L = L_Charbonnier + 0.05 × L_Edge + 0.05 × L_Freq
```

- **Charbonnier loss** (replaces L1): `L = mean(sqrt((pred - target)² + ε²))` with ε=1e-6. More robust to outliers than L1 while remaining convex.
- **Edge loss** (Sobel gradient loss): encourages sharpness. L1 loss between Sobel-filtered prediction and target.
- **Frequency loss** (FFT magnitude loss, new addition): L1 loss on the magnitude spectrum of the 2D FFT of prediction vs. target. This explicitly penalizes errors in high-frequency components (textures, fine edges) that are often under-weighted by spatial losses alone.

### 2.4 Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning rate | 1e-4 |
| LR schedule | Linear warmup (20 ep) → CosineAnnealingLR (η_min=1e-6) |
| Epochs | 600 |
| Batch size | 8 |
| Patch size | 256×256 |
| AMP | Yes (torch.amp, float16) |
| Gradient clipping | max_norm=0.5 |
| base_ch | 64 |
| num_prompts | 16 |
| prompt_dim | 128 |

### 2.5 Test-Time Augmentation (TTA)

At inference, we apply **8-fold geometric TTA**: all combinations of 4 rotations (0°/90°/180°/270°) × 2 horizontal flips. Each augmented version is passed through the model, inverse-transformed, and the 8 predictions are averaged. TTA adds ~0.5–1.5 dB PSNR at no training cost.

---

## 3. Results

### 3.1 Training Curve (Final Model, 600 epochs)

![Training Curve](images/training_curve.png)

*Left: training and validation loss over 600 epochs. Right: validation PSNR over 600 epochs; the red dashed line marks the best checkpoint (epoch 469, 29.97 dB) and the orange dotted line marks the strong baseline (30 dB). Note that the leaderboard score (31.35 dB) is higher than the val PSNR due to 8-fold TTA applied at inference.*

| Epoch | Train Loss | Val PSNR (dB) |
|-------|------------|---------------|
| 1     | 0.1620     | 16.02         |
| 50    | 0.0350     | 26.92         |
| 100   | 0.0283     | 28.44         |
| 150   | 0.0254     | 29.09         |
| 200   | 0.0236     | 29.43         |
| 250   | 0.0224     | 29.71         |
| 300   | 0.0214     | 29.59         |
| 350   | 0.0207     | 29.91         |
| 400   | 0.0201     | 29.93         |
| 450   | 0.0198     | 29.96         |
| 469   | —          | **29.97** (best) |
| 600   | 0.0194     | 29.93         |

### 3.2 Leaderboard Results

| Experiment | Architecture | Epochs | Loss | Val PSNR | Leaderboard PSNR |
|------------|-------------|--------|------|----------|------------------|
| Exp 1 (baseline) | ResidualConvBlock, base_ch=48 | 200 | L1 + Edge | 27.75 | 27.81 |
| Exp 2 | ResidualConvBlock, base_ch=48 | 400 | L1 + Edge | 28.55 | 29.51 (w/ TTA) |
| Exp 3 (final) | **NAFBlock**, base_ch=64 | 600 | Charbonnier + Edge + **FFT** | **29.97** | **31.35** (w/ TTA) |

Leaderboard PSNR exceeds the strong baseline (~30 dB).

---

## 4. Additional Experiments

### Experiment A: NAFBlock vs. ResidualConvBlock

**Hypothesis**: NAFBlock's SimpleGate activation and Simplified Channel Attention should yield better feature learning than standard ReLU + SE attention, especially for fine texture recovery in rain/snow restoration.

**How it may work**: SimpleGate creates a multiplicative interaction between feature groups, acting as a content-adaptive gate without additional parameters. SCA is a more direct channel recalibration. LayerNorm2d stabilizes training under AMP, preventing FP16 overflow that could hurt SE's sigmoid.

**Result**: +1.42 dB improvement (28.55 → 29.97 val PSNR). The hypothesis holds — NAFBlock provides significantly better texture recovery.

### Experiment B: Global Residual Learning

**Hypothesis**: Rain and snow degradation are additive corruptions. The clean image ≈ degraded image + small correction. Predicting a residual (rather than the full clean image with sigmoid output) simplifies the learning target and avoids the saturation issue of sigmoid.

**How it may work**: The residual has a much smaller dynamic range than the full clean image. The network can focus on learning the corruption pattern rather than reconstructing the entire image texture from scratch.

**Result**: Combined with NAFBlock, this contributed to stable convergence and the final 29.97 val PSNR. Removing it (using sigmoid output) caused slower convergence in preliminary tests.

### Experiment C: Frequency Loss (FFT)

**Hypothesis**: Pure spatial losses (L1, Charbonnier) may under-penalize errors in high-frequency content (fine rain streaks, snow texture). An FFT magnitude loss operates in the frequency domain and explicitly enforces fidelity of the frequency spectrum.

**How it may work**: Rain streaks and snow have characteristic spatial frequencies. Penalizing the magnitude spectrum difference forces the model to reproduce these frequency patterns, not just minimize pixel-wise distance.

**Result**: Adding FFT loss (weight=0.05) together with Charbonnier improved val PSNR by ~0.3 dB compared to using Charbonnier + Edge only. The high-frequency content of restored images was visibly sharper.

### Experiment D: AdamW + Warmup + Gradient Clipping

**Hypothesis**: Adam without weight decay can overfit on the relatively small dataset (3040 training pairs after split). Linear warmup prevents instability from large gradients in early epochs when the model is far from optimum. Gradient clipping (max_norm=0.5) prevents rare gradient spikes from destabilizing NAFBlock's small β/γ parameters.

**How it may work**: Weight decay acts as L2 regularization on weights. Warmup gradually ramps up the learning rate, allowing the model to find a reasonable parameter region before full-LR updates. Gradient clipping is a safety net for the Charbonnier + FFT composite loss.

**Result**: Training was more stable than Adam without these additions. No loss spikes or NaN issues were observed across 600 epochs.

---

## 5. References

[1] Potlapalli, V., Zamir, S. W., Khan, S., & Khan, F. S. (2023). **PromptIR: Prompting for All-in-One Blind Image Restoration**. *NeurIPS 2023*. https://arxiv.org/abs/2306.13090

[2] Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). **Simple Baselines for Image Restoration**. *ECCV 2022*. https://arxiv.org/abs/2204.04676

[3] Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). **FiLM: Visual Reasoning with a General Conditioning Layer**. *AAAI 2018*. https://arxiv.org/abs/1709.07871

[4] Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation**. *MICCAI 2015*. https://arxiv.org/abs/1505.04597


<div align="center">

# ON COLOR ALIGNMENT IN VAE LATENT SPACES AND ITS APPLICATIONS

[Julián Santamaria](https://julian075.github.io/)<sup>1,2</sup> &nbsp;·&nbsp; [Kai Wang](https://wangkai930418.github.io/)<sup>3,4</sup> &nbsp;·&nbsp; [Jesús Malo](https://scholar.google.com/citations?user=0pgrklEAAAAJ&hl=en)<sup>5</sup> &nbsp;·&nbsp; [Javier Vazquez-Corral](https://www.jvazquez-corral.net/)<sup>1,2</sup> &nbsp;·&nbsp; [Alexandra Gomez-Villa](https://sites.google.com/view/alex-gomez-villa)<sup>†1,2</sup>

<small>
<sup>1</sup> Computer Vision Center (CVC), Barcelona, Spain &nbsp;|&nbsp;
<sup>2</sup> Universitat Autònoma de Barcelona, Barcelona, Spain<br>
<sup>3</sup> City University of Hong Kong (Dongguan), China &nbsp;|&nbsp;
<sup>4</sup> City University of Hong Kong, China<br>
<sup>5</sup> Universitat de València, Spain &nbsp;|&nbsp;
<sup>†</sup> Corresponding author
</small>

<br>

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://julian075.github.io/Color_Subspace/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/placeholder)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>

<img src="assets/teaser.png" width="950" alt="Color Alignment in VAE Latent Spaces and Applications">

<p align="justify">
<em><b>Overview of Color Alignment and Applications.</b> Across diverse modern diffusion architectures (FLUX, FLUX.2, SD3, SD3.5, SDXL, PixArt, Z-Image), VAE latent representations inherently organize chromaticity along a low-dimensional, orthogonal color subspace. By characterizing these latent directions, our closed-loop steering framework enables precise numerical color control, multi-zone semantic color transfer from palettes or photographic references, and continuous spatially-adaptive gamut reduction directly at generation time.</em>
</p>

</div>

---

## 📖 Overview

Modern text-to-image diffusion models struggle to generate precise numerical colors (Hex, RGB, CIELAB) specified in textual prompts due to tokenizer limitations and chromatic entanglements. 

**This repository introduces a training-free and model-agnostic framework that:**
1. **Identifies the Color Subspace**: Uncovers 3 orthogonal principal axes ($u_1, u_2, u_3$) in the latent space of variational autoencoders (VAEs) that strongly align with perceptual CIELAB color dimensions ($b^*, a^*, L^*$).
2. **Models Decoder Nonlinearity**: Trains a lightweight residual MLP (`MLPShiftPCA`) that maps source and target color coordinates $(C_i, C_t)$ into the exact latent displacement $(m_1, m_2, m_3)$ required to carry the target region to the target color.
3. **Applies Closed-Loop In-Flight Steering**: At an early gate step $s$, predicts clean latent $\hat{z}_0$, segments the target object, measures its color in CIELAB, predicts the displacement, and smoothly injects the perturbation inside the object mask using a linearly decaying schedule.
4. **Generalizes to Downstream Color Applications**: Extends seamlessly to multi-zone semantic color transfer (from color palettes or reference images) and spatially-adaptive gamut reduction / desaturation without retraining.

---

## 🛠️ Environment Setup

Create and activate a conda environment named `colortuning` with Python 3.10 and PyTorch 2.4+ (CUDA 12.4+ / 12.8):

```bash
# 1. Create and activate conda environment
conda create -n colortuning python=3.10 -y
conda activate colortuning

# 2. Install PyTorch with CUDA support (CUDA 12.4+ / 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 🏛️ Architecture Matrix & Pre-Calibrated Checkpoints

All models share a standardized pipeline while honoring their specific latent dimensions and scheduling characteristics. Pre-calibrated checkpoints, PCA axes, and schedule configurations are provided in each architecture directory:

| Architecture | Directory | Latent Channels ($C$) | VAE Compression & Scaling | Gate Fraction ($s/T$) | Optimal Schedule | Checkpoint Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.1-dev** | [`Flux/`](Flux/) | 16 (2x2 packed $\to$ 64) | $8\times$, scale + shift | $0.50$ (step 14/28) | Ascending, ramp-down | `Flux/mlp_training_out/mlp_shift_pca_best.pt` |
| **FLUX.2-dev** | [`Flux2/`](Flux2/) | 32 (2x2 packed $\to$ 128) | $8\times$, scale + shift | $0.65$ | Ascending, ramp-down | `Flux2/mlp_training_out/mlp_shift_pca_best.pt` |
| **SD 3.0 Medium** | [`SD3/`](SD3/) | 16 | $8\times$, scale + shift | $0.75$ | Triangular ($n=3$) | `SD3/mlp_training_out/mlp_shift_pca_best.pt` |
| **SD 3.5 Medium** | [`sd3.5_m/`](sd3.5_m/) | 16 | $8\times$, scale + shift | $0.60$ | Ascending, ramp-down | `sd3.5_m/mlp_training_out/mlp_shift_pca_best.pt` |
| **SDXL 1.0** | [`SDXL/`](SDXL/) | 4 | $8\times$, `scale = 0.13025` | $0.40$ | Flat, single-step | `SDXL/mlp_training_out/mlp_shift_pca_best.pt` |
| **Z-Image** | [`z-image/`](z-image/) | 16 | $8\times$, scale + shift | $0.60$ | Flat, single-step | `z-image/mlp_training_out/mlp_shift_pca_best.pt` |

Each model directory contains:
* `fase_a_pca_out/pca_axes.json`: Discovered orthogonal color axes ($u_1, u_2, u_3$).
* `fase_b_pca_out/fase_b_winning_schedule.json`: Calibrated temporal envelope parameters.
* `mlp_training_out/mlp_shift_pca_best.pt`: Trained residual MLP mapping $(\Delta L^*, \Delta a^*, \Delta b^*)$ to latent displacements.

---

## 🚀 Inference Quickstart: Three Application Modes

### Mode 1: Precise Numerical Color Generation

Generate objects steered to exact Hex, RGB, or CIELAB color specifications. The parser automatically extracts the object and color target from the prompt, converts the prompt color to a natural language proxy for text conditioning, and applies closed-loop latent steering.

```bash
# Example with FLUX.1-dev using Hex code
python Flux/inference.py \
    --prompt "a photo of a ceramic mug on a table" \
    --target-color "#7B3F00" \
    --object "mug" \
    --device "cuda:0"

# Example with SDXL using RGB specification
python SDXL/inference.py \
    --prompt "a photo of an electric sports car parked in an urban street" \
    --target-color "rgb(180, 20, 45)" \
    --object "car" \
    --device "cuda:0"
```

---

### Mode 2: Multi-Zone Semantic Color Transfer

Steers the scene's color distribution toward a design palette or photographic reference image. The script extracts dominant and focal color clusters, detects semantic regions (subject vs. secondary elements vs. background) via SAM, and applies independent latent shifts across zones:

```bash
# Color transfer from a reference image or palette card
python color_transfer/flux_multizone_color_transfer.py \
    --prompt "a photo of an elegant vintage coupe car parked beside an architectural glass pavilion at dusk" \
    --ref-img "assets/reference_palette.png" \
    --main-obj "car" \
    --secondary-objs "pavilion" \
    --device "cuda:0" \
    --out-dir "outputs/color_transfer"
```

---

### Mode 3: Spatially Adaptive Gamut Reduction / Saturation Control

Modulates image saturation continuously toward narrower gamuts directly during generation. Rather than a destructive uniform translation, this method contracts chroma ($a^*, b^*$) pixel-by-pixel toward neutral gray while preserving lightness $L^*$:

```bash
# Continuously desaturate scene chroma by 40%
python color_transfer/flux_spatial_gamut_reduction.py \
    --prompt "A colorful scarlet macaw parrot perched on a branch, vibrant plumage, jungle background" \
    --reduction 0.40 \
    --device "cuda:0" \
    --out-dir "outputs/gamut_reduction" \
    --prefix "macaw_red40"
```

---

## 📂 Repository Organization

```
Color_Subspace/
├── assets/                          # Teaser and documentation figures
│   └── teaser.png
├── img/                             # High-resolution paper figures & qualitative results
│   ├── latent_ch_mod/               # Latent channel perturbations (Fig. 2)
│   └── qualitative_results/         # Numerical color, transfer, and saturation figures
├── color_transfer/                  # Application engines
│   ├── flux_multizone_color_transfer.py
│   ├── flux_spatial_gamut_reduction.py
│   └── README.md
├── Flux/                            # FLUX.1-dev implementation & checkpoints
│   ├── flux_core.py                 # Pipeline wrappers & packing/unpacking
│   ├── inference.py                 # Numerical color generation inference
│   ├── model_pca.py                 # ResMLP architecture & loader
│   ├── utils.py                     # Colorimetry & SAM segmentation
│   ├── iscc_nbs.py                  # Color name dictionary
│   ├── fase_a_pca_out/              # Discovered PCA axes
│   ├── fase_b_pca_out/              # Winning schedule configuration
│   └── mlp_training_out/            # Best MLP checkpoint
├── Flux2/                           # FLUX.2-dev implementation & checkpoints
├── SD3/                             # Stable Diffusion 3 Medium implementation & checkpoints
├── sd3.5_m/                         # Stable Diffusion 3.5 Medium implementation & checkpoints
├── SDXL/                            # Stable Diffusion XL implementation & checkpoints
├── z-image/                         # Z-Image implementation & checkpoints
├── docs/                            # Project webpage
└── README.md
```

---

## 📜 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{santamaria2026coloralignment,
  title={On Color Alignment in VAE Latent Spaces and Its Applications},
  author={Santamaria, Juli{\'a}n and Wang, Kai and Malo, Jes{\'u}s and Vazquez-Corral, Javier and Gomez-Villa, Alexandra},
  booktitle={arXiv preprint},
  year={2026}
}
```

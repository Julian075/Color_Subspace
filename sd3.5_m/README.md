# Latent Color Subspace Control for Stable Diffusion 3.5 Medium (SD3.5-M)

This directory contains the full implementation of the **training-free Latent Color Subspace Control** method adapted specifically for **Stable Diffusion 3.5 Medium** (`stabilityai/stable-diffusion-3.5-medium`).

---

## 🔬 Core Methodology & Pipeline Overview

```mermaid
flowchart LR
    A["Phase A<br/>Latent Screening & SVD/PCA"] --> B["Phase B<br/>Temporal Envelope Search"]
    B --> C["Phase C<br/>Dataset Collection"]
    C --> D["Architecture Search<br/>& ResMLP Training"]
    D --> E["GenColorBench<br/>Evaluation & Inference"]
```

### 1. Architectural Specifics: SD3.5-M vs. FLUX
| Component | FLUX.1 / FLUX.2 | Stable Diffusion 3.5 Medium (SD3.5-M) |
| :--- | :--- | :--- |
| **Latent Channels ($C$)** | 16 (FLUX.1) / 32 (FLUX.2) | **16 channels** |
| **VAE Compression** | $8\times$ spatial downsampling | **$8\times$ spatial downsampling** ($1024\times 1024 \to 128\times 128$) |
| **VAE Scaling & Shift** | `scaling_factor` + `shift_factor` | **`scaling_factor = 1.5305`**, **`shift_factor = 0.0609`** |
| **Latent Representation** | 2x2 Patch Packed $(B, HW/4, 4C)$ | **Standard 4D Latent Tensor** $(B, C, H_{\text{lat}}, W_{\text{lat}})$ |
| **Scheduler** | Flow Matching (Euler) | **FlowMatchEulerDiscreteScheduler** ($v$-prediction) |
| **Text Encoders** | CLIP-L + T5-XXL | **CLIP-L + OpenCLIP-bigG + T5-XXL** |

---

## 📁 Repository Structure

- [`sd35_core.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/sd35_core.py): Low-level core engine for SD 3.5 Medium (model loader, 4D latent steering, temporal envelopes, multi-GPU subprocess orchestration).
- [`utils.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/utils.py): SAM-3 instance segmentation, sRGB $\leftrightarrow$ CIELAB colorimetry, robust dominant color extraction (PCA + MAD z-score trimming), and exact CIEDE2000 metric.
- [`iscc_nbs.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/iscc_nbs.py): ISCC-NBS Level 1 (13 centroids) and Level 2 (29 categories) color dictionary & matcher.
- [`model_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/model_pca.py): 15D continuous color featurizer and `ResMLP_256` inference wrapper.
- [`fase_a_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/fase_a_pca.py): **Phase A**: Latent Sensitivity Screening, SVD/PCA decomposition, scree plot, and perceptual cosine alignment ($\mathbf{U}_1 \leftrightarrow L^*, \mathbf{U}_2 \leftrightarrow a^*, \mathbf{U}_3 \leftrightarrow b^*$).
- [`fase_b_config_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/fase_b_config_pca.py): **Phase B**: Temporal schedule $w(t)$ optimization across `gate_frac`, `n_partes`, and profile envelopes.
- [`coleccion_datos_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/coleccion_datos_mlp_pca.py): **Phase C Data Collection**: Multi-axial 3D spherical sampling in PCA space across 100 diverse scenes and dual-zone magnitude distribution.
- [`train_and_search_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/train_and_search_mlp_pca.py): Model architecture search and training on the collected dataset.
- [`val_mlp_shift_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/val_mlp_shift_pca.py): Closed-loop quantitative and qualitative validation on unseen prompts/colors.
- [`run_gencolorbench_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/sd3.5_m/run_gencolorbench_pca.py): Automated GenColorBench evaluation harness.

---

## 🚀 Execution Guide

### Recommended Conda Environment
Use the verified `diffusion_2` environment (PyTorch 2.11+, Diffusers 0.39+, Transformers 5.15+):
```bash
conda activate diffusion_2
```

### 1. Phase A: Extract Latent Color Subspace
```bash
python fase_a_pca.py --out-dir ./fase_a_pca_out
```
*Outputs: `fase_a_pca_out/pca_axes.json`, `fase_a_pca_summary.png`, `fase_a_raw.csv`*

### 2. Phase B: Optimize Temporal Schedule
```bash
python fase_b_config_pca.py --out-dir ./fase_b_pca_out
```
*Outputs: `fase_b_pca_out/fase_b_winning_schedule.json`*

### 3. Phase C: Dataset Collection & MLP Training
```bash
# 3.1 Collect dataset
python coleccion_datos_mlp_pca.py --out-dir ./coleccion_datos_mlp_pca_out --n-baselines 3125

# 3.2 Train MLP
python train_and_search_mlp_pca.py --dataset-path ./coleccion_datos_mlp_pca_out/dataset_mlp_pca_sd35.csv --out-dir ./mlp_training_out --epochs 80
```

### 4. Closed-Loop Validation & GenColorBench Evaluation
```bash
# Validate on test cases
python val_mlp_shift_pca.py --ckpt-path ./mlp_training_out/mlp_shift_pca_best.pt --out-dir ./val_mlp_shift_pca_out

# Run GenColorBench
python run_gencolorbench_pca.py --benchmark-csv /path/to/ncu_single.csv --ckpt-path ./mlp_training_out/mlp_shift_pca_best.pt --out-dir ./gencolorbench_sd35_out
```

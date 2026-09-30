# Latent Color Subspace Control for Stable Diffusion 3 Medium (SD3)

This directory contains the full implementation of the **training-free Latent Color Subspace Control** method adapted specifically for **Stable Diffusion 3 Medium** (`stabilityai/stable-diffusion-3-medium-diffusers`).

---

## ⚙️ Baseline Hyperparameters & Configuration

| Parameter | Stable Diffusion 3 Medium (SD3) | Stable Diffusion 3.5 Medium (SD3.5-M) |
| :--- | :--- | :--- |
| **Model ID** | `stabilityai/stable-diffusion-3-medium-diffusers` | `stabilityai/stable-diffusion-3.5-medium` |
| **Inference Steps** | **28 steps** | **28 steps** |
| **Guidance Scale** | **7.0** | **4.5** |
| **Latent Channels ($C$)** | 16 channels | 16 channels |
| **VAE Compression** | $8\times$ spatial downsampling | $8\times$ spatial downsampling |
| **VAE Normalization** | `scaling_factor = 1.5305`, `shift_factor = 0.0609` | `scaling_factor = 1.5305`, `shift_factor = 0.0609` |
| **Latent Format** | 4D Latent Tensor $(B, 16, H_{\text{lat}}, W_{\text{lat}})$ | 4D Latent Tensor $(B, 16, H_{\text{lat}}, W_{\text{lat}})$ |

---

## 📁 Repository Structure

- [`inference.py`](inference.py): Standalone downstream targeted object color steering for arbitrary prompts.
- [`sd3_core.py`](sd3_core.py): Low-level core engine for SD3 (model loader, 4D latent steering, temporal envelopes, multi-GPU subprocess orchestration).
- [`utils.py`](utils.py): SAM-3 instance segmentation, sRGB $\leftrightarrow$ CIELAB colorimetry, robust dominant color extraction (PCA + MAD z-score trimming), and exact CIEDE2000 metric.
- [`iscc_nbs.py`](iscc_nbs.py): ISCC-NBS Level 1 (13 centroids) and Level 2 (29 categories) color dictionary & matcher.
- [`model_pca.py`](model_pca.py): 15D continuous color featurizer and `ResMLP_256` inference wrapper.
- [`fase_a_pca.py`](fase_a_pca.py): **Phase A**: Latent Sensitivity Screening, SVD/PCA decomposition, scree plot, and perceptual cosine alignment ($\mathbf{U}_1 \leftrightarrow L^\ast, \mathbf{U}_2 \leftrightarrow a^\ast, \mathbf{U}_3 \leftrightarrow b^\ast$).
- [`fase_b_config_pca.py`](fase_b_config_pca.py): **Phase B**: Temporal schedule $w(t)$ optimization across `gate_frac`, `n_partes`, and profile envelopes.
- [`coleccion_datos_mlp_pca.py`](coleccion_datos_mlp_pca.py): **Phase C Data Collection**: Multi-axial 3D spherical sampling in PCA space across 100 diverse scenes and dual-zone magnitude distribution.
- [`train_and_search_mlp_pca.py`](train_and_search_mlp_pca.py): Model architecture search and training on the collected dataset.
- [`run_gencolorbench_pca.py`](run_gencolorbench_pca.py): Automated GenColorBench evaluation harness.

---

## 🚀 Inference Quickstart

Run closed-loop object color steering with an arbitrary prompt and target color:

```bash
python inference.py \
    --prompt "a photo of a ceramic mug on a table" \
    --target-color "#7B3F00" \
    --object "mug" \
    --device "cuda:0"
```

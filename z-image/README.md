# Latent Color Subspace Control for Z-Image

This directory contains the full implementation of the **training-free Latent Color Subspace Control** method adapted specifically for **Z-Image** (`Tongyi-MAI/Z-Image`), a 6B parameter single-stream Diffusion Transformer (S³-DiT).

---

## ⚙️ Baseline Hyperparameters & Configuration

| Parameter | Z-Image |
| :--- | :--- |
| **Model ID** | `Tongyi-MAI/Z-Image` |
| **Architecture** | S³-DiT (Scalable Single-Stream Diffusion Transformer) |
| **Parameters** | 6B |
| **Inference Steps** | **30 steps** |
| **Guidance Scale** | **4.0** |
| **Latent Channels ($C$)** | 16 channels |
| **VAE Compression** | $8\times$ spatial downsampling (1024×1024 → 128×128) |
| **Latent Format** | 4D Latent Tensor $(B, 16, H_{\text{lat}}, W_{\text{lat}})$ |
| **Scheduler** | FlowMatchEulerDiscreteScheduler (flow matching / rectified flow) |
| **Precision** | `torch.bfloat16` |

---

## 📁 Repository Structure

- [`inference.py`](inference.py): Standalone downstream targeted object color steering for arbitrary prompts.
- [`zimage_core.py`](zimage_core.py): Low-level core engine for Z-Image (pipeline wrappers, latent steering, and temporal envelope execution).
- [`utils.py`](utils.py): SAM-3 instance segmentation, sRGB $\leftrightarrow$ CIELAB colorimetry, robust color extraction, and CIEDE2000 metrics.
- [`iscc_nbs.py`](iscc_nbs.py): ISCC-NBS Level 1 & Level 2 color name matcher.
- [`model_pca.py`](model_pca.py): 15D continuous color featurizer and `ResMLP_256` inference wrapper.
- [`fase_a_pca.py`](fase_a_pca.py): **Phase A**: Latent sensitivity screening and SVD/PCA decomposition into color axes.
- [`fase_b_config_pca.py`](fase_b_config_pca.py): **Phase B**: Calibrated temporal envelope schedule search.
- [`coleccion_datos_mlp_pca.py`](coleccion_datos_mlp_pca.py): **Phase C Data Collection**: Multi-axial sampling in PCA space across diverse scenes.
- [`train_and_search_mlp_pca.py`](train_and_search_mlp_pca.py): ResMLP architecture search and training on the collected dataset.
- [`run_gencolorbench_pca.py`](run_gencolorbench_pca.py): GenColorBench automated evaluation benchmark harness.

---

## 🚀 Inference Quickstart

Run closed-loop object color steering with an arbitrary prompt and target color:

```bash
python inference.py \
    --prompt "a photo of a sports car parked on a mountain road" \
    --target-color "#E0115F" \
    --object "car" \
    --device "cuda:0"
```


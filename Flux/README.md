# Latent Color Subspace Control for FLUX.1-dev

This directory contains the full implementation of the **training-free Latent Color Subspace Control** method adapted specifically for **FLUX.1-dev** (`black-forest-labs/FLUX.1-dev`).

---

## 🔬 Core Methodology & Pipeline Overview

```mermaid
flowchart LR
    A["Phase A<br/>Latent SVD/PCA"] --> B["Phase B<br/>Temporal Envelope Search"]
    B --> C["Phase C<br/>Dataset Collection"]
    C --> D["Architecture Search<br/>& ResMLP Training"]
    D --> E["Downstream Inference<br/>& GenColorBench"]
```

### Architectural Specifics: FLUX.1-dev
| Property | Specification |
| :--- | :--- |
| **Latent Channels ($C$)** | 16 channels (packed 2x2 patches: $b \times \frac{h}{2} \cdot \frac{w}{2} \times 64$) |
| **VAE Compression** | $8\times$ spatial downsampling + $2\times2$ patch packing |
| **Scheduler** | Flow Matching (Euler) with dynamic shift |
| **Text Encoders** | CLIP-L + T5-XXL |
| **Winning Schedule** | `gate_frac = 0.50`, `n_partes = 1`, `perfil = ascendente` |

---

## 📁 Repository Structure

- [`flux_core.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/flux_core.py): Low-level core engine for FLUX (model loader, 2x2 patch pack/unpacking, flow-matching scheduler hooks).
- [`inference.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/inference.py): Standalone downstream targeted object color steering for arbitrary prompts.
- [`utils.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/utils.py): SAM-3 instance segmentation, sRGB $\leftrightarrow$ CIELAB colorimetry, and exact CIEDE2000 metric.
- [`iscc_nbs.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/iscc_nbs.py): ISCC-NBS Level 1 and Level 2 color dictionary.
- [`model_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/model_pca.py): 15D continuous color featurizer and `MLPShiftPCA` inference wrapper.
- [`fase_a_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/fase_a_pca.py): **Phase A**: Latent screening across 16 channels and PCA basis extraction.
- [`fase_b_config_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/fase_b_config_pca.py): **Phase B**: Temporal schedule $w(t)$ optimization across `gate_frac`, `n_partes`, and profile envelopes.
- [`coleccion_datos_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/coleccion_datos_mlp_pca.py): **Phase C Data Collection**: Multi-axial 3D spherical sampling in PCA space.
- [`train_and_search_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/train_and_search_mlp_pca.py): MLP architecture search and training.
- [`run_gencolorbench_pca.py`](file:///home/jsantamaria/projects/Color_Subspace/Flux/run_gencolorbench_pca.py): Automated GenColorBench evaluation harness.

---

## 🚀 Standalone Inference Quickstart

Run closed-loop object color steering with an arbitrary prompt and target color:

```bash
python inference.py \
    --prompt "a photo of a ceramic mug on a wooden desk" \
    --target-color "#A52A2A" \
    --seed 42 \
    --out-dir ./inference_outputs
```
# PixArt Latent Color Subspace Control (PixArt-α & PixArt-Σ)

Training-free, continuous numerical and perceptual color control for **PixArt-α** and **PixArt-Σ** text-to-image diffusion transformer architectures via latent subspace steering.

---

## 🔬 Model Architecture & Color Subspace Parameters

| Parameter | PixArt-α (`alpha`) | PixArt-Σ (`sigma`) |
| :--- | :--- | :--- |
| **Model ID** | `PixArt-alpha/PixArt-XL-2-1024-MS` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` |
| **Pipeline** | `PixArtAlphaPipeline` | `PixArtSigmaPipeline` |
| **Max Sequence Length** | **120 tokens** | **300 tokens** |
| **Denoising Steps** | **`STEPS = 20`** | **`STEPS = 20`** |
| **Guidance Scale** | **`GUIDANCE = 4.5`** | **`GUIDANCE = 4.5`** |
| **Latent Space Channels** | **4 channels** ($128 \times 128$ for $1024 \times 1024$) | **4 channels** ($128 \times 128$ for $1024 \times 1024$) |
| **VAE Architecture** | `AutoencoderKL` (`scaling_factor = 0.18215`) | `AutoencoderKL` (`scaling_factor = 0.18215`) |
| **Spatial Downsampling** | $8\times$ | $8\times$ |
| **Segmentation Engine** | SAM-3 (`facebook/sam3`) | SAM-3 (`facebook/sam3`) |

---

## 📁 Repository Structure

- [`pixart_core.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/pixart_core.py): PixArt pipeline loader (alpha/sigma), 4D latent channel/direction/PCA operators, temporal scheduling envelopes, multi-GPU orchestration.
- [`utils.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/utils.py): sRGB $\leftrightarrow$ CIELAB batch conversions, SAM-3 segmentation, robust PCA+MAD dominant color extraction, and CIEDE2000 metric.
- [`iscc_nbs.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/iscc_nbs.py): ISCC-NBS Level 1 (13) and Level 2 (30) reference centroids and nearest-color matcher.
- [`model_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/model_pca.py): 15D perceptual feature engineering and [`ResMLP_256`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/model_pca.py#L29-L49) inference wrapper.
- [`fase_a_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/fase_a_pca.py): Phase A latent sensitivity screening across 4 channels, SVD/PCA basis extraction ($\mathbf{U}_1, \mathbf{U}_2, \mathbf{U}_3$), scree plot, and cosine alignment matrix (`steps=20`, `guidance=4.5`).
- [`fase_b_config_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/fase_b_config_pca.py): Phase B temporal envelope optimization (`gate_frac`, `n_partes`, profile shape) (`steps=20`, `guidance=4.5`).
- [`coleccion_datos_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/coleccion_datos_mlp_pca.py): Phase C dataset generation across 100 object categories with multi-axial spherical direction sampling on $\mathbb{S}^2$ (`steps=20`, `guidance=4.5`).
- [`train_and_search_mlp_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/train_and_search_mlp_pca.py): Architecture search (Linear, Shallow, Medium, Deep, ResMLP) and training.
- [`val_mlp_shift_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/val_mlp_shift_pca.py): Closed-loop validation on PixArt evaluating numerical color accuracy against text prompting.
- [`run_gencolorbench_pca.py`](file:///home/jsantamaria/projects/Color_Subspace_local/PixArt/run_gencolorbench_pca.py): GenColorBench automated evaluation runner (`steps=20`, `guidance=4.5`).

---

## 🚀 Execution Workflow

### 1. Phase A: Latent Sensitivity Screening & PCA Subspace Discovery
```bash
conda activate diffusion_2
cd /home/jsantamaria/projects/Color_Subspace_local/PixArt

# For PixArt-alpha (120 tokens, steps=20, guidance=4.5)
python fase_a_pca.py --model-type alpha --out-dir ./fase_a_pca_out

# For PixArt-Sigma (300 tokens, steps=20, guidance=4.5)
python fase_a_pca.py --model-type sigma --out-dir ./fase_a_pca_out
```

### 2. Phase B: Temporal Envelope Schedule Search
```bash
python fase_b_config_pca.py --model-type alpha --out-dir ./fase_b_pca_out
```

### 3. Phase C: Dataset Generation & ResMLP Training
```bash
# Generate 3,125 paired training samples across 100 everyday scenes
python coleccion_datos_mlp_pca.py --model-type alpha --out-dir ./coleccion_datos_mlp_pca_out --n-total-samples 3125

# Train ResMLP_256 with architecture search
python train_and_search_mlp_pca.py --dataset-path ./coleccion_datos_mlp_pca_out/dataset_mlp_pca_pixart.csv --out-dir ./mlp_training_out --epochs 80
```

### 4. Closed-Loop Validation & GenColorBench
```bash
# Quantitative and qualitative validation
python val_mlp_shift_pca.py --model-type alpha --ckpt-path ./mlp_training_out/mlp_shift_pca_best.pt --out-dir ./val_mlp_shift_pca_out

# Full GenColorBench evaluation
python run_gencolorbench_pca.py --model-type alpha --benchmark-csv /path/to/gencolorbench_task.csv --ckpt-path ./mlp_training_out/mlp_shift_pca_best.pt --out-dir ./gencolorbench_pixart_out
```

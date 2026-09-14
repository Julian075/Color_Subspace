# Color Subspace Control for Z-Image

Latent Color Subspace methodology adapted for **Z-Image** (Tongyi-MAI/Z-Image), a 6B parameter single-stream Diffusion Transformer (S³-DiT).

## Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `Tongyi-MAI/Z-Image` |
| Architecture | S³-DiT (Scalable Single-Stream Diffusion Transformer) |
| Parameters | 6B |
| Latent Channels | 16 |
| Spatial Downsampling | 8× (1024×1024 → 128×128) |
| Latent Format | 4D Tensor `(B, 16, H/8, W/8)` — no patch packing |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (flow matching / rectified flow) |
| Text Encoder | Qwen series (bilingual EN/ZH) |
| Precision | `torch.bfloat16` |
| Default Steps | 30 |
| Default Guidance | 4.0 |
| Negative Prompt | Supported |

## Pipeline Execution Order

```
1. Phase A: Sensitivity Screening      (fase_a_pca.py)           → pca_axes.json
2. Phase B: Temporal Envelope Search    (fase_b_config_pca.py)    → fase_b_winning_schedule.json
3. Dense Calibration                    (experiment_dense_magnitude_curve.py) → dense_calibration_summary.json
4. Phase C: Dataset Collection          (coleccion_datos_mlp_pca.py)  → dataset_mlp_pca_25k.csv
5. Phase D: MLP Training & Search       (train_and_search_mlp_pca.py) → mlp_shift_pca_best.pt
6. Phase E: Validation                  (val_mlp_shift_pca.py)        → validation_results.csv
7. GenColorBench Evaluation             (run_gencolorbench_pca.py)    → manifest.csv
```

## Quick Start

```bash
# Phase A: Characterize Z-Image's 16D latent color subspace
python fase_a_pca.py --parallel

# Phase B: Optimize temporal envelope schedule
python fase_b_config_pca.py --parallel

# Dense Calibration: Map linear vs saturation magnitude zones
python experiment_dense_magnitude_curve.py --parallel

# Phase C: Collect 25k training samples
python coleccion_datos_mlp_pca.py --parallel

# Phase D: Train ResMLP-256 shift predictor
python train_and_search_mlp_pca.py

# Phase E: Validate closed-loop color control
python val_mlp_shift_pca.py

# Run GenColorBench benchmark
python run_gencolorbench_pca.py --auto-multi-gpu --task ncu

# Or run the full pipeline queue:
python run_zimage_pipeline_queue.py --gpu 0
```

## Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers
pip install transformers>=4.45.0 accelerate safetensors sentencepiece protobuf
pip install numpy pandas matplotlib scikit-image pillow
```

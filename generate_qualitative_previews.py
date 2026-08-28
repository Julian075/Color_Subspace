"""
GENERATE_QUALITATIVE_PREVIEWS.PY -- Visual qualitative inspection of PCA color shifts.

Generates side-by-side baseline vs shifted comparisons across all 3 PC axes
(+PC1, -PC1, +PC2, -PC2, +PC3, -PC3) at magnitudes m in [1.5, 3.0].
Creates high-resolution comparison grids with deltaE, SSIM, and PSNR annotations.
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flux_core import (
    build_envelope_bands, setup_flux, latent_hw, unpack_to_4d, decode_latents_4d,
    build_mask_latent, save_mask_preview, run_generation, get_gpu_list, spawn_workers,
)

try:
    from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000
except ImportError as _e:
    raise ImportError(f"Failed to import from utils.py: {_e}")

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
except ImportError as _e:
    raise ImportError(f"Failed to import from skimage: {_e}")

# =========================== CONFIG ===========================
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5
REF_MODE = "none"

OUT_DIR = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/qualitative_previews"
PCA_AXES_PATH = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/fase_a_pca_out/pca_axes.json"
WINNING_SCHEDULE_PATH = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/fase_b_pca_out/fase_b_winning_schedule.json"

# Load PCA Loading Vectors
with open(PCA_AXES_PATH) as f:
    pca_data = json.load(f)["axes"]

PCA_DIRECTIONS = {
    "+PC1": [(c, float(pca_data["PC1"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
    "-PC1": [(c, -float(pca_data["PC1"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
    "+PC2": [(c, float(pca_data["PC2"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
    "-PC2": [(c, -float(pca_data["PC2"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
    "+PC3": [(c, float(pca_data["PC3"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
    "-PC3": [(c, -float(pca_data["PC3"]["loadings"][c])) for c in range(NUM_LATENT_CHANNELS)],
}

# Load winning schedule
if os.path.exists(WINNING_SCHEDULE_PATH):
    with open(WINNING_SCHEDULE_PATH) as f:
        sched_cfg = json.load(f)
else:
    sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}

BANDS = build_envelope_bands(
    sched_cfg["gate_frac"],
    PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
    "ramp_down"
)

# Representative test objects
PREVIEW_OBJECTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, minimalist, no shadows, no texture", 0, "sphere"),
    ("a red toy car on a plain white background, studio lighting, product photo, no shadows", 1, "car"),
    ("a blue ceramic mug on a plain white background, studio lighting, product photo, no shadows", 2, "mug"),
    ("a bright yellow rubber duck on a plain white background, studio lighting, product photo, no shadows", 3, "duck"),
]

TEST_MAGNITUDES = [2.0, 4.0]


def gen(pipe, prompt, seed, device, direction=None, magnitude=None, bands=None, mask_latent=None):
    return run_generation(
        pipe, prompt, int(seed), RESOLUTION, RESOLUTION, device, STEPS, GUIDANCE,
        channel_idxs=None, direction=direction, combo=None, magnitude=magnitude, bands=bands,
        mask_latent=mask_latent, ref_mode=REF_MODE, ref_channels=None
    )


def calculate_psnr_masked(img1_np, img2_np, mask_2d=None):
    if mask_2d is not None and mask_2d.sum() > 0:
        diff = (img1_np.astype(np.float32) - img2_np.astype(np.float32))[mask_2d]
        mse = float(np.mean(diff ** 2))
    else:
        mse = float(np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 75)
    print(f"Generating Qualitative Previews into: {OUT_DIR}")
    print(f"Active Schedule: gate={sched_cfg['gate_frac']} n_partes={sched_cfg['n_partes']} perfil={sched_cfg['perfil_name']}")
    print("=" * 75)

    seg_models = setup_seg_models(DEVICE)
    pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)

    for prompt, seed, obj_word in PREVIEW_OBJECTS:
        print(f"\n>>> Generating Previews for: [{obj_word}] (seed={seed})")

        # 1. Baseline
        lat_base = gen(pipe, prompt, seed, DEVICE)
        img_base = decode_latents_4d(vae, unpack_to_4d(pipe, lat_base, RESOLUTION, RESOLUTION))
        base_path = os.path.join(OUT_DIR, f"{obj_word}_baseline.png")
        Image.fromarray(img_base).save(base_path)

        mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
        if mask_pixel is None:
            mask_pixel = np.ones((RESOLUTION, RESOLUTION), dtype=bool)

        base_lab = measure_color_gt(img_base, mask_pixel) or [50.0, 0.0, 0.0]
        lh, lw = latent_hw(pipe, RESOLUTION, RESOLUTION)
        mask_latent = build_mask_latent(mask_pixel, lh, lw, DEVICE)

        # 2. Shift generations for each PC direction and magnitude
        results_grid = []
        for dir_name in ["+PC1", "-PC1", "+PC2", "-PC2", "+PC3", "-PC3"]:
            direction = PCA_DIRECTIONS[dir_name]
            dir_row = []
            for mag in TEST_MAGNITUDES:
                lat_mod = gen(
                    pipe, prompt, seed, DEVICE, direction=direction, magnitude=mag,
                    bands=BANDS, mask_latent=mask_latent
                )
                img_mod = decode_latents_4d(vae, unpack_to_4d(pipe, lat_mod, RESOLUTION, RESOLUTION))

                mod_lab = measure_color_gt(img_mod, mask_pixel) or base_lab
                dE = float(ciede2000(base_lab, mod_lab))
                ssim_val = float(structural_similarity(img_base, img_mod, channel_axis=2, data_range=255))
                psnr_val = calculate_psnr_masked(img_base, img_mod, mask_pixel)

                # Save individual shifted image
                shift_fn = f"{obj_word}_{dir_name}_mag{mag:.1f}.png"
                Image.fromarray(img_mod).save(os.path.join(OUT_DIR, shift_fn))

                dir_row.append({
                    "dir_name": dir_name,
                    "magnitude": mag,
                    "img": img_mod,
                    "dE": dE,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "lab": mod_lab,
                })
                print(f"  {dir_name} m={mag:.1f} -> ΔE={dE:.2f} | SSIM={ssim_val:.3f} | PSNR={psnr_val:.1f} dB", flush=True)

            results_grid.append(dir_row)

        # 3. Create high-resolution comparison collage grid
        # 7 columns: Baseline | +PC1 | -PC1 | +PC2 | -PC2 | +PC3 | -PC3
        # 2 rows for the 2 magnitudes
        fig, axes = plt.subplots(2, 7, figsize=(26, 8.5))
        fig.suptitle(f"Qualitative PCA Latent Shifts: Object '{obj_word}' (seed={seed})\n"
                     f"Baseline Lab = ({base_lab[0]:.1f}, {base_lab[1]:.1f}, {base_lab[2]:.1f}) | Schedule: gate={sched_cfg['gate_frac']}, n_partes={sched_cfg['n_partes']}",
                     fontsize=15, fontweight="bold")

        # Column 0: Baseline
        for r in range(2):
            axes[r, 0].imshow(img_base)
            axes[r, 0].set_title(f"Baseline\nLab=({base_lab[0]:.1f}, {base_lab[1]:.1f}, {base_lab[2]:.1f})", fontsize=11, fontweight="bold")
            axes[r, 0].axis("off")

        # Columns 1 to 6: Shifted images
        col_names = ["+PC1", "-PC1", "+PC2", "-PC2", "+PC3", "-PC3"]
        for col_idx, dir_name in enumerate(col_names):
            for row_idx, mag in enumerate(TEST_MAGNITUDES):
                cell = results_grid[col_idx][row_idx]
                ax = axes[row_idx, col_idx + 1]
                ax.imshow(cell["img"])
                title_text = f"{dir_name} (m={mag:.1f})\nΔE={cell['dE']:.1f} | SSIM={cell['ssim']:.3f}\nPSNR={cell['psnr']:.1f}dB"
                ax.set_title(title_text, fontsize=10, fontweight="bold")
                ax.axis("off")

        fig.tight_layout()
        grid_path = os.path.join(OUT_DIR, f"{obj_word}_pca_comparison_grid.png")
        fig.savefig(grid_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f">>> Saved comparison grid: {grid_path}")

    print("\n" + "=" * 75)
    print("ALL QUALITATIVE PREVIEWS GENERATED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()

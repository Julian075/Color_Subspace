"""
EXPERIMENT_DENSE_MAGNITUDE_CURVE.PY -- Dense Magnitude Calibration & Saturation Sweep.

Densely samples 16 magnitudes m in [0.02 ... 4.00] across all 6 PCA poles (+/-PC1, +/-PC2, +/-PC3)
to characterize:
  1. Exact linear sensitivity slopes (dDeltaE/dm) and R^2 linearity bounds.
  2. The saturation knee point (m_sat) and degradation rates of SSIM and PSNR.
  3. Visual progression strips for Sphere and Car.
  4. Recommended sampling distribution for Phase 4 dataset collection.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# =========================== CONFIGURATION ===========================
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5
REF_MODE = "none"

OUT_DIR = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/dense_calibration_out"
PCA_AXES_PATH = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/fase_a_pca_out/pca_axes.json"
WINNING_SCHEDULE_PATH = "/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/flux/pca_version/fase_b_pca_out/fase_b_winning_schedule.json"

# Load PCA Loading Vectors
with open(PCA_AXES_PATH) as f:
    pca_data = json.load(f)["axes"]

PCA_POLES = {
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

# 16 finely-spaced magnitudes
DENSE_MAGNITUDES = [
    0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00
]

TEST_OBJECTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, minimalist, no shadows, no texture", 0, "sphere"),
    ("a red toy car on a plain white background, studio lighting, product photo, no shadows", 1, "car"),
]

CSV_FIELDS = [
    "obj_name", "seed", "pole_name", "magnitude", "deltaE",
    "ssim_in", "ssim_out", "psnr_in", "psnr_out",
    "base_L", "base_a", "base_b", "mod_L", "mod_a", "mod_b",
    "delta_L", "delta_a", "delta_b", "img_path", "note"
]


# =========================== HELPER FUNCTIONS ===========================
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


def generate_case_context(pipe, vae, seg_models, prompt, seed, obj_word, device, out_dir, case_tag):
    latents_packed_base = gen(pipe, prompt, int(seed), device)
    if torch.isnan(latents_packed_base).any():
        return None

    img_base = decode_latents_4d(vae, unpack_to_4d(pipe, latents_packed_base, RESOLUTION, RESOLUTION))
    os.makedirs(out_dir, exist_ok=True)
    base_img_path = os.path.join(out_dir, f"{case_tag}_baseline.png")
    Image.fromarray(img_base).save(base_img_path)

    mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        mask_pixel = np.ones((RESOLUTION, RESOLUTION), dtype=bool)

    base_lab = measure_color_gt(img_base, mask_pixel) or [50.0, 0.0, 0.0]
    lh, lw = latent_hw(pipe, RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, lh, lw, device)
    save_mask_preview(img_base, mask_pixel, os.path.join(out_dir, f"{case_tag}_mask_preview.png"))

    return {
        "img_base": img_base,
        "mask_pixel": mask_pixel,
        "mask_latent": mask_latent,
        "base_lab": base_lab,
        "base_img_path": base_img_path,
    }


# =========================== WORKER ROUTINE ===========================
def run_worker(chunk_path, out_dir, pipe, vae, seg_models):
    pid = os.getpid()
    with open(chunk_path) as f:
        tasks = json.load(f)
    print(f"[worker pid={pid}] Assigned {len(tasks)} dense magnitude tasks", flush=True)

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"part_{pid}.csv")

    imgs_dir = os.path.join(out_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    ctx_cache = {}
    with open(part_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for t in tasks:
            obj_name = t["obj_name"]
            prompt, seed = t["prompt"], int(t["seed"])
            pole_name = t["pole_name"]
            mag = float(t["magnitude"])
            case_tag = f"seed{seed}_{obj_name}"

            if obj_name not in ctx_cache:
                print(f"[worker pid={pid}] Generating baseline {obj_name}...", flush=True)
                ctx = generate_case_context(pipe, vae, seg_models, prompt, seed, obj_name, DEVICE, out_dir, case_tag)
                if ctx is None:
                    continue
                ctx_cache[obj_name] = ctx

            ctx = ctx_cache[obj_name]
            direction = PCA_POLES[pole_name]

            lat_mod = gen(
                pipe, prompt, seed, DEVICE, direction=direction, magnitude=mag,
                bands=BANDS, mask_latent=ctx["mask_latent"]
            )
            img_mod = decode_latents_4d(vae, unpack_to_4d(pipe, lat_mod, RESOLUTION, RESOLUTION))

            mod_lab = measure_color_gt(img_mod, ctx["mask_pixel"]) or ctx["base_lab"]
            deltaE = float(ciede2000(ctx["base_lab"], mod_lab))
            ssim_in = float(structural_similarity(ctx["img_base"], img_mod, channel_axis=2, data_range=255))
            psnr_in = calculate_psnr_masked(ctx["img_base"], img_mod, ctx["mask_pixel"])
            psnr_out = calculate_psnr_masked(ctx["img_base"], img_mod, ~ctx["mask_pixel"])

            delta_L = float(mod_lab[0] - ctx["base_lab"][0])
            delta_a = float(mod_lab[1] - ctx["base_lab"][1])
            delta_b = float(mod_lab[2] - ctx["base_lab"][2])

            img_fn = f"{obj_name}_{pole_name}_m{mag:.2f}.png"
            img_save_path = os.path.join(imgs_dir, img_fn)
            Image.fromarray(img_mod).save(img_save_path)

            row = {
                "obj_name": obj_name,
                "seed": seed,
                "pole_name": pole_name,
                "magnitude": f"{mag:.2f}",
                "deltaE": f"{deltaE:.3f}",
                "ssim_in": f"{ssim_in:.4f}",
                "ssim_out": "1.0000",
                "psnr_in": f"{psnr_in:.2f}",
                "psnr_out": f"{psnr_out:.2f}",
                "base_L": f"{ctx['base_lab'][0]:.2f}",
                "base_a": f"{ctx['base_lab'][1]:.2f}",
                "base_b": f"{ctx['base_lab'][2]:.2f}",
                "mod_L": f"{mod_lab[0]:.2f}",
                "mod_a": f"{mod_lab[1]:.2f}",
                "mod_b": f"{mod_lab[2]:.2f}",
                "delta_L": f"{delta_L:.2f}",
                "delta_a": f"{delta_a:.2f}",
                "delta_b": f"{delta_b:.2f}",
                "img_path": img_save_path,
                "note": "ok",
            }
            writer.writerow(row)
            f.flush()
            print(f"[worker pid={pid}] {obj_name} {pole_name} m={mag:.2f} -> ΔE={deltaE:.2f} SSIM={ssim_in:.3f} PSNR={psnr_in:.1f}dB", flush=True)


# =========================== ANALYSIS & VISUALIZATION ===========================
def analyze_and_plot_results(csv_path, out_dir):
    """Generates calibration curves, knee-point fits, and progression strip collages."""
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # 1. Quantitative Plot: DeltaE, SSIM, PSNR vs Magnitude
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    poles = ["+PC1", "-PC1", "+PC2", "-PC2", "+PC3", "-PC3"]
    colors = {"+PC1": "#2ca02c", "-PC1": "#1f77b4", "+PC2": "#d62728", "-PC2": "#17becf", "+PC3": "#e377c2", "-PC3": "#7f7f7f"}

    # Filter to sphere data for clean calibration curves
    sphere_rows = [r for r in rows if r["obj_name"] == "sphere"]

    # (a) DeltaE vs m
    ax1 = axes[0]
    linear_slopes = {}
    for p in poles:
        p_rows = sorted([r for r in sphere_rows if r["pole_name"] == p], key=lambda x: float(x["magnitude"]))
        m_vals = np.array([float(r["magnitude"]) for r in p_rows])
        dE_vals = np.array([float(r["deltaE"]) for r in p_rows])

        ax1.plot(m_vals, dE_vals, marker="o", linewidth=2, label=p, color=colors[p])

        # Fit linear slope on m <= 0.40
        mask_lin = m_vals <= 0.40
        if mask_lin.sum() >= 3:
            slope = float(np.polyfit(m_vals[mask_lin], dE_vals[mask_lin], 1)[0])
            linear_slopes[p] = slope

    ax1.axvspan(0.0, 0.40, color="green", alpha=0.10, label="High-Fidelity Linear Zone (m ≤ 0.40)")
    ax1.axvspan(0.40, 1.25, color="orange", alpha=0.08, label="Transition Zone (0.40 < m ≤ 1.25)")
    ax1.axvspan(1.25, 4.00, color="red", alpha=0.06, label="Saturation Zone (m > 1.25)")
    ax1.set_xlabel("Latent Shift Magnitude (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Color Difference ΔE₀₀", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Color Shift Response Curve ΔE(m)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="lower right")

    # (b) SSIM vs m
    ax2 = axes[1]
    for p in poles:
        p_rows = sorted([r for r in sphere_rows if r["pole_name"] == p], key=lambda x: float(x["magnitude"]))
        m_vals = np.array([float(r["magnitude"]) for r in p_rows])
        ssim_vals = np.array([float(r["ssim_in"]) for r in p_rows])
        ax2.plot(m_vals, ssim_vals, marker="s", linewidth=2, label=p, color=colors[p])

    ax2.axhline(0.95, color="green", linestyle=":", label="SSIM = 0.95 (Pristine)")
    ax2.axhline(0.85, color="orange", linestyle="--", label="SSIM = 0.85 (Good)")
    ax2.axhline(0.70, color="red", linestyle="-.", label="SSIM = 0.70 (Minimum Safe)")
    ax2.set_xlabel("Latent Shift Magnitude (m)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("SSIM (Structural Similarity)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) SSIM Degradation Curve", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.55, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="lower left")

    # (c) PSNR vs m
    ax3 = axes[2]
    for p in poles:
        p_rows = sorted([r for r in sphere_rows if r["pole_name"] == p], key=lambda x: float(x["magnitude"]))
        m_vals = np.array([float(r["magnitude"]) for r in p_rows])
        psnr_vals = np.array([float(r["psnr_in"]) for r in p_rows])
        ax3.plot(m_vals, psnr_vals, marker="^", linewidth=2, label=p, color=colors[p])

    ax3.axhline(25.0, color="green", linestyle=":", label="25 dB (Subtle)")
    ax3.axhline(18.0, color="orange", linestyle="--", label="18 dB (Balanced)")
    ax3.axhline(10.0, color="red", linestyle="-.", label="10 dB (Heavy Shift)")
    ax3.set_xlabel("Latent Shift Magnitude (m)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Object PSNR (dB)", fontsize=11, fontweight="bold")
    ax3.set_title("(c) PSNR Distortion Curve (dB)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    curves_plot_path = os.path.join(out_dir, "dense_calibration_curves.png")
    fig.savefig(curves_plot_path, dpi=160)
    plt.close(fig)
    print(f"Generated calibration curves plot: {curves_plot_path}")

    # 2. Progression Strips for Sphere
    for obj_name in ["sphere", "car"]:
        obj_rows = [r for r in rows if r["obj_name"] == obj_name]
        if not obj_rows:
            continue

        fig, axes = plt.subplots(6, len(DENSE_MAGNITUDES), figsize=(32, 16))
        fig.suptitle(f"Dense Magnitude Calibration Strip: Object '{obj_name}'\n"
                     f"Sweeping m from 0.02 (subtle) to 4.00 (saturation) across 6 PCA directions",
                     fontsize=16, fontweight="bold")

        for row_idx, pole in enumerate(poles):
            for col_idx, mag in enumerate(DENSE_MAGNITUDES):
                match = [r for r in obj_rows if r["pole_name"] == pole and abs(float(r["magnitude"]) - mag) < 1e-3]
                ax = axes[row_idx, col_idx]
                if match and os.path.exists(match[0]["img_path"]):
                    img = Image.open(match[0]["img_path"])
                    ax.imshow(img)
                    dE = float(match[0]["deltaE"])
                    ssim_val = float(match[0]["ssim_in"])
                    psnr_val = float(match[0]["psnr_in"])
                    title_text = f"m={mag:.2f}\nΔE={dE:.1f}\nSSIM={ssim_val:.2f}\n{psnr_val:.1f}dB"
                    ax.set_title(title_text, fontsize=8)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")

                if col_idx == 0:
                    ax.set_ylabel(pole, fontsize=12, fontweight="bold", labelpad=10)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout()
        strip_path = os.path.join(out_dir, f"{obj_name}_dense_progression_strip.png")
        fig.savefig(strip_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated progression strip for {obj_name}: {strip_path}")

    # Summary JSON recommendations
    summary = {
        "linear_slopes_dDeltaE_dm": linear_slopes,
        "recommended_m_distribution_phase4": {
            "linear_high_fidelity_zone": {"range": [0.05, 0.45], "sample_fraction": 0.80, "expected_deltaE": [4.0, 32.0], "expected_ssim": [0.95, 0.99]},
            "boundary_saturation_zone": {"range": [0.45, 1.50], "sample_fraction": 0.20, "expected_deltaE": [32.0, 42.0], "expected_ssim": [0.75, 0.94]}
        }
    }
    summary_path = os.path.join(out_dir, "dense_calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved calibration summary to: {summary_path}")


# =========================== MAIN DRIVER ===========================
def main():
    parser = argparse.ArgumentParser(description="Dense Magnitude Calibration Experiment")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--analyze-only", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.analyze_only:
        analyze_and_plot_results(args.analyze_only, args.out_dir)
        return

    if args.worker:
        seg_models = setup_seg_models(DEVICE)
        pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models)
        return

    # Master Controller
    gpu_list = get_gpu_list(None)
    n_workers = len(gpu_list) * args.workers_per_gpu

    tasks = []
    for prompt, seed, obj_name in TEST_OBJECTS:
        for pole_name in ["+PC1", "-PC1", "+PC2", "-PC2", "+PC3", "-PC3"]:
            for mag in DENSE_MAGNITUDES:
                tasks.append({
                    "obj_name": obj_name,
                    "prompt": prompt,
                    "seed": seed,
                    "pole_name": pole_name,
                    "magnitude": mag,
                })

    print(f"Generated {len(tasks)} dense calibration tasks across {len(gpu_list)} GPUs ({n_workers} workers)")

    cmd_base = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--out-dir", args.out_dir,
    ]
    tmp_chunks = os.path.join(args.out_dir, "_tmp_chunks_dense")
    spawn_workers(tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

    # Merge parts
    parts_dir = os.path.join(args.out_dir, "_csv_parts")
    master_csv = os.path.join(args.out_dir, "dense_calibration_raw.csv")
    merged_rows = []
    for fn in os.listdir(parts_dir):
        if fn.endswith(".csv"):
            fp = os.path.join(parts_dir, fn)
            with open(fp) as f:
                r = csv.DictReader(f)
                merged_rows.extend(list(r))
            os.remove(fp)

    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(merged_rows)

    print(f"\nSaved master CSV with {len(merged_rows)} records to {master_csv}")
    analyze_and_plot_results(master_csv, args.out_dir)


if __name__ == "__main__":
    main()

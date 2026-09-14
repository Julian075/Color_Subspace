"""
FASE_B_CONFIG_PCA.PY -- Phase B: Temporal Envelope Optimization for SD3 in PCA Latent Space.

Explores grid of temporal envelope hyperparameters on SD3 (stabilityai/stable-diffusion-3-medium-diffusers):
  - gate_frac in [0.25, 0.4, 0.5, 0.6, 0.75]
  - n_partes in [1, 2, 3]
  - perfil_name in ['plano', 'ascendente', 'descendente', 'triangular']

Evaluates:
  1. Target object color shift magnitude (dE00)
  2. Inside-object structure preservation (SSIM_in)
  3. Outside-object background fidelity (PSNR_out, SSIM_out)
  4. Composite Color Score = dE00 * sqrt(SSIM_in)
Exports winning schedule to fase_b_winning_schedule.json and 3-panel publication plot.
"""

import os
import sys
import gc
import json
import math
import argparse
import subprocess
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Local imports
from sd3_core import (
    MODEL_ID, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION, STEPS, GUIDANCE,
    setup_sd3, decode_latents_4d, build_envelope_bands, build_mask_latent,
    latent_hw, run_generation, PERFIL_GENERATORS, spawn_workers
)
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCA_AXES_PATH = os.path.join(BASE_DIR, "fase_a_pca_out", "pca_axes.json")

TEST_PROMPTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting", "sphere", 42),
    ("a ceramic coffee mug on a neutral gray studio background, minimalist product photography", "mug", 101),
    ("a gray fabric armchair on a plain white studio background", "armchair", 202),
    ("a simple gray sports shoe on a clean white background", "shoe", 303),
]

# Shifts along PCA basis
TEST_SHIFTS = [
    (0.4, 0.0, 0.0),   # Lightness shift (+L*)
    (-0.4, 0.0, 0.0),  # Darkness shift (-L*)
    (0.0, 0.5, 0.0),   # Red-Green shift (+a*)
    (0.0, 0.0, 0.5),   # Yellow-Blue shift (+b*)
]

GATE_FRACS = [0.25, 0.40, 0.50, 0.60, 0.75]
N_PARTES_LIST = [1, 2, 3]
PERFILES = ["plano", "ascendente", "descendente", "triangular"]


def load_pca_basis(axes_path=PCA_AXES_PATH):
    if os.path.exists(axes_path):
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        print(f"[INFO] PCA axes file not found at {axes_path}. Using placeholder standard axes.")
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return u1, u2, u3


def evaluate_trial(pipe, vae, seg_models, device, prompt, obj_word, seed,
                   u1, u2, u3, m1, m2, m3, bands, steps=STEPS, guidance=GUIDANCE):
    # 1. Baseline
    latents_base = run_generation(pipe, prompt, seed, RESOLUTION, RESOLUTION, device, steps, guidance)
    img_base = decode_latents_4d(vae, latents_base)

    mask_pixel = utils.get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        h, w = img_base.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        mask_pixel = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= (min(h, w) * 0.35) ** 2

    lab_base = utils.measure_color_gt(img_base, mask_pixel)
    if lab_base is None:
        lab_base = (50.0, 0.0, 0.0)

    # 2. Shifted
    latent_h, latent_w = latent_hw(RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)

    latents_mod = run_generation(
        pipe, prompt, seed, RESOLUTION, RESOLUTION, device, steps, guidance,
        pca_basis=(u1, u2, u3), m_vector=(m1, m2, m3), bands=bands, mask_latent=mask_latent
    )
    img_mod = decode_latents_4d(vae, latents_mod)
    lab_mod = utils.measure_color_gt(img_mod, mask_pixel)
    if lab_mod is None:
        lab_mod = lab_base

    dE00 = utils.ciede2000(lab_base, lab_mod)

    # Metrics
    mask_inv = ~mask_pixel
    ssim_in = ssim(img_base, img_mod, channel_axis=2, data_range=255)
    ssim_out = ssim(img_base * mask_inv[:, :, None], img_mod * mask_inv[:, :, None], channel_axis=2, data_range=255)
    psnr_out = psnr(img_base * mask_inv[:, :, None], img_mod * mask_inv[:, :, None], data_range=255)

    color_score = float(dE00 * math.sqrt(max(ssim_in, 0.01)))

    return {
        "dE00": float(dE00),
        "ssim_in": float(ssim_in),
        "ssim_out": float(ssim_out),
        "psnr_out": float(psnr_out),
        "color_score": color_score,
    }


def worker_main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    u1, u2, u3 = load_pca_basis(args.axes_path)
    pipe, vae = setup_sd3(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)

    # Build schedules
    schedules = []
    for gf in GATE_FRACS:
        for np_val in N_PARTES_LIST:
            for p_name in PERFILES:
                schedules.append({
                    "gate_frac": gf,
                    "n_partes": np_val,
                    "perfil_name": p_name,
                })

    start_idx = args.task_start
    end_idx = min(args.task_end, len(schedules))
    worker_scheds = schedules[start_idx:end_idx]
    print(f"[WORKER {args.worker_id}] Evaluating schedules {start_idx} to {end_idx} ({len(worker_scheds)} total)")

    results = []
    for s_idx, sched in enumerate(worker_scheds):
        bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")

        scores = []
        for prompt, obj_word, seed in TEST_PROMPTS:
            for m1, m2, m3 in TEST_SHIFTS:
                m_res = evaluate_trial(
                    pipe, vae, seg_models, device, prompt, obj_word, seed,
                    u1, u2, u3, m1, m2, m3, bands, steps=args.steps, guidance=args.guidance
                )
                scores.append(m_res)

        avg_color_score = float(np.mean([s["color_score"] for s in scores]))
        avg_dE00 = float(np.mean([s["dE00"] for s in scores]))
        avg_ssim_in = float(np.mean([s["ssim_in"] for s in scores]))
        avg_ssim_out = float(np.mean([s["ssim_out"] for s in scores]))
        avg_psnr_out = float(np.mean([s["psnr_out"] for s in scores]))

        results.append({
            "gate_frac": sched["gate_frac"],
            "n_partes": sched["n_partes"],
            "perfil_name": sched["perfil_name"],
            "avg_color_score": avg_color_score,
            "avg_dE00": avg_dE00,
            "avg_ssim_in": avg_ssim_in,
            "avg_ssim_out": avg_ssim_out,
            "avg_psnr_out": avg_psnr_out,
        })
        print(f"[WORKER {args.worker_id}] Sched {s_idx+1}/{len(worker_scheds)}: {sched} -> Score: {avg_color_score:.3f} | dE: {avg_dE00:.2f} | PSNR_bg: {avg_psnr_out:.1f} dB")

    out_csv = os.path.join(args.out_dir, f"fase_b_worker_{args.worker_id}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[WORKER {args.worker_id}] Saved results to: {out_csv}")


def generate_schedule_summary_plot(summary_csv: str, out_dir: str):
    """3-panel visual summary of the Phase B schedule sweep for SD3."""
    if not os.path.exists(summary_csv):
        return

    df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    params = [
        ("n_partes", "1. Number of Bands (n_partes)"),
        ("gate_frac", "2. Gate Fraction (gate_frac)"),
        ("perfil_name", "3. Profile Shape (perfil)"),
    ]

    for ax_idx, (col, title) in enumerate(params):
        grouped = df.groupby(col).agg({"avg_dE00": "mean", "avg_ssim_in": "mean"}).reset_index()
        cands = grouped[col].tolist()
        delta_means = grouped["avg_dE00"].tolist()
        ssim_means = grouped["avg_ssim_in"].tolist()

        ax = axes[ax_idx]
        x = np.arange(len(cands))
        ax.bar(x - 0.18, delta_means, width=0.35, color="#2b5c8f", label="Mean ΔE")
        ax2 = ax.twinx()
        ax2.plot(x + 0.18, ssim_means, color="#d95f02", marker="o", linewidth=2.5, label="Mean SSIM")
        ax.set_xticks(x)
        ax.set_xticklabels(cands, rotation=25 if col == "perfil_name" else 0, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel("Mean ΔE (Color Shift)", color="#2b5c8f", fontsize=11)
        ax2.set_ylabel("Mean SSIM", color="#d95f02", fontsize=11)
        ax2.set_ylim(0.5, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "fase_b_pca_schedule_search_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f">>> Generated schedule search diagnostic plot: {plot_path}")


def aggregate_and_select_winner(out_dir: str):
    csv_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("fase_b_worker_") and f.endswith(".csv")]
    if not csv_files:
        print("[ERROR] No worker CSV files found in", out_dir)
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by="avg_color_score", ascending=False, inplace=True)

    summary_csv = os.path.join(out_dir, "fase_b_all_schedules.csv")
    df.to_csv(summary_csv, index=False)

    winner = df.iloc[0].to_dict()
    winning_schedule = {
        "gate_frac": float(winner["gate_frac"]),
        "n_partes": int(winner["n_partes"]),
        "perfil_name": str(winner["perfil_name"]),
        "avg_color_score": float(winner["avg_color_score"]),
        "avg_dE00": float(winner["avg_dE00"]),
        "avg_ssim_in": float(winner["avg_ssim_in"]),
        "avg_ssim_out": float(winner["avg_ssim_out"]),
        "avg_psnr_out": float(winner["avg_psnr_out"]),
    }

    winner_json = os.path.join(out_dir, "fase_b_winning_schedule.json")
    with open(winner_json, "w") as f:
        json.dump(winning_schedule, f, indent=2)
    print(f"\n[WINNER] Selected winning schedule saved to {winner_json}:")
    print(json.dumps(winning_schedule, indent=2))

    generate_schedule_summary_plot(summary_csv, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase B: Temporal Envelope Search for SD3")
    parser.add_argument("--out-dir", default="./fase_b_pca_out")
    parser.add_argument("--axes-path", default=PCA_AXES_PATH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--analyze-only", type=str, default=None)

    # Worker arguments
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=999999)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.analyze_only:
        generate_schedule_summary_plot(args.analyze_only, args.out_dir)
        return

    if args.worker_id is not None:
        worker_main(args)
    else:
        n_schedules = len(GATE_FRACS) * len(N_PARTES_LIST) * len(PERFILES)
        extra_args = [
            "--out-dir", args.out_dir,
            "--axes-path", args.axes_path,
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
        ]
        spawn_workers(__file__, n_schedules, extra_args)
        aggregate_and_select_winner(args.out_dir)


if __name__ == "__main__":
    main()

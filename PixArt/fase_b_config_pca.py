"""
FASE_B_CONFIG_PCA.PY -- Phase B: Temporal Envelope Optimization for PixArt.

Explores grid of temporal envelope hyperparameters on PixArt (PixArt-alpha / PixArt-Sigma):
  - gate_frac in [0.25, 0.4, 0.5, 0.6, 0.75]
  - n_partes in [1, 2, 3]
  - perfil_name in ['plano', 'ascendente', 'descendente', 'triangular']

Evaluates:
  1. Target object color shift magnitude (dE00)
  2. Inside-object structure preservation (SSIM_in)
  3. Outside-object background fidelity (PSNR_out, SSIM_out)
  4. Composite Color Score = dE00 * sqrt(SSIM_in)
Exports winning schedule to fase_b_winning_schedule.json.
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
from pixart_core import (
    MODEL_ID_ALPHA, MODEL_ID_SIGMA, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION,
    STEPS, GUIDANCE, MAX_SEQ_LEN_ALPHA, MAX_SEQ_LEN_SIGMA,
    setup_pixart, decode_latents_4d, build_envelope_bands, build_mask_latent,
    latent_hw, run_generation, PERFIL_GENERATORS, spawn_workers
)
import utils

PCA_AXES_PATH = os.path.join(os.path.dirname(__file__), "fase_a_pca_out", "pca_axes.json")

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


def generate_case_context(pipe, vae, seg_models, prompt, obj_word, seed, device, max_seq_len, steps, guidance):
    """Generates and caches baseline image, mask, and baseline CIELAB color."""
    latents_base = run_generation(
        pipe, prompt, seed, RESOLUTION, RESOLUTION, device, steps, guidance,
        max_sequence_length=max_seq_len
    )
    img_base = decode_latents_4d(vae, latents_base)
    mask_pixel = utils.get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        h, w = img_base.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        mask_pixel = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= (min(h, w) * 0.35) ** 2

    lab_base = utils.measure_color_gt(img_base, mask_pixel)
    if lab_base is None:
        lab_base = (50.0, 0.0, 0.0)

    latent_h, latent_w = latent_hw(RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)

    return {
        "img_base": img_base,
        "mask_pixel": mask_pixel,
        "mask_latent": mask_latent,
        "lab_base": lab_base,
    }


def evaluate_shifted_trial(pipe, vae, device, prompt, seed, u1, u2, u3, m1, m2, m3, bands, ctx, steps=STEPS, guidance=GUIDANCE, max_seq_len=MAX_SEQ_LEN_ALPHA):
    """Evaluates shifted color generation against cached baseline context."""
    img_base = ctx["img_base"]
    mask_pixel = ctx["mask_pixel"]
    mask_latent = ctx["mask_latent"]
    lab_base = ctx["lab_base"]

    latents_mod = run_generation(
        pipe, prompt, seed, RESOLUTION, RESOLUTION, device, steps, guidance,
        max_sequence_length=max_seq_len,
        pca_basis=(u1, u2, u3), m_vector=(m1, m2, m3), bands=bands, mask_latent=mask_latent
    )
    img_mod = decode_latents_4d(vae, latents_mod)
    lab_mod = utils.measure_color_gt(img_mod, mask_pixel)
    if lab_mod is None:
        lab_mod = lab_base

    dE00 = utils.ciede2000(lab_base, lab_mod)

    mask_inv = ~mask_pixel
    ssim_in = ssim(img_base, img_mod, channel_axis=2, data_range=255)
    ssim_out = ssim(img_base * mask_inv[:, :, None], img_mod * mask_inv[:, :, None], channel_axis=2, data_range=255)
    
    # Safe PSNR computation
    diff_out = (img_base.astype(np.float32) - img_mod.astype(np.float32)) * mask_inv[:, :, None]
    mse_out = float(np.mean(diff_out ** 2))
    psnr_out = float(10.0 * np.log10((255.0 ** 2) / mse_out)) if mse_out > 1e-10 else 99.0

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
    max_seq_len = MAX_SEQ_LEN_SIGMA if args.model_type == "sigma" else MAX_SEQ_LEN_ALPHA

    pipe, vae = setup_pixart(model_type=args.model_type, device=device, dtype=DTYPE, num_latent_channels=NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models("cpu")

    # Pre-generate baseline contexts once
    print(f"[WORKER {args.worker_id}] Pre-generating baseline contexts for {len(TEST_PROMPTS)} cases...")
    ctx_cache = {}
    for p_idx, (prompt, obj_word, seed) in enumerate(TEST_PROMPTS):
        ctx_cache[p_idx] = generate_case_context(
            pipe, vae, seg_models, prompt, obj_word, seed, device, max_seq_len, args.steps, args.guidance
        )
        print(f"  [Case {p_idx+1}/{len(TEST_PROMPTS)}] Cached context for '{obj_word}' (seed {seed})")

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
    out_csv = os.path.join(args.out_dir, f"fase_b_worker_{args.worker_id}.csv")
    print(f"[WORKER {args.worker_id}] Evaluating schedules {start_idx} to {end_idx} ({len(worker_scheds)} total)")

    results = []
    done_keys = set()
    if os.path.exists(out_csv):
        try:
            prev_df = pd.read_csv(out_csv)
            for _, row in prev_df.iterrows():
                results.append(row.to_dict())
                done_keys.add((float(row["gate_frac"]), int(row["n_partes"]), str(row["perfil_name"])))
            print(f"[WORKER {args.worker_id}] Resumed {len(results)} previously evaluated schedules from {out_csv}")
        except Exception as e:
            print(f"[WORKER {args.worker_id}] Note: could not load previous CSV: {e}")

    for s_idx, sched in enumerate(worker_scheds):
        s_key = (float(sched["gate_frac"]), int(sched["n_partes"]), str(sched["perfil_name"]))
        if s_key in done_keys:
            print(f"[WORKER {args.worker_id}] Sched {s_idx+1}/{len(worker_scheds)}: {sched} [ALREADY DONE - SKIPPING]")
            continue

        bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")

        scores = []
        for p_idx, (prompt, obj_word, seed) in enumerate(TEST_PROMPTS):
            ctx = ctx_cache[p_idx]
            for m1, m2, m3 in TEST_SHIFTS:
                m_res = evaluate_shifted_trial(
                    pipe, vae, device, prompt, seed,
                    u1, u2, u3, m1, m2, m3, bands, ctx, steps=args.steps, guidance=args.guidance,
                    max_seq_len=max_seq_len
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
        done_keys.add(s_key)
        print(f"[WORKER {args.worker_id}] Sched {s_idx+1}/{len(worker_scheds)}: {sched} -> Score: {avg_color_score:.3f} | dE: {avg_dE00:.2f} | PSNR_bg: {avg_psnr_out:.1f} dB")
        pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"[WORKER {args.worker_id}] Saved results to: {out_csv}")


def generate_phase_b_plots(df: pd.DataFrame, out_dir: str, model_type: str = "alpha", winner: dict = None):
    try:
        df_clean = df.drop_duplicates(subset=["gate_frac", "n_partes", "perfil_name"]).copy()
        df_clean.sort_values(by="avg_color_score", ascending=False, inplace=True)

        all_csv_path = os.path.join(out_dir, "fase_b_all_schedules.csv")
        df_clean.to_csv(all_csv_path, index=False)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Phase B: Temporal Envelope Optimization — PixArt-{model_type.upper()}", fontsize=16, fontweight="bold", y=0.98)

        # Panel 1: Top 10 Schedules Ranked by Composite Score
        top10 = df_clean.head(10).copy()
        top10["label"] = top10.apply(lambda r: f"{r['perfil_name']} (N={int(r['n_partes'])}, G={r['gate_frac']:.2f})", axis=1)
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(top10))]
        bars = axes[0, 0].barh(range(len(top10)), top10["avg_color_score"], color=colors, edgecolor="black", alpha=0.85)
        axes[0, 0].set_yticks(range(len(top10)))
        axes[0, 0].set_yticklabels(top10["label"], fontsize=10)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("Composite Color Score (ΔE00 · √SSIM_in)", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Top 10 Temporal Schedules Ranked", fontsize=12, fontweight="bold")
        axes[0, 0].grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            axes[0, 0].text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.2f}", ha="left", va="center", fontsize=9, fontweight="bold")

        # Panel 2: Profile vs Gate Fraction
        profiles = df_clean["perfil_name"].unique()
        prof_colors = {"ascendente": "#2ecc71", "triangular": "#e67e22", "plano": "#3498db", "descendente": "#e74c3c"}
        for prof in profiles:
            sub = df_clean[df_clean["perfil_name"] == prof]
            grouped = sub.groupby("gate_frac")["avg_color_score"].mean().reset_index()
            c = prof_colors.get(prof, "#9b59b6")
            axes[0, 1].plot(grouped["gate_frac"], grouped["avg_color_score"], marker="o", linewidth=2.5, label=f"{prof.capitalize()}", color=c)
        axes[0, 1].set_xlabel("Gate Fraction (Early Guidance Stop)", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Mean Composite Color Score", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Color Score vs. Gate Fraction by Envelope Profile", fontsize=12, fontweight="bold")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)
        axes[0, 1].legend(loc="lower right", fontsize=10)

        # Panel 3: Pareto Frontier: ΔE00 vs Object Structure (SSIM_in)
        scatter = axes[1, 0].scatter(
            df_clean["avg_dE00"], df_clean["avg_ssim_in"],
            c=df_clean["avg_color_score"], cmap="viridis",
            s=df_clean["avg_psnr_out"] * 8, edgecolor="black", alpha=0.8
        )
        if winner is not None:
            axes[1, 0].scatter(
                [winner["avg_dE00"]], [winner["avg_ssim_in"]],
                color="red", s=250, marker="*", edgecolor="black", linewidth=1.5,
                label=f"Winner: {winner['perfil_name']} (N={int(winner['n_partes'])}, G={winner['gate_frac']:.2f})",
                zorder=10
            )
            axes[1, 0].legend(loc="lower left", fontsize=10)
        cbar = plt.colorbar(scatter, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar.set_label("Color Score", fontsize=10)
        axes[1, 0].set_xlabel("Mean Color Shift Magnitude (ΔE00)", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Inside Structure Preservation (SSIM_in)", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Pareto Trade-off: Color Shift vs. Object Structure", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Background Isolation Fidelity (PSNR_out vs SSIM_out)
        for prof in profiles:
            sub = df_clean[df_clean["perfil_name"] == prof]
            c = prof_colors.get(prof, "#9b59b6")
            axes[1, 1].scatter(sub["avg_ssim_out"], sub["avg_psnr_out"], label=prof.capitalize(), color=c, s=50, alpha=0.7)
        axes[1, 1].set_xlabel("Outside Background SSIM (SSIM_out)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Outside Background PSNR (dB)", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("Background Isolation Fidelity", fontsize=12, fontweight="bold")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        axes[1, 1].legend(loc="lower right", fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(out_dir, "fase_b_optimization_summary.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"[ANALYZER] Saved Phase B summary plot to: {plot_path}")
    except Exception as e:
        print(f"[WARNING] Could not generate Phase B plot: {e}")


def aggregate_and_select_winner(out_dir: str, model_type: str = "alpha"):
    csv_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("fase_b_worker_") and f.endswith(".csv")]
    if not csv_files:
        all_csv = os.path.join(out_dir, "fase_b_all_schedules.csv")
        if os.path.exists(all_csv):
            df = pd.read_csv(all_csv)
        else:
            print("[ERROR] No worker CSV files found in", out_dir)
            return
    else:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    df.sort_values(by="avg_color_score", ascending=False, inplace=True)

    # Winner selection (prefer 'ascendente' on ties / equal max score)
    max_score = df["avg_color_score"].max()
    top_candidates = df[df["avg_color_score"] >= max_score - 1e-4]
    asc_match = top_candidates[top_candidates["perfil_name"] == "ascendente"]
    if len(asc_match) > 0:
        winner = asc_match.iloc[0].to_dict()
    else:
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

    generate_phase_b_plots(df, out_dir, model_type=model_type, winner=winning_schedule)


def main():
    parser = argparse.ArgumentParser(description="Phase B: Temporal Envelope Search for PixArt")
    parser.add_argument("--model-type", choices=["alpha", "sigma"], default="alpha")
    parser.add_argument("--out-dir", default="./fase_b_pca_out")
    parser.add_argument("--axes-path", default=PCA_AXES_PATH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)

    # Worker arguments
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=999999)
    args = parser.parse_args()

    if args.axes_path is None or args.axes_path == PCA_AXES_PATH:
        default_axes = os.path.join(os.path.dirname(__file__), f"fase_a_{args.model_type}_out", "pca_axes.json")
        if os.path.exists(default_axes):
            args.axes_path = default_axes

    if args.out_dir == "./fase_b_pca_out":
        args.out_dir = f"./fase_b_{args.model_type}_out"

    os.makedirs(args.out_dir, exist_ok=True)

    if args.worker_id is not None:
        worker_main(args)
    else:
        n_schedules = len(GATE_FRACS) * len(N_PARTES_LIST) * len(PERFILES)
        extra_args = [
            "--model-type", args.model_type,
            "--out-dir", args.out_dir,
            "--axes-path", args.axes_path,
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
        ]
        spawn_workers(__file__, n_schedules, extra_args)
        aggregate_and_select_winner(args.out_dir, model_type=args.model_type)


if __name__ == "__main__":
    main()

"""
FASE_A_PCA.PY -- Phase A: Latent Sensitivity Screening & PCA Subspace Discovery for SD3.

Performs:
  1. Latent channel sensitivity screening across all 16 channels in SD3 VAE latent space.
  2. Perturbs each channel with positive/negative shifts delta in [-magnitude, +magnitude].
  3. Measures CIELAB delta vectors (dL, da, db) via SAM-3 segmentation and robust colorimetry.
  4. Computes Sensitivity Jacobian Matrix S in R^{16 x 3}.
  5. Performs SVD on S -> Orthonormal basis vectors U1, U2, U3.
  6. Computes Scree Plot (Explained Variance Ratio) and Perceptual Cosine Alignment with L*, a*, b*.
  7. Exports pca_axes.json, fase_a_raw.csv, and summary figures.
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

# Local imports
from sd3_core import (
    MODEL_ID, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION, STEPS, GUIDANCE,
    setup_sd3, decode_latents_4d, build_envelope_bands, build_mask_latent,
    latent_hw, run_generation, PERFIL_GENERATORS, spawn_workers, get_available_gpus
)
import utils
from iscc_nbs import get_iscc_l1_centroid

MAGNITUDES = [-0.6, -0.3, 0.3, 0.6]
GATE_FRAC = 0.5
PERFIL_NAME = "ascendente"
N_PARTES = 1

SCREENING_PROMPTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, product photo, no shadows", "sphere", 42),
    ("a clean white ceramic coffee mug on a neutral gray studio background, minimalist product photography", "mug", 101),
    ("a gray fabric armchair on a plain white studio background, commercial product shot", "armchair", 202),
    ("a simple gray sports shoe on a clean white background, studio lighting", "shoe", 303),
    ("a smooth gray metallic cylinder on a white background, studio lighting", "cylinder", 404),
    ("a gray backpack on a clean white background, minimalist product photo", "backpack", 505),
    ("a modern gray teapot on a white surface, studio product photo", "teapot", 606),
    ("a smooth gray ceramic vase on a plain white background, elegant studio shot", "vase", 707),
]


def run_single_screening(pipe, vae, seg_models, device,
                         prompt, obj_word, seed, channel_idx, mag,
                         bands, height=RESOLUTION, width=RESOLUTION,
                         steps=STEPS, guidance=GUIDANCE):
    """
    Runs baseline and perturbed generation for a single channel and magnitude.
    """
    # 1. Baseline generation
    latents_base = run_generation(
        pipe, prompt, seed, height, width, device, steps, guidance
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

    # 2. Perturbed generation
    latent_h, latent_w = latent_hw(height, width)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)

    latents_pert = run_generation(
        pipe, prompt, seed, height, width, device, steps, guidance,
        channel_idxs=[channel_idx], magnitude=mag, bands=bands, mask_latent=mask_latent
    )
    img_pert = decode_latents_4d(vae, latents_pert)
    lab_pert = utils.measure_color_gt(img_pert, mask_pixel)
    if lab_pert is None:
        lab_pert = lab_base

    dL = lab_pert[0] - lab_base[0]
    da = lab_pert[1] - lab_base[1]
    db = lab_pert[2] - lab_base[2]
    dE = utils.ciede2000(lab_base, lab_pert)

    return {
        "prompt": prompt,
        "object": obj_word,
        "seed": seed,
        "channel": channel_idx,
        "magnitude": mag,
        "base_L": lab_base[0], "base_a": lab_base[1], "base_b": lab_base[2],
        "pert_L": lab_pert[0], "pert_a": lab_pert[1], "pert_b": lab_pert[2],
        "dL": dL, "da": da, "db": db,
        "deltaE": dE,
    }


def worker_main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    bands = build_envelope_bands(args.gate_frac, PERFIL_GENERATORS[args.perfil_name](args.n_partes), "ramp_down")

    pipe, vae = setup_sd3(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)

    # Build all screening tasks
    tasks = []
    for p_idx, (prompt, obj_word, seed) in enumerate(SCREENING_PROMPTS):
        for ch in range(NUM_LATENT_CHANNELS):
            for mag in MAGNITUDES:
                tasks.append({
                    "prompt_idx": p_idx,
                    "prompt": prompt,
                    "object": obj_word,
                    "seed": seed,
                    "channel": ch,
                    "magnitude": mag,
                })

    start_idx = args.task_start
    end_idx = min(args.task_end, len(tasks))
    worker_tasks = tasks[start_idx:end_idx]
    print(f"[WORKER {args.worker_id}] Processing tasks {start_idx} to {end_idx} ({len(worker_tasks)} total) on {device}")

    results = []
    for i, t in enumerate(worker_tasks):
        res = run_single_screening(
            pipe, vae, seg_models, device,
            t["prompt"], t["object"], t["seed"],
            t["channel"], t["magnitude"], bands,
            steps=args.steps, guidance=args.guidance
        )
        results.append(res)
        if (i + 1) % 10 == 0 or (i + 1) == len(worker_tasks):
            print(f"[WORKER {args.worker_id}] Progress: {i+1}/{len(worker_tasks)} done.")

    out_csv = os.path.join(args.out_dir, f"fase_a_worker_{args.worker_id}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[WORKER {args.worker_id}] Saved results to: {out_csv}")


def aggregate_and_analyze(out_dir: str):
    """
    Merges worker outputs, computes SVD/PCA basis, scree plot, and cosine alignment.
    """
    csv_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("fase_a_worker_") and f.endswith(".csv")]
    if not csv_files:
        print("[ERROR] No worker CSV files found in", out_dir)
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    raw_path = os.path.join(out_dir, "fase_a_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"[ANALYZER] Aggregated {len(df)} trials into {raw_path}")

    # Build Sensitivity Jacobian Matrix S in R^{16 x 3}
    # Sensitivity s_{c} = (dL/dm, da/dm, db/dm)
    S = np.zeros((NUM_LATENT_CHANNELS, 3), dtype=np.float64)

    for ch in range(NUM_LATENT_CHANNELS):
        df_ch = df[df["channel"] == ch]
        mags = df_ch["magnitude"].values
        dLs = df_ch["dL"].values
        das = df_ch["da"].values
        dbs = df_ch["db"].values

        # Linear regression slope: delta_c / magnitude
        s_L = np.polyfit(mags, dLs, 1)[0] if len(mags) > 1 else 0.0
        s_a = np.polyfit(mags, das, 1)[0] if len(mags) > 1 else 0.0
        s_b = np.polyfit(mags, dbs, 1)[0] if len(mags) > 1 else 0.0
        S[ch] = [s_L, s_a, s_b]

    print("\n--- SENSITIVITY JACOBIAN MATRIX S (16x3) ---")
    print(np.round(S, 3))

    # SVD Decomposition: S = U Sigma V^T
    # S has shape (16, 3). Left singular vectors U in R^{16 x 3} span the latent subspace!
    U, Sigma, Vt = np.linalg.svd(S, full_matrices=False)

    # Explained Variance
    var_exp = (Sigma ** 2) / np.sum(Sigma ** 2)
    cum_var = np.cumsum(var_exp)

    print("\n--- SINGULAR VALUES & EXPLAINED VARIANCE ---")
    for i, (s, v, cv) in enumerate(zip(Sigma, var_exp, cum_var)):
        print(f"  PC{i+1}: Singular Value = {s:.4f} | Var = {v*100:.2f}% | CumVar = {cv*100:.2f}%")

    # Map Principal Vectors to Perceptual Axes
    # Project basis vectors back to CIELAB response: R_i = S^T U_i
    R = S.T @ U  # Shape (3, 3)

    # Cosine alignments with canonical L*, a*, b*
    canonical = {
        "L*": np.array([1.0, 0.0, 0.0]),
        "a*": np.array([0.0, 1.0, 0.0]),
        "b*": np.array([0.0, 0.0, 1.0]),
    }

    cos_sim = np.zeros((3, 3))
    for i in range(3):
        r_i = R[:, i]
        norm_r = np.linalg.norm(r_i) + 1e-8
        for j, (name, vec) in enumerate(canonical.items()):
            cos_sim[i, j] = np.dot(r_i, vec) / (norm_r * np.linalg.norm(vec))

    # Determine orientation (+/- sign) to match positive direction
    u1 = U[:, 0].copy()
    u2 = U[:, 1].copy()
    u3 = U[:, 2].copy()

    if cos_sim[0, 0] < 0:
        u1 *= -1; cos_sim[0] *= -1
    if cos_sim[1, 1] < 0:
        u2 *= -1; cos_sim[1] *= -1
    if cos_sim[2, 2] < 0:
        u3 *= -1; cos_sim[2] *= -1

    # Determine matched axes from cosine similarity
    lab_names = ["L*", "a*", "b*"]
    matched_lab = []
    for i in range(3):
        dom = int(np.argmax(np.abs(cos_sim[i])))
        matched_lab.append(lab_names[dom])

    pca_axes = {
        "model": MODEL_ID,
        "num_channels": NUM_LATENT_CHANNELS,
        "singular_values": Sigma.tolist(),
        "explained_variance_ratio": var_exp.tolist(),
        "axes": {
            "PC1": {"name": matched_lab[0], "loadings": u1.tolist(), "explained_var": float(var_exp[0])},
            "PC2": {"name": matched_lab[1], "loadings": u2.tolist(), "explained_var": float(var_exp[1])},
            "PC3": {"name": matched_lab[2], "loadings": u3.tolist(), "explained_var": float(var_exp[2])},
        },
        "cosine_similarity_matrix": cos_sim.tolist(),
    }

    json_path = os.path.join(out_dir, "pca_axes.json")
    with open(json_path, "w") as f:
        json.dump(pca_axes, f, indent=2)

    full_results = {
        "model": MODEL_ID,
        "pca_metrics": {
            "singular_values": Sigma.tolist(),
            "explained_variance_ratio": var_exp.tolist(),
            "cumulative_variance_ratio": cum_var.tolist(),
            "matched_lab_axes": matched_lab,
            "alignment_matrix_rows_PC_cols_Lab": cos_sim.tolist(),
            "principal_components": {
                f"PC1_{matched_lab[0]}": {"loading_vector_16d": u1.tolist(), "explained_var": float(var_exp[0])},
                f"PC2_{matched_lab[1]}": {"loading_vector_16d": u2.tolist(), "explained_var": float(var_exp[1])},
                f"PC3_{matched_lab[2]}": {"loading_vector_16d": u3.tolist(), "explained_var": float(var_exp[2])},
            }
        }
    }
    results_json_path = os.path.join(out_dir, "fase_a_pca_results.json")
    with open(results_json_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[ANALYZER] Saved PCA basis to: {json_path}")
    print(f"[ANALYZER] Saved full results to: {results_json_path}")

    # Plot Scree, Cosine Alignment Matrix, and 16D Loadings
    evr_pct = var_exp * 100
    cum_evr_pct = cum_var * 100
    components = [f"PC1\n({evr_pct[0]:.1f}%)", f"PC2\n({evr_pct[1]:.1f}%)", f"PC3\n({evr_pct[2]:.1f}%)"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Plot 1: Scree Plot
    x = np.arange(len(evr_pct))
    bars = axes[0].bar(x, evr_pct, color="#2b5c8f", width=0.45, label="Individual EVR (%)")
    axes[0].plot(x, cum_evr_pct, color="#d95f02", marker="o", linewidth=2.5, markersize=8, label="Cumulative (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(components, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Explained Variance (%)", fontsize=11)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("1. Explained Variance per Principal Component", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="center right", fontsize=10)
    for bar, val in zip(bars, evr_pct):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Plot 2: Cosine Alignment Matrix
    im = axes[1].imshow(cos_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"], fontsize=10, fontweight="bold")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["PC1", "PC2", "PC3"], fontsize=11, fontweight="bold")
    axes[1].set_title("2. Directional Alignment (Cosine Similarity)", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("cos(θ) alignment", fontsize=10)

    for i in range(3):
        for j in range(3):
            val = cos_sim[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            axes[1].text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    # Plot 3: 16D Latent Loadings
    channels = np.arange(NUM_LATENT_CHANNELS)
    width = 0.27
    axes[2].bar(channels - width, u1, width=width, label=f"PC1 ({evr_pct[0]:.1f}%)", color="#1b9e77")
    axes[2].bar(channels, u2, width=width, label=f"PC2 ({evr_pct[1]:.1f}%)", color="#d95f02")
    axes[2].bar(channels + width, u3, width=width, label=f"PC3 ({evr_pct[2]:.1f}%)", color="#7570b3")

    axes[2].set_xticks(channels)
    axes[2].set_xlabel("SD3 Latent Channel Index (0 – 15)", fontsize=11)
    axes[2].set_ylabel("Linear Loading Weight (u_k,c)", fontsize=11)
    axes[2].set_title("3. 16D Latent Loadings (Linear Combinations)", fontsize=12, fontweight="bold")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    summary_plot_path = os.path.join(out_dir, "fase_a_pca_summary.png")
    fig.savefig(summary_plot_path, dpi=300)
    alignment_plot_path = os.path.join(out_dir, "fase_a_pca_alignment_plot.png")
    fig.savefig(alignment_plot_path, dpi=300)
    plt.close(fig)
    print(f"[ANALYZER] Saved summary figures to:\n - {summary_plot_path}\n - {alignment_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase A: Latent Sensitivity Screening for SD3")
    parser.add_argument("--out-dir", default="./fase_a_pca_out")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--gate-frac", type=float, default=GATE_FRAC)
    parser.add_argument("--perfil-name", default=PERFIL_NAME)
    parser.add_argument("--n-partes", type=int, default=N_PARTES)

    # Worker arguments
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=999999)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.worker_id is not None:
        worker_main(args)
    else:
        # Master orchestrator
        n_tasks = len(SCREENING_PROMPTS) * NUM_LATENT_CHANNELS * len(MAGNITUDES)
        extra_args = [
            "--out-dir", args.out_dir,
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
            "--gate-frac", str(args.gate_frac),
            "--perfil-name", args.perfil_name,
            "--n-partes", str(args.n_partes),
        ]
        spawn_workers(__file__, n_tasks, extra_args)
        aggregate_and_analyze(args.out_dir)


if __name__ == "__main__":
    main()

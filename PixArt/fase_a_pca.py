"""
FASE_A_PCA.PY -- Phase A: Latent Sensitivity Screening & PCA Subspace Discovery for PixArt.

Performs:
  1. Latent channel sensitivity screening across all 4 channels in PixArt VAE latent space.
  2. Perturbs each channel with positive/negative shifts delta in [-magnitude, +magnitude].
  3. Measures CIELAB delta vectors (dL, da, db) via SAM-3 segmentation and robust colorimetry.
  4. Computes Sensitivity Jacobian Matrix S in R^{4 x 3}.
  5. Performs SVD on S -> Orthonormal basis vectors U1, U2, U3 in R^4.
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
from pixart_core import (
    MODEL_ID_ALPHA, MODEL_ID_SIGMA, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION,
    STEPS, GUIDANCE, MAX_SEQ_LEN_ALPHA, MAX_SEQ_LEN_SIGMA,
    setup_pixart, decode_latents_4d, build_envelope_bands, build_mask_latent,
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
                         steps=STEPS, guidance=GUIDANCE, max_seq_len=MAX_SEQ_LEN_ALPHA):
    """
    Runs baseline and perturbed generation for a single channel and magnitude.
    """
    # 1. Baseline generation
    latents_base = run_generation(
        pipe, prompt, seed, height, width, device, steps, guidance,
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

    # 2. Perturbed generation
    latent_h, latent_w = latent_hw(height, width)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)

    latents_pert = run_generation(
        pipe, prompt, seed, height, width, device, steps, guidance,
        max_sequence_length=max_seq_len,
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
    max_seq_len = MAX_SEQ_LEN_SIGMA if args.model_type == "sigma" else MAX_SEQ_LEN_ALPHA

    pipe, vae = setup_pixart(model_type=args.model_type, device=device, dtype=DTYPE, num_latent_channels=NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models("cpu")

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
    out_csv = os.path.join(args.out_dir, f"fase_a_worker_{args.worker_id}.csv")
    print(f"[WORKER {args.worker_id}] Processing tasks {start_idx} to {end_idx} ({len(worker_tasks)} total) on {device}")

    results = []
    for i, t in enumerate(worker_tasks):
        res = run_single_screening(
            pipe, vae, seg_models, device,
            t["prompt"], t["object"], t["seed"],
            t["channel"], t["magnitude"], bands,
            steps=args.steps, guidance=args.guidance, max_seq_len=max_seq_len
        )
        results.append(res)
        print(f"[WORKER {args.worker_id}] Task {i+1}/{len(worker_tasks)} (ch={t['channel']}, mag={t['magnitude']}): "
              f"dL={res['dL']:.2f}, da={res['da']:.2f}, db={res['db']:.2f}, dE={res['deltaE']:.2f}")
        pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"[WORKER {args.worker_id}] Finished all tasks. Results saved to: {out_csv}")


def aggregate_and_analyze(out_dir: str, model_type: str = "alpha"):
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

    # Build Sensitivity Jacobian Matrix S in R^{4 x 3}
    S = np.zeros((NUM_LATENT_CHANNELS, 3), dtype=np.float64)

    for ch in range(NUM_LATENT_CHANNELS):
        df_ch = df[df["channel"] == ch]
        mags = df_ch["magnitude"].values
        dLs = df_ch["dL"].values
        das = df_ch["da"].values
        dbs = df_ch["db"].values

        s_L = np.polyfit(mags, dLs, 1)[0] if len(mags) > 1 else 0.0
        s_a = np.polyfit(mags, das, 1)[0] if len(mags) > 1 else 0.0
        s_b = np.polyfit(mags, dbs, 1)[0] if len(mags) > 1 else 0.0
        S[ch] = [s_L, s_a, s_b]

    print("\n--- SENSITIVITY JACOBIAN MATRIX S (4x3) ---")
    print(np.round(S, 3))

    # SVD Decomposition: S = U Sigma V^T
    # S has shape (4, 3). Left singular vectors U in R^{4 x 3} span the latent subspace!
    U, Sigma, Vt = np.linalg.svd(S, full_matrices=False)

    # Explained Variance
    var_exp = (Sigma ** 2) / np.sum(Sigma ** 2)
    cum_var = np.cumsum(var_exp)

    print("\n--- SINGULAR VALUES & EXPLAINED VARIANCE ---")
    for i, (s, v, cv) in enumerate(zip(Sigma, var_exp, cum_var)):
        print(f"  PC{i+1}: Singular Value = {s:.4f} | Var = {v*100:.2f}% | CumVar = {cv*100:.2f}%")

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

    pca_axes = {
        "model": MODEL_ID_ALPHA if model_type == "alpha" else MODEL_ID_SIGMA,
        "model_type": model_type,
        "num_channels": NUM_LATENT_CHANNELS,
        "singular_values": Sigma.tolist(),
        "explained_variance_ratio": var_exp.tolist(),
        "axes": {
            "PC1": {"name": "L*", "loadings": u1.tolist(), "explained_var": float(var_exp[0])},
            "PC2": {"name": "a*", "loadings": u2.tolist(), "explained_var": float(var_exp[1])},
            "PC3": {"name": "b*", "loadings": u3.tolist(), "explained_var": float(var_exp[2])},
        },
        "cosine_similarity_matrix": cos_sim.tolist(),
    }

    json_path = os.path.join(out_dir, "pca_axes.json")
    with open(json_path, "w") as f:
        json.dump(pca_axes, f, indent=2)
    print(f"\n[ANALYZER] Saved PCA basis to: {json_path}")

    # 3-Panel Diagnostic Figure (FLUX style)
    evr = var_exp * 100
    cum_evr = cum_var * 100
    components = [f"PC1\n({evr[0]:.1f}%)", f"PC2\n({evr[1]:.1f}%)", f"PC3\n({evr[2]:.1f}%)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Scree Plot
    x = np.arange(len(evr))
    bars = axes[0].bar(x, evr, color="#3470a3", width=0.5, label="Individual EVR (%)")
    line = axes[0].plot(x, cum_evr, color="#d95f02", marker="o", linewidth=2, label="Cumulative (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(components, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Explained Variance (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].set_title(f"Scree Plot: Explained Variance per PC (PixArt-{model_type.upper()})", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=10)
    for bar, val in zip(bars, evr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Alignment Heatmap
    im = axes[1].imshow(cos_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"], fontsize=10)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["PC1", "PC2", "PC3"], fontsize=11, fontweight="bold")
    axes[1].set_title("Cosine Alignment: PC Directions vs CIELAB Axes", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("cos(θ)", fontsize=10)

    for i in range(3):
        for j in range(3):
            val = cos_sim[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            axes[1].text(j, i, f"{val:+.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    # Panel 3: 4D Latent Loadings
    channels = np.arange(NUM_LATENT_CHANNELS)
    width = 0.26
    c_u1 = u1
    c_u2 = u2
    c_u3 = u3

    axes[2].bar(channels - width, c_u1, width=width, label=f"PC1 ({evr[0]:.1f}%)", color="#1b9e77")
    axes[2].bar(channels, c_u2, width=width, label=f"PC2 ({evr[1]:.1f}%)", color="#d95f02")
    axes[2].bar(channels + width, c_u3, width=width, label=f"PC3 ({evr[2]:.1f}%)", color="#7570b3")

    axes[2].set_xticks(channels)
    axes[2].set_xlabel("Latent Channel Index (0 - 3)", fontsize=11)
    axes[2].set_ylabel("Loading Weight (u_k,c)", fontsize=11)
    axes[2].set_title("4D Latent Loadings (Subspace Basis)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "fase_a_pca_summary.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[ANALYZER] Saved FLUX-style summary figure to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase A: Latent Sensitivity Screening for PixArt")
    parser.add_argument("--model-type", choices=["alpha", "sigma"], default="alpha")
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
            "--model-type", args.model_type,
            "--out-dir", args.out_dir,
            "--steps", str(args.steps),
            "--guidance", str(args.guidance),
            "--gate-frac", str(args.gate_frac),
            "--perfil-name", args.perfil_name,
            "--n-partes", str(args.n_partes),
        ]
        spawn_workers(__file__, n_tasks, extra_args)
        aggregate_and_analyze(args.out_dir, model_type=args.model_type)


if __name__ == "__main__":
    main()

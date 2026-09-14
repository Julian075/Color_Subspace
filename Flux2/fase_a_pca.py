"""
FASE_A_PCA.PY -- Characterization of FLUX latent color space via PCA.

Phase 1:
1. Sweeps through all 16 latent channels across reference objects and magnitudes
   to measure linear slopes (deltaL, deltaA, deltaB per unit magnitude).
   Uses REF_MODE = "none" (absolute displacement in latent units).
2. Performs SVD / PCA on the channel effect matrix S (16 channels x 3 Lab deltas)
   to determine:
   - Explained variance per principal component (scree analysis).
   - Quantitative cosine alignment between each PC response and the canonical Lab axes (L, a, b).
   - Channel loading vectors for each principal component.
3. Exports comprehensive JSON statistics, alignment tables, and visualization plots.
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

# Add current directory to sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from flux2_core import (
    build_envelope_bands, setup_flux, latent_hw, unpack_to_4d, decode_latents_4d,
    build_mask_latent, save_mask_preview, run_generation, get_gpu_list, spawn_workers,
)

try:
    from utils import setup_seg_models, get_object_mask, measure_color_gt
    _UTILS_IMPORT_ERROR = None
except ImportError as _e:
    setup_seg_models = get_object_mask = measure_color_gt = None
    _UTILS_IMPORT_ERROR = str(_e)

try:
    from skimage.metrics import structural_similarity
    _SKIMAGE_IMPORT_ERROR = None
except ImportError as _e:
    structural_similarity = None
    _SKIMAGE_IMPORT_ERROR = str(_e)


def _require_all_deps():
    if setup_seg_models is None or measure_color_gt is None:
        raise ImportError(
            "Could not import setup_seg_models/get_object_mask/measure_color_gt from utils.py.\n"
            f"  Original error: {_UTILS_IMPORT_ERROR}\n  cwd={os.getcwd()}\n"
            f"  sys.path[0]={sys.path[0] if sys.path else '(empty)'}")
    if structural_similarity is None:
        raise ImportError(
            "Could not import structural_similarity from skimage.\n"
            f"  Original error: {_SKIMAGE_IMPORT_ERROR}")


# =========================== CONFIG ===========================
MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5
NUM_LATENT_CHANNELS = 32

# Fixed exploration schedule
GATE_FRAC = 0.7
ENVELOPE_CHRONO = [0.2, 0.6, 1.0]
SCHED_MODE = "ramp_down"

# In the PCA version, we use absolute shift without scaling by arbitrary reference channels
REF_MODE = "none"
REF_CHANNELS = None

CHANNELS_TO_TEST = list(range(NUM_LATENT_CHANNELS))
MAGNITUDES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

# 7 canonical reference objects covering all Lab gamut poles (+L, -L, +a, -a, +b, -b, neutral)
REFERENCE_OBJECTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, minimalist, no shadows, no texture", 0, "sphere"),
    ("a red toy car on a plain white background, studio lighting, product photo, no shadows", 1, "car"),
    ("a blue ceramic mug on a plain white background, studio lighting, product photo, no shadows", 2, "mug"),
    ("a bright yellow rubber duck on a plain white background, studio lighting, product photo, no shadows", 3, "duck"),
    ("a green apple on a plain white background, studio lighting, product photo, no shadows", 4, "apple"),
    ("a white ceramic vase on a plain light gray background, studio lighting, product photo, no shadows", 5, "vase"),
    ("a black leather wallet on a plain white background, studio lighting, product photo, no shadows", 6, "wallet"),
]

OUT_DIR = "./fase_a_pca_out"

PARALLEL = False
AVAILABLE_GPUS = None
WORKERS_PER_GPU = 1

MIN_EFFECT_NORM_FRAC = 0.15
MIN_SSIM_OK = 0.75

CSV_FIELDS = ["case_idx", "obj_word", "seed", "channel", "magnitude",
              "deltaL", "deltaA", "deltaB", "ssim", "note"]


# =========================== CORE GENERATION & METRICS ===========================
def gen(pipe, prompt, seed, device, channel_idxs=None, magnitude=None, bands=None, mask_latent=None):
    return run_generation(
        pipe, prompt, seed, RESOLUTION, RESOLUTION, device, STEPS, GUIDANCE,
        channel_idxs=channel_idxs, magnitude=magnitude, bands=bands,
        mask_latent=mask_latent, ref_mode=REF_MODE, ref_channels=REF_CHANNELS
    )


def generate_case_context(pipe, vae, seg_models, prompt, seed, obj_word, device, out_dir, case_tag):
    print(f"\n=== Case [{obj_word}] seed={seed} ===")
    latents_packed_base = gen(pipe, prompt, seed, device)
    if torch.isnan(latents_packed_base).any():
        print("  [ERROR] NaN in baseline latents -- case discarded")
        return None
    img_base = decode_latents_4d(vae, unpack_to_4d(pipe, latents_packed_base, RESOLUTION, RESOLUTION))
    Image.fromarray(img_base).save(os.path.join(out_dir, f"{case_tag}_baseline.png"))

    mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        print(f"  [WARNING] SAM3 did not detect '{obj_word}' -- case discarded")
        return None
    base_lab = measure_color_gt(img_base, mask_pixel)
    if base_lab is None:
        print("  [WARNING] measure_color_gt returned None -- case discarded")
        return None

    latent_h, latent_w = latent_hw(pipe, RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)
    save_mask_preview(img_base, mask_pixel, os.path.join(out_dir, f"{case_tag}_mask_preview.png"))
    bands = build_envelope_bands(GATE_FRAC, ENVELOPE_CHRONO, SCHED_MODE)
    print(f"  Mask OK, base_lab=({base_lab[0]:.1f}, {base_lab[1]:.1f}, {base_lab[2]:.1f})  "
          f"{int(mask_pixel.sum())} px ({100 * mask_pixel.mean():.1f}%)")

    return {
        "img_base": img_base,
        "mask_pixel": mask_pixel,
        "mask_latent": mask_latent,
        "base_lab": base_lab,
        "bands": bands
    }


def compute_cell_metrics(pipe, vae, prompt, seed, device, channel_idx, magnitude, ctx):
    latents_packed = gen(
        pipe, prompt, seed, device, channel_idxs=[channel_idx], magnitude=magnitude,
        bands=ctx["bands"], mask_latent=ctx["mask_latent"]
    )
    if torch.isnan(latents_packed).any():
        return {"deltaL": "", "deltaA": "", "deltaB": "", "ssim": "", "note": "nan_latent"}
    latents_4d = unpack_to_4d(pipe, latents_packed, RESOLUTION, RESOLUTION)
    img = decode_latents_4d(vae, latents_4d)
    ssim_val = structural_similarity(ctx["img_base"], img, channel_axis=2, data_range=255)
    lab = measure_color_gt(img, ctx["mask_pixel"])
    if lab is None:
        return {"deltaL": "", "deltaA": "", "deltaB": "", "ssim": f"{ssim_val:.4f}", "note": "mask_lost"}
    dL = lab[0] - ctx["base_lab"][0]
    dA = lab[1] - ctx["base_lab"][1]
    dB = lab[2] - ctx["base_lab"][2]
    return {
        "deltaL": f"{dL:.3f}",
        "deltaA": f"{dA:.3f}",
        "deltaB": f"{dB:.3f}",
        "ssim": f"{ssim_val:.4f}",
        "note": "ok"
    }


# =========================== SEQUENTIAL & PARALLEL EXECUTION ===========================
def run_case(pipe, vae, seg_models, case_idx, prompt, seed, obj_word, out_dir, device,
             channels_to_test, csv_writer, csv_file):
    case_tag = f"seed{seed}_{obj_word}"
    ctx = generate_case_context(pipe, vae, seg_models, prompt, seed, obj_word, device, out_dir, case_tag)
    if ctx is None:
        return
    for c in channels_to_test:
        for mag in MAGNITUDES:
            m = compute_cell_metrics(pipe, vae, prompt, seed, device, c, mag, ctx)
            csv_writer.writerow({
                "case_idx": case_idx, "obj_word": obj_word, "seed": seed,
                "channel": c, "magnitude": mag, **m
            })
            csv_file.flush()
            extra = f" dL={m['deltaL']} dA={m['deltaA']} dB={m['deltaB']} ssim={m['ssim']}" if m["note"] == "ok" else ""
            print(f"  ch{c} mag={mag:+.2f} -> {m['note']}{extra}")


def build_all_tasks(channels_to_test):
    return [
        {"case_idx": ci, "channel": c, "mag": m}
        for ci in range(len(REFERENCE_OBJECTS))
        for c in channels_to_test
        for m in MAGNITUDES
    ]


def run_worker(chunk_path, out_dir, pipe, vae, seg_models):
    pid = os.getpid()
    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(chunk_path) as f:
        tasks = json.load(f)
    print(f"[worker pid={pid} GPU_phys={gpu_env}] {len(tasks)} cells assigned")

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"part_{pid}.csv")
    ctx_cache = {}
    with open(part_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in tasks:
            case_idx = t["case_idx"]
            prompt, seed, obj_word = REFERENCE_OBJECTS[case_idx]
            case_tag = f"seed{seed}_{obj_word}"
            if case_idx not in ctx_cache:
                print(f"[worker pid={pid}] Generating baseline case {case_idx} ({obj_word})...")
                ctx_cache[case_idx] = generate_case_context(
                    pipe, vae, seg_models, prompt, seed, obj_word, DEVICE, out_dir, case_tag)
            ctx = ctx_cache[case_idx]
            if ctx is None:
                continue
            c, mag = t["channel"], t["mag"]
            m = compute_cell_metrics(pipe, vae, prompt, seed, DEVICE, c, mag, ctx)
            writer.writerow({
                "case_idx": case_idx, "obj_word": obj_word, "seed": seed,
                "channel": c, "magnitude": mag, **m
            })
            f.flush()
            print(f"[worker pid={pid}] ch{c} mag={mag:+.2f} case={obj_word} -> {m['note']}")
    print(f"[worker pid={pid} GPU_phys={gpu_env}] Finished")


def get_completed_task_keys(out_dir):
    parts_dir = os.path.join(out_dir, "_csv_parts")
    completed = set()
    if os.path.isdir(parts_dir):
        for pf in os.listdir(parts_dir):
            if pf.endswith(".csv"):
                p_path = os.path.join(parts_dir, pf)
                try:
                    with open(p_path, newline="") as f:
                        for row in csv.DictReader(f):
                            if "case_idx" in row and "channel" in row and "magnitude" in row:
                                try:
                                    c_idx = int(row["case_idx"])
                                    ch = int(row["channel"])
                                    mag = float(row["magnitude"])
                                    completed.add((c_idx, ch, round(mag, 4)))
                                except Exception:
                                    pass
                except Exception:
                    pass
    return completed


def run_parallel(out_dir, channels_to_test, workers_per_gpu=WORKERS_PER_GPU):
    _require_all_deps()
    gpu_list = get_gpu_list(AVAILABLE_GPUS)
    n_workers = len(gpu_list) * workers_per_gpu
    all_tasks = build_all_tasks(channels_to_test)
    completed_keys = get_completed_task_keys(out_dir)
    tasks = [t for t in all_tasks if (t["case_idx"], t["channel"], round(float(t["mag"]), 4)) not in completed_keys]

    print(f"\n>>> RESUME CHECK: Found {len(completed_keys)} already completed cells.")
    print(f">>> PARALLEL: {len(tasks)} remaining tasks ({len(all_tasks)} total) -> {len(gpu_list)} GPU(s) x {workers_per_gpu} worker/GPU = {n_workers} workers total")

    if len(tasks) > 0:
        cmd_base = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--out-dir", out_dir, "--resolution", str(RESOLUTION), "--steps", str(STEPS),
            "--guidance", str(GUIDANCE), "--gate-frac", str(GATE_FRAC)
        ]
        exit_codes = spawn_workers(
            tasks, n_workers, gpu_list, workers_per_gpu,
            os.path.join(out_dir, "_tmp_chunks"), cmd_base
        )
        n_failed = sum(1 for c in exit_codes if c != 0)
        if n_failed:
            print(f"  [WARNING] {n_failed}/{len(exit_codes)} worker(s) returned error exit codes")

    merge_csv_parts(out_dir)


def merge_csv_parts(out_dir):
    parts_dir = os.path.join(out_dir, "_csv_parts")
    out_path = os.path.join(out_dir, "fase_a_raw.csv")
    part_files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".csv")) if os.path.isdir(parts_dir) else []
    seen = set()
    rows = []
    for pf in part_files:
        p_path = os.path.join(parts_dir, pf)
        try:
            with open(p_path, newline="") as fin:
                for row in csv.DictReader(fin):
                    if "case_idx" in row and "channel" in row and "magnitude" in row:
                        try:
                            key = (int(row["case_idx"]), int(row["channel"]), round(float(row["magnitude"]), 4))
                            if key not in seen:
                                seen.add(key)
                                rows.append(row)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning reading {pf}: {e}")
    with open(out_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n>>> Merged {len(rows)} unique rows from {len(part_files)} worker part(s) -> {out_path}")


# =========================== PCA & LAB ALIGNMENT ANALYSIS ===========================
def analyze_pca(csv_path, out_dir, channels_to_test):
    """
    Performs Principal Component Analysis (PCA / SVD) on the channel sensitivity matrix.
    Computes:
      1. Slope matrix S (N_channels x 3)
      2. SVD: S = U @ diag(sigma) @ V.T
      3. Explained variance per PC (scree analysis)
      4. Cosine similarity alignment between PC responses and canonical Lab axes (L, a, b)
      5. Canonicalized loading vectors in R^16
      6. Diagnostic plots (Scree plot, Lab Alignment Heatmap, Channel Loadings Bar Chart)
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_channels = len(channels_to_test)
    by_channel = {c: {"mag": [], "dL": [], "dA": [], "dB": [], "ssim": []} for c in channels_to_test}
    n_mask_lost = {c: 0 for c in channels_to_test}
    n_nan = {c: 0 for c in channels_to_test}

    for r in rows:
        c = int(r["channel"])
        if c not in by_channel:
            continue
        if r["note"] == "ok":
            by_channel[c]["mag"].append(float(r["magnitude"]))
            by_channel[c]["dL"].append(float(r["deltaL"]))
            by_channel[c]["dA"].append(float(r["deltaA"]))
            by_channel[c]["dB"].append(float(r["deltaB"]))
            by_channel[c]["ssim"].append(float(r["ssim"]))
        elif r["note"] == "mask_lost":
            n_mask_lost[c] += 1
            if r["ssim"]:
                by_channel[c]["ssim"].append(float(r["ssim"]))
        elif r["note"] == "nan_latent":
            n_nan[c] += 1

    # Linear regression slopes (dL/m, dA/m, dB/m)
    S = np.zeros((n_channels, 3), dtype=np.float64)  # (16, 3)
    ssim_worst = np.full(n_channels, np.nan)

    for i, c in enumerate(channels_to_test):
        d = by_channel[c]
        if len(d["mag"]) >= 2:
            mags = np.array(d["mag"])
            for j, key in enumerate(("dL", "dA", "dB")):
                slope, _ = np.polyfit(mags, np.array(d[key]), 1)
                S[i, j] = slope
        if d["ssim"]:
            ssim_worst[i] = min(d["ssim"])

    # Channel response norms and filtering diagnostics
    channel_norms = np.linalg.norm(S, axis=1)
    max_norm = channel_norms.max() if channel_norms.max() > 0 else 1.0
    negligible = [channels_to_test[i] for i in range(n_channels) if channel_norms[i] < MIN_EFFECT_NORM_FRAC * max_norm]
    distorting = [channels_to_test[i] for i in range(n_channels) if not np.isnan(ssim_worst[i]) and ssim_worst[i] < MIN_SSIM_OK]

    # SVD decomposition: S = U @ S_diag @ V.T
    # S is (16, 3).
    # U: (16, 3) contains orthogonal column vectors in latent channel space (loadings u_k in R^16)
    # sigma: (3,) singular values
    # V: (3, 3) contains right singular vectors in Lab space
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt.T  # (3, 3)

    eigenvalues = sigma ** 2
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = (eigenvalues / (total_variance + 1e-12)) if total_variance > 0 else np.zeros_like(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Calculate Lab responses for each principal component: r_k = S.T @ u_k
    # r_k = sigma_k * v_k
    # Canonicalize signs: ensure the dominant Lab component is positive (+L, +a, or +b)
    canonical_U = U.copy()
    canonical_V = V.copy()
    canonical_responses = []

    lab_axis_names = ["L", "a", "b"]
    canonical_axes_dict = {}

    for k in range(3):
        v_k = canonical_V[:, k]  # (3,)
        u_k = canonical_U[:, k]  # (16,)
        dominant_idx = int(np.argmax(np.abs(v_k)))
        if v_k[dominant_idx] < 0:
            u_k = -u_k
            v_k = -v_k
            canonical_U[:, k] = u_k
            canonical_V[:, k] = v_k

        resp = sigma[k] * v_k
        canonical_responses.append(resp)

    # Alignment Matrix: cosine similarity with canonical Lab unit vectors e_L=[1,0,0], e_a=[0,1,0], e_b=[0,0,1]
    # cos(theta_{k, j}) = v_{k, j}
    alignment_matrix = canonical_V.T  # (3 components x 3 Lab axes)

    # Associate each PC with its primary matching Lab axis
    matched_lab_axes = []
    for k in range(3):
        dom_idx = int(np.argmax(np.abs(alignment_matrix[k])))
        matched_lab_axes.append(lab_axis_names[dom_idx])
        canonical_axes_dict[f"PC{k+1}_{lab_axis_names[dom_idx]}"] = {
            "component_idx": k,
            "primary_axis": lab_axis_names[dom_idx],
            "singular_value": float(sigma[k]),
            "explained_variance_ratio": float(explained_variance_ratio[k]),
            "loading_vector_16d": canonical_U[:, k].tolist(),
            "lab_response_vector": canonical_responses[k].tolist(),
            "cosine_alignment": {
                "L": float(alignment_matrix[k, 0]),
                "a": float(alignment_matrix[k, 1]),
                "b": float(alignment_matrix[k, 2])
            }
        }

    # Save complete PCA analysis JSON
    analysis_results = {
        "channels": channels_to_test,
        "sensitivity_matrix_S": S.tolist(),
        "channel_norms": channel_norms.tolist(),
        "ssim_worst_case": [None if np.isnan(v) else v for v in ssim_worst.tolist()],
        "n_mask_lost": n_mask_lost,
        "n_nan": n_nan,
        "negligible_channels": negligible,
        "distorting_channels": distorting,
        "pca_metrics": {
            "singular_values": sigma.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "total_variance": float(total_variance),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
            "matched_lab_axes": matched_lab_axes,
            "alignment_matrix_rows_PC_cols_Lab": alignment_matrix.tolist(),
            "principal_components": canonical_axes_dict
        }
    }

    results_json_path = os.path.join(out_dir, "fase_a_pca_results.json")
    with open(results_json_path, "w") as f:
        json.dump(analysis_results, f, indent=2)

    # Save simplified axes definition for downstream pipeline phases
    axes_export = {
        "method": "PCA",
        "num_latent_channels": NUM_LATENT_CHANNELS,
        "num_components": 3,
        "ref_mode": REF_MODE,
        "axes": {
            f"PC{k+1}": {
                "label": f"PC{k+1}_{matched_lab_axes[k]}",
                "primary_lab": matched_lab_axes[k],
                "evr": float(explained_variance_ratio[k]),
                "loadings": canonical_U[:, k].tolist()
            }
            for k in range(3)
        }
    }
    axes_json_path = os.path.join(out_dir, "pca_axes.json")
    with open(axes_json_path, "w") as f:
        json.dump(axes_export, f, indent=2)

    # =========================== PRINT CONSOLE SUMMARY ===========================
    print("\n" + "=" * 80)
    print("                      PCA COLOR-SPACE CHARACTERIZATION RESULTS                 ")
    print("=" * 80)

    print("\n1. EXPLAINED VARIANCE RATIO (SCREE ANALYSIS):")
    for k in range(3):
        print(f"   PC{k+1} ({matched_lab_axes[k]}): "
              f"Singular Value = {sigma[k]:6.3f} | "
              f"Explained Variance = {explained_variance_ratio[k]*100:5.2f}% | "
              f"Cumulative = {cumulative_variance_ratio[k]*100:5.2f}%")
    print(f"   Top 3 components explain {cumulative_variance_ratio[2]*100:5.2f}% of total color variation.")

    print("\n2. QUANTITATIVE ALIGNMENT WITH CANONICAL Lab AXES (COSINE SIMILARITY):")
    print("   " + "-" * 60)
    print(f"   {'Component':<12} | {'cos(theta, L)':<14} | {'cos(theta, a)':<14} | {'cos(theta, b)':<14} | Matched")
    print("   " + "-" * 60)
    for k in range(3):
        print(f"   PC{k+1:<10} | {alignment_matrix[k, 0]:+14.4f} | {alignment_matrix[k, 1]:+14.4f} | {alignment_matrix[k, 2]:+14.4f} | -> {matched_lab_axes[k]}")
    print("   " + "-" * 60)

    print("\n3. PRINCIPAL COMPONENT LOADINGS IN 16D LATENT SPACE:")
    for k in range(3):
        loadings_k = canonical_U[:, k]
        top_pos = np.where(loadings_k > 0.15)[0]
        top_neg = np.where(loadings_k < -0.15)[0]
        pos_str = ", ".join(f"ch{c}(+{loadings_k[c]:.2f})" for c in top_pos)
        neg_str = ", ".join(f"ch{c}({loadings_k[c]:.2f})" for c in top_neg)
        print(f"   PC{k+1} ({matched_lab_axes[k]}):")
        print(f"      Positive drivers: {pos_str if pos_str else 'none'}")
        print(f"      Negative drivers: {neg_str if neg_str else 'none'}")

    if negligible:
        print(f"\n   Negligible effect channels (<{MIN_EFFECT_NORM_FRAC*100:.0f}% max): {negligible}")
    if distorting:
        print(f"   Distorting channels (SSIM < {MIN_SSIM_OK}): {distorting}")

    print(f"\nSaved analysis JSON: {results_json_path}")
    print(f"Saved PCA axes JSON:   {axes_json_path}")

    # Generate visual plots
    plot_pca_diagnostics(analysis_results, out_dir)


def plot_pca_diagnostics(results, out_dir):
    """
    Generates 3 diagnostic figures:
      1. Scree plot (Explained Variance & Cumulative Variance)
      2. Alignment Heatmap (Cosine similarity of PC vs Lab axes)
      3. 16D Channel Loadings Bar Chart per PC (Linear combination of all 16 channels)
    """
    pca_metrics = results["pca_metrics"]
    evr = np.array(pca_metrics["explained_variance_ratio"]) * 100
    cum_evr = np.array(pca_metrics["cumulative_variance_ratio"]) * 100
    alignment = np.array(pca_metrics["alignment_matrix_rows_PC_cols_Lab"])
    components = [f"PC1\n({evr[0]:.1f}%)", f"PC2\n({evr[1]:.1f}%)", f"PC3\n({evr[2]:.1f}%)"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Plot 1: Scree Plot
    x = np.arange(len(evr))
    bars = axes[0].bar(x, evr, color="#3470a3", width=0.5, label="Individual EVR (%)")
    line = axes[0].plot(x, cum_evr, color="#d95f02", marker="o", linewidth=2, label="Cumulative (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(components, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Explained Variance (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Scree Plot: Explained Variance per PC", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")
    for bar, val in zip(bars, evr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Plot 2: Lab Alignment Heatmap
    im = axes[1].imshow(alignment, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["L* (Luminance)", "a* (Green-Red)", "b* (Blue-Yellow)"], fontsize=10)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["PC1", "PC2", "PC3"], fontsize=11, fontweight="bold")
    axes[1].set_title("Cosine Alignment: PC Directions vs CIELAB Axes", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(3):
        for j in range(3):
            val = alignment[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            axes[1].text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)

    # Plot 3: 16D Channel Loadings (Linear combination weights of all 16 channels)
    channels = np.arange(NUM_LATENT_CHANNELS)
    width = 0.26
    pcs_dict = pca_metrics["principal_components"]
    pc_keys = list(pcs_dict.keys())
    c_u1 = np.array(pcs_dict[pc_keys[0]]["loading_vector_16d"])
    c_u2 = np.array(pcs_dict[pc_keys[1]]["loading_vector_16d"])
    c_u3 = np.array(pcs_dict[pc_keys[2]]["loading_vector_16d"])

    axes[2].bar(channels - width, c_u1, width=width, label=f"PC1 ({evr[0]:.1f}%)", color="#1b9e77")
    axes[2].bar(channels, c_u2, width=width, label=f"PC2 ({evr[1]:.1f}%)", color="#d95f02")
    axes[2].bar(channels + width, c_u3, width=width, label=f"PC3 ({evr[2]:.1f}%)", color="#7570b3")

    axes[2].set_xticks(channels)
    axes[2].set_xlabel("Latent Channel Index (0 - 15)", fontsize=11)
    axes[2].set_ylabel("Loading Weight (u_k,c)", fontsize=11)
    axes[2].set_title("16D Latent Loadings (Linear Combination Weights)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "fase_a_pca_alignment_plot.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Generated clean diagnostic plot: {plot_path}")


# =========================== MAIN ENTRYPOINT ===========================
def main():
    global RESOLUTION, STEPS, GUIDANCE, GATE_FRAC, AVAILABLE_GPUS
    parser = argparse.ArgumentParser(description="Phase 1: PCA Characterization & Lab Alignment")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--gate-frac", type=float, default=GATE_FRAC)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=WORKERS_PER_GPU,
                        help="Number of parallel subprocess workers per physical GPU (e.g. 2 for 64GB A100).")
    parser.add_argument("--available-gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs (e.g. '0,1,2,3').")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Skip generation; analyze existing CSV file directly.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.available_gpus:
        AVAILABLE_GPUS = [int(x) for x in args.available_gpus.split(",")]

    RESOLUTION, STEPS, GUIDANCE, GATE_FRAC = args.resolution, args.steps, args.guidance, args.gate_frac
    channels_to_test = [int(x) for x in args.channels.split(",")] if args.channels else CHANNELS_TO_TEST
    workers_per_gpu = args.workers_per_gpu
    os.makedirs(args.out_dir, exist_ok=True)

    if args.analyze_only:
        analyze_pca(args.analyze_only, args.out_dir, channels_to_test)
        return

    if args.worker:
        _require_all_deps()
        seg_models = setup_seg_models("cpu")
        pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models)
        return

    _require_all_deps()
    n_tasks = len(REFERENCE_OBJECTS) * len(channels_to_test) * len(MAGNITUDES)
    print(f"\nPhase 1 Screening: {len(REFERENCE_OBJECTS)} object(s) x {len(channels_to_test)} channels x "
          f"{len(MAGNITUDES)} magnitudes = {n_tasks} complete generations ({STEPS} steps each)")

    if PARALLEL or args.parallel:
        run_parallel(args.out_dir, channels_to_test, workers_per_gpu=workers_per_gpu)
    else:
        seg_models = setup_seg_models("cpu")
        pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        csv_path = os.path.join(args.out_dir, "fase_a_raw.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for case_idx, (prompt, seed, obj_word) in enumerate(REFERENCE_OBJECTS):
                run_case(
                    pipe, vae, seg_models, case_idx, prompt, seed, obj_word,
                    args.out_dir, DEVICE, channels_to_test, writer, f
                )

    analyze_pca(os.path.join(args.out_dir, "fase_a_raw.csv"), args.out_dir, channels_to_test)


if __name__ == "__main__":
    main()

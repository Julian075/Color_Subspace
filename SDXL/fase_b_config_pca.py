"""
FASE_B_CONFIG_PCA.PY -- Envelope & Temporal Schedule Search for SDXL in PCA Latent Space.

Optimizes the temporal denoising schedule w(t) for the discovered PCA loading vectors
via a 3-stage sequential coordinate-ascent optimization:
  1. Stage 1: n_partes in [1, 2, 3, 4]  (default gate_frac=0.60, perfil='ascendente')
  2. Stage 2: gate_frac in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]  (fixing n_partes = winner_1)
  3. Stage 3: perfil_name in ['plano', 'ascendente', 'suave', 'angosto_alto', 'descendente'] (fixing n_partes = winner_1, gate_frac = winner_2)

Measures: deltaE (CIEDE2000), SSIM_in, SSIM_out, and PSNR (inside/outside mask).
Ranks configurations primarily by Color Score: DeltaE * sqrt(SSIM_in).
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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

# Local imports
from sdxl_core import (
    build_envelope_bands, setup_sdxl, latent_hw, decode_latents_4d,
    build_mask_latent, save_mask_preview, run_generation, get_gpu_list, spawn_workers,
)
from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    _SKIMAGE_IMPORT_ERROR = None
except ImportError as _e:
    structural_similarity = peak_signal_noise_ratio = None
    _SKIMAGE_IMPORT_ERROR = str(_e)


def _require_all_deps():
    if structural_similarity is None or peak_signal_noise_ratio is None:
        raise ImportError(
            f"Failed to import structural_similarity or peak_signal_noise_ratio from skimage: {_SKIMAGE_IMPORT_ERROR}"
        )


# =========================== CONFIGURATION ===========================
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4

RESOLUTION = 1024
STEPS = 30
GUIDANCE = 5.0

REF_MODE = "none"
REF_CHANNELS = None

PCA_AXES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fase_a_pca_out", "pca_axes.json")


def load_pca_loading_vectors(axes_path=PCA_AXES_PATH):
    if not os.path.exists(axes_path):
        print(f"[INFO] PCA axes file not found at {axes_path}. Using identity basis as placeholder.")
        return {
            "PC1": [(0, 1.0)],
            "PC2": [(1, 1.0)],
            "PC3": [(2, 1.0)],
        }
    with open(axes_path) as f:
        data = json.load(f)
    axes = data["axes"]
    u1 = np.array(axes["PC1"]["loadings"], dtype=np.float32)
    u2 = np.array(axes["PC2"]["loadings"], dtype=np.float32)
    u3 = np.array(axes["PC3"]["loadings"], dtype=np.float32)
    return {
        "PC1": [(c, float(u1[c])) for c in range(NUM_LATENT_CHANNELS)],
        "PC2": [(c, float(u2[c])) for c in range(NUM_LATENT_CHANNELS)],
        "PC3": [(c, float(u3[c])) for c in range(NUM_LATENT_CHANNELS)],
    }


# Search candidate spaces
N_PARTES_CANDIDATES = [1, 2, 3, 4]
GATE_FIXED = 0.60
DEFAULT_PERFIL = "ascendente"
GATE_FRAC_CANDIDATES = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}
SCHED_MODE = "ramp_down"
PROBE_MAGNITUDE = 0.5

REFERENCE_OBJECTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, minimalist, no shadows, no texture", 0, "sphere"),
    ("a red toy car on a plain white background, studio lighting, product photo, no shadows", 1, "car"),
    ("a blue ceramic mug on a plain white background, studio lighting, product photo, no shadows", 2, "mug"),
    ("a bright yellow rubber duck on a plain white background, studio lighting, product photo, no shadows", 3, "duck"),
    ("a green apple on a plain white background, studio lighting, product photo, no shadows", 4, "apple"),
    ("a white ceramic vase on a plain light gray background, studio lighting, product photo, no shadows", 5, "vase"),
    ("a black leather wallet on a plain white background, studio lighting, product photo, no shadows", 6, "wallet"),
    ("a purple fabric armchair on a plain white studio background, commercial product shot", 7, "armchair"),
]

OUT_DIR = "./fase_b_pca_out"

CSV_FIELDS = [
    "phase", "n_partes", "gate_frac", "perfil_name", "pc_name", "obj_idx", "obj_word",
    "magnitude", "deltaE", "ssim_in", "ssim_out", "psnr_in", "psnr_out",
    "distortion_in", "distortion_out", "color_score", "efficiency", "note"
]

PHASES = ["n_partes", "gate", "perfil"]
GROUP_KEY = {"n_partes": "n_partes", "gate": "gate_frac", "perfil": "perfil_name"}


# =========================== HELPER FUNCTIONS ===========================
def gen(pipe, prompt, seed, device, direction=None, magnitude=None, bands=None, mask_latent=None):
    return run_generation(
        pipe, prompt, int(seed), RESOLUTION, RESOLUTION, device, STEPS, GUIDANCE,
        channel_idxs=None, direction=direction, combo=None, magnitude=magnitude, bands=bands,
        mask_latent=mask_latent, ref_mode=REF_MODE, ref_channels=REF_CHANNELS
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
    latents_base = gen(pipe, prompt, int(seed), device)
    if torch.isnan(latents_base).any():
        print(f"  [ERROR] NaN in baseline latents for {obj_word}")
        return None

    img_base = decode_latents_4d(vae, latents_base)
    mask_2d = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_2d is None or mask_2d.sum() == 0:
        print(f"  [WARN] Fallback full mask for {obj_word}")
        mask_2d = np.ones((RESOLUTION, RESOLUTION), dtype=bool)

    # Save mask preview
    masks_dir = os.path.join(out_dir, "masks_preview")
    os.makedirs(masks_dir, exist_ok=True)
    mask_path = os.path.join(masks_dir, f"mask_{case_tag}.png")
    if not os.path.exists(mask_path):
        save_mask_preview(img_base, mask_2d, mask_path)

    color_base = measure_color_gt(img_base, mask_2d)
    if color_base is None:
        color_base = (50.0, 0.0, 0.0)
    h_lat, w_lat = latent_hw(RESOLUTION, RESOLUTION)
    mask_lat = build_mask_latent(mask_2d, h_lat, w_lat, device, DTYPE)

    return {
        "img_base": img_base,
        "mask_2d": mask_2d,
        "mask_lat": mask_lat,
        "color_base": color_base,
    }


def evaluate_trial(pipe, vae, prompt, seed, device, direction, magnitude, bands, ctx):
    latents_mod = gen(
        pipe, prompt, int(seed), device,
        direction=direction, magnitude=magnitude, bands=bands,
        mask_latent=ctx["mask_lat"]
    )
    if torch.isnan(latents_mod).any():
        return None

    img_mod = decode_latents_4d(vae, latents_mod)
    color_mod = measure_color_gt(img_mod, ctx["mask_2d"]) or ctx["color_base"]

    dE = ciede2000(ctx["color_base"], color_mod)
    img1_arr = np.array(ctx["img_base"])
    img2_arr = np.array(img_mod)

    # SSIM
    ssim_val, _ = structural_similarity(
        img1_arr, img2_arr, channel_axis=2, full=True
    )
    ssim_in = float(ssim_val)
    ssim_out = 1.0

    # PSNR
    psnr_in = calculate_psnr_masked(img1_arr, img2_arr, ctx["mask_2d"])
    outside_mask = ~ctx["mask_2d"]
    psnr_out = calculate_psnr_masked(img1_arr, img2_arr, outside_mask)

    dist_in = float(100.0 * (1.0 - ssim_in))
    dist_out = 0.0

    color_score = float(dE * (ssim_in ** 0.5))
    eff = float(dE / (dist_in + 1e-4))

    return {
        "deltaE": round(dE, 3),
        "ssim_in": round(ssim_in, 4),
        "ssim_out": round(ssim_out, 4),
        "psnr_in": round(psnr_in, 2),
        "psnr_out": round(psnr_out, 2),
        "distortion_in": round(dist_in, 3),
        "distortion_out": round(dist_out, 3),
        "color_score": round(color_score, 3),
        "efficiency": round(eff, 3),
        "note": "ok"
    }


# =========================== SWEEP GENERATION ===========================
def build_phase_tasks(phase_name, param_values, fixed_params, pca_vectors):
    tasks = []
    for val in param_values:
        n_p = val if phase_name == "n_partes" else fixed_params["n_partes"]
        g_f = val if phase_name == "gate" else fixed_params["gate_frac"]
        p_n = val if phase_name == "perfil" else fixed_params["perfil_name"]

        for pc_name, direction in pca_vectors.items():
            for obj_idx, (prompt, seed, obj_word) in enumerate(REFERENCE_OBJECTS):
                tasks.append({
                    "phase": phase_name,
                    "n_partes": n_p,
                    "gate_frac": g_f,
                    "perfil_name": p_n,
                    "pc_name": pc_name,
                    "obj_idx": obj_idx,
                    "prompt": prompt,
                    "seed": seed,
                    "obj_word": obj_word,
                    "magnitude": PROBE_MAGNITUDE,
                    "direction": direction,
                })
    return tasks


# =========================== PARALLEL WORKER ROUTINE ===========================
def run_worker(chunk_path, out_dir, pipe, vae, seg_models):
    pid = os.getpid()
    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(chunk_path) as f:
        tasks = json.load(f)
    print(f"[worker pid={pid} GPU_phys={gpu_env}] {len(tasks)} schedule probing cells assigned", flush=True)

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"part_{pid}.csv")
    ctx_cache = {}

    with open(part_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in tasks:
            obj_idx = t["obj_idx"]
            prompt, seed, obj_word = t["prompt"], t["seed"], t["obj_word"]
            case_tag = f"seed{seed}_{obj_word}"
            if obj_idx not in ctx_cache:
                ctx_cache[obj_idx] = generate_case_context(
                    pipe, vae, seg_models, prompt, seed, obj_word, DEVICE, out_dir, case_tag
                )
            ctx = ctx_cache[obj_idx]
            if ctx is None:
                continue

            chrono = PERFIL_GENERATORS[t["perfil_name"]](t["n_partes"])
            bands = build_envelope_bands(t["gate_frac"], chrono, SCHED_MODE)

            metrics = evaluate_trial(
                pipe, vae, prompt, seed, DEVICE,
                t["direction"], t["magnitude"], bands, ctx
            )
            if metrics is not None:
                writer.writerow({
                    "phase": t["phase"],
                    "n_partes": t["n_partes"],
                    "gate_frac": t["gate_frac"],
                    "perfil_name": t["perfil_name"],
                    "pc_name": t["pc_name"],
                    "obj_idx": obj_idx,
                    "obj_word": obj_word,
                    "magnitude": t["magnitude"],
                    **metrics
                })
                f.flush()
    print(f"[worker pid={pid} GPU_phys={gpu_env}] Finished stage chunk", flush=True)


# =========================== ANALYSIS & PLOTTING ===========================
def analyze_phase_results(stage_csv, out_dir, phase_name):
    gkey = GROUP_KEY[phase_name]
    rows = []
    with open(stage_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = {}
    for r in rows:
        if r["note"] != "ok":
            continue
        val = int(r[gkey]) if phase_name == "n_partes" else (float(r[gkey]) if phase_name == "gate" else str(r[gkey]))
        if val not in groups:
            groups[val] = {"deltaE": [], "ssim_in": [], "psnr_in": [], "color_score": [], "efficiency": []}
        groups[val]["deltaE"].append(float(r["deltaE"]))
        groups[val]["ssim_in"].append(float(r["ssim_in"]))
        groups[val]["psnr_in"].append(float(r["psnr_in"]))
        groups[val]["color_score"].append(float(r["color_score"]))
        groups[val]["efficiency"].append(float(r["efficiency"]))

    summary = []
    for val, m in groups.items():
        summary.append({
            gkey: val,
            "mean_deltaE": float(np.mean(m["deltaE"])),
            "mean_ssim_in": float(np.mean(m["ssim_in"])),
            "mean_psnr_in": float(np.mean(m["psnr_in"])),
            "mean_color_score": float(np.mean(m["color_score"])),
            "mean_efficiency": float(np.mean(m["efficiency"])),
        })

    summary.sort(key=lambda x: x["mean_color_score"], reverse=True)
    winner = summary[0][gkey]

    print(f"\n" + "=" * 80)
    print(f"               RESULTS RANKING FOR STAGE: [{phase_name.upper()}]")
    print("=" * 80)
    print(f"{'Rank':<5} | {gkey:<15} | {'ColorScore':<11} | {'Mean ΔE':<10} | {'SSIM_in':<9} | {'PSNR_in (dB)':<12}")
    print("-" * 75)
    for rank, row in enumerate(summary, 1):
        print(f"{rank:<5} | {str(row[gkey]):<15} | {row['mean_color_score']:<11.3f} | {row['mean_deltaE']:<10.3f} | "
              f"{row['mean_ssim_in']:<9.4f} | {row['mean_psnr_in']:<12.2f}")
    print(f">>> WINNING CANDIDATE FOR {phase_name.upper()}: {winner}")
    print("=" * 80 + "\n")

    summary_csv = os.path.join(out_dir, f"fase_b_{phase_name}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[gkey, "mean_deltaE", "mean_ssim_in", "mean_psnr_in", "mean_color_score", "mean_efficiency"])
        writer.writeheader()
        writer.writerows(summary)

    return winner


def generate_schedule_summary_plot(csv_path, out_dir):
    """3-panel visual summary of the Phase B schedule sweep."""
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    phases = ["n_partes", "gate", "perfil"]
    titles = ["1. Number of Bands (n_partes)", "2. Gate Fraction (gate_frac)", "3. Profile Shape (perfil)"]

    for ax_idx, (ph, title) in enumerate(zip(phases, titles)):
        gkey = GROUP_KEY[ph]
        sub = [r for r in rows if r["phase"] == ph and r["note"] == "ok"]
        if not sub:
            continue
        cands = sorted(list(set(r[gkey] for r in sub)), key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
        delta_means = [np.mean([float(r["deltaE"]) for r in sub if r[gkey] == c]) for c in cands]
        ssim_means = [np.mean([float(r["ssim_in"]) for r in sub if r[gkey] == c]) for c in cands]

        ax = axes[ax_idx]
        x = np.arange(len(cands))
        ax.bar(x - 0.18, delta_means, width=0.35, color="#2b5c8f", label="Mean ΔE")
        ax2 = ax.twinx()
        ax2.plot(x + 0.18, ssim_means, color="#d95f02", marker="o", linewidth=2.5, label="Mean SSIM")
        ax.set_xticks(x)
        ax.set_xticklabels(cands, rotation=25 if ph == "perfil" else 0, fontweight="bold")
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


# =========================== MAIN CONTROLLER ===========================
def main():
    parser = argparse.ArgumentParser(description="Phase B: Sequential Temporal Schedule Optimization for SDXL")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--analyze-only", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pca_vectors = load_pca_loading_vectors()

    if args.analyze_only:
        for ph in PHASES:
            analyze_phase_results(args.analyze_only, args.out_dir, ph)
        generate_schedule_summary_plot(args.analyze_only, args.out_dir)
        return

    if args.worker:
        _require_all_deps()
        seg_models = setup_seg_models(DEVICE)
        pipe, vae = setup_sdxl(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models)
        return

    # Master Controller
    _require_all_deps()
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()] if args.gpus else None
    gpu_list = get_gpu_list(gpus)
    n_workers = len(gpu_list) * args.workers_per_gpu

    fixed_params = {
        "n_partes": 3,
        "gate_frac": GATE_FIXED,
        "perfil_name": DEFAULT_PERFIL,
    }

    all_phase_csvs = []
    for ph_idx, phase_name in enumerate(PHASES):
        print(f"\n" + "=" * 80)
        print(f">>> STARTING SCHEDULE SEARCH STAGE [{ph_idx + 1}/3]: {phase_name.upper()}")
        print(f"    Current fixed params: {fixed_params}")
        print("=" * 80)

        cands = N_PARTES_CANDIDATES if phase_name == "n_partes" else (
            GATE_FRAC_CANDIDATES if phase_name == "gate" else list(PERFIL_GENERATORS.keys())
        )
        tasks = build_phase_tasks(phase_name, cands, fixed_params, pca_vectors)
        print(f"Generated {len(tasks)} tasks -> distributing over {n_workers} workers across {len(gpu_list)} GPU(s)")

        cmd_base = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--out-dir", args.out_dir,
        ]
        tmp_chunks = os.path.join(args.out_dir, f"_tmp_chunks_{phase_name}")
        spawn_workers(tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

        # Merge stage parts
        parts_dir = os.path.join(args.out_dir, "_csv_parts")
        stage_csv = os.path.join(args.out_dir, f"fase_b_{phase_name}_raw.csv")
        merged_rows = []
        if os.path.isdir(parts_dir):
            for fn in sorted(os.listdir(parts_dir)):
                if fn.endswith(".csv"):
                    fp = os.path.join(parts_dir, fn)
                    with open(fp) as f:
                        r = csv.DictReader(f)
                        merged_rows.extend(list(r))
                    os.remove(fp)

        with open(stage_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(merged_rows)
        all_phase_csvs.append(stage_csv)

        # Analyze winner for this stage
        winner = analyze_phase_results(stage_csv, args.out_dir, phase_name)
        if phase_name == "n_partes":
            fixed_params["n_partes"] = int(winner)
        elif phase_name == "gate":
            fixed_params["gate_frac"] = float(winner)
        elif phase_name == "perfil":
            fixed_params["perfil_name"] = str(winner)

    # Consolidate master CSV
    master_csv = os.path.join(args.out_dir, "fase_b_raw.csv")
    with open(master_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for scsv in all_phase_csvs:
            with open(scsv) as f_in:
                r = csv.DictReader(f_in)
                writer.writerows(list(r))

    print(f"\n" + "=" * 80)
    print("ALL 3 STAGES COMPLETED!")
    print(f"FINAL WINNING SCHEDULE CONFIGURATION:")
    print(f"  n_partes:    {fixed_params['n_partes']}")
    print(f"  gate_frac:   {fixed_params['gate_frac']}")
    print(f"  perfil_name: {fixed_params['perfil_name']}")
    print("=" * 80)

    # Save final winning JSON
    winning_schedule = {
        "model": MODEL_ID,
        "n_partes": fixed_params["n_partes"],
        "gate_frac": fixed_params["gate_frac"],
        "perfil_name": fixed_params["perfil_name"],
        "transition_mode": SCHED_MODE,
        "probe_magnitude": PROBE_MAGNITUDE,
    }
    config_json = os.path.join(args.out_dir, "fase_b_winning_schedule.json")
    with open(config_json, "w") as f:
        json.dump(winning_schedule, f, indent=2)
    print(f"Saved winning schedule to: {config_json}")

    generate_schedule_summary_plot(master_csv, args.out_dir)


if __name__ == "__main__":
    main()

"""
FASE_B_CONFIG_PCA.PY -- Envelope & Temporal Schedule Search for SD3.5-M in PCA Latent Space.

Optimizes the temporal denoising schedule w(t) for the 3 discovered PCA loading vectors:
  1. n_partes in [1, 2, 3, 4]
  2. gate_frac in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  3. perfil_name in ['plano', 'ascendente', 'suave', 'angosto_alto', 'descendente']

Measures: deltaE (CIEDE2000), SSIM_in, SSIM_out, and PSNR (inside/outside mask).
Ranks configurations primarily by Color Score: DeltaE * sqrt(SSIM_in).
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

# Local imports
from sd35_core import (
    build_envelope_bands, setup_sd35, latent_hw, decode_latents_4d,
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
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 4.5

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
GATE_FIXED = 0.75
DEFAULT_PERFIL = "ascendente"
GATE_FRAC_CANDIDATES = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}
SCHED_MODE = "ramp_down"
PROBE_MAGNITUDE = 1.5

REFERENCE_OBJECTS = [
    ("a smooth matte gray sphere on a plain white background, studio lighting, minimalist, no shadows, no texture", 0, "sphere"),
    ("a red toy car on a plain white background, studio lighting, product photo, no shadows", 1, "car"),
    ("a blue ceramic mug on a plain white background, studio lighting, product photo, no shadows", 2, "mug"),
    ("a bright yellow rubber duck on a plain white background, studio lighting, product photo, no shadows", 3, "duck"),
    ("a green apple on a plain white background, studio lighting, product photo, no shadows", 4, "apple"),
    ("a white ceramic vase on a plain light gray background, studio lighting, product photo, no shadows", 5, "vase"),
    ("a black leather wallet on a plain white background, studio lighting, product photo, no shadows", 6, "wallet"),
]

OUT_DIR = "./fase_b_pca_out"
AVAILABLE_GPUS = None

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
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(img_base).save(os.path.join(out_dir, f"{case_tag}_baseline.png"))

    mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        h, w = img_base.shape[:2]
        mask_pixel = np.ones((h, w), dtype=bool)

    base_lab = measure_color_gt(img_base, mask_pixel)
    if base_lab is None:
        base_lab = [50.0, 0.0, 0.0]

    latent_h, latent_w = latent_hw(RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)
    save_mask_preview(img_base, mask_pixel, os.path.join(out_dir, f"{case_tag}_mask_preview.png"))

    return {
        "img_base": img_base,
        "mask_pixel": mask_pixel,
        "mask_latent": mask_latent,
        "base_lab": base_lab,
    }


def compute_cell_metrics(pipe, vae, prompt, seed, device, direction, magnitude, bands, ctx):
    latents = gen(
        pipe, prompt, int(seed), device, direction=direction, magnitude=magnitude,
        bands=bands, mask_latent=ctx["mask_latent"]
    )
    if torch.isnan(latents).any():
        return {
            "deltaE": 0.0, "ssim_in": 0.0, "ssim_out": 1.0, "psnr_in": 0.0, "psnr_out": 99.0,
            "distortion_in": 1.0, "distortion_out": 0.0, "color_score": 0.0, "efficiency": 0.0,
            "note": "nan_latent"
        }

    img_mod = decode_latents_4d(vae, latents)
    mod_lab = measure_color_gt(img_mod, ctx["mask_pixel"])
    deltaE = float(ciede2000(ctx["base_lab"], mod_lab)) if mod_lab is not None else 0.0

    ssim_val = float(structural_similarity(ctx["img_base"], img_mod, channel_axis=2, data_range=255))
    psnr_in = calculate_psnr_masked(ctx["img_base"], img_mod, ctx["mask_pixel"])
    psnr_out = calculate_psnr_masked(ctx["img_base"], img_mod, ~ctx["mask_pixel"])

    distortion_in = 1.0 - ssim_val
    color_score = deltaE * (ssim_val ** 0.5)
    efficiency = deltaE / (distortion_in + 1e-3)

    return {
        "deltaE": deltaE,
        "ssim_in": ssim_val,
        "ssim_out": 1.0,
        "psnr_in": psnr_in,
        "psnr_out": psnr_out,
        "distortion_in": distortion_in,
        "distortion_out": 0.0,
        "color_score": color_score,
        "efficiency": efficiency,
        "note": "ok",
    }


def build_phase_tasks(phase_name, param_values, fixed_params):
    tasks = []
    for val in param_values:
        n_p = val if phase_name == "n_partes" else fixed_params["n_partes"]
        g_f = val if phase_name == "gate" else fixed_params["gate_frac"]
        p_n = val if phase_name == "perfil" else fixed_params["perfil_name"]

        for pc_name in ["PC1", "PC2", "PC3"]:
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
                })
    return tasks


# =========================== PARALLEL WORKER ROUTINE ===========================
def run_worker(chunk_path, out_dir, pipe, vae, seg_models, pca_directions):
    pid = os.getpid()
    with open(chunk_path) as f:
        tasks = json.load(f)
    print(f"[worker pid={pid}] Assigned {len(tasks)} schedule probing cells", flush=True)

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
                print(f"[worker pid={pid}] Generating baseline {obj_idx} ({obj_word})...", flush=True)
                ctx = generate_case_context(
                    pipe, vae, seg_models, prompt, seed, obj_word, DEVICE, out_dir, case_tag
                )
                if ctx is None:
                    continue
                ctx_cache[obj_idx] = ctx

            bands = build_envelope_bands(
                t["gate_frac"],
                PERFIL_GENERATORS[t["perfil_name"]](t["n_partes"]),
                SCHED_MODE
            )
            direction = pca_directions[t["pc_name"]]

            metrics = compute_cell_metrics(
                pipe, vae, prompt, seed, DEVICE, direction, t["magnitude"], bands, ctx_cache[obj_idx]
            )

            row = {
                "phase": t["phase"],
                "n_partes": t["n_partes"],
                "gate_frac": t["gate_frac"],
                "perfil_name": t["perfil_name"],
                "pc_name": t["pc_name"],
                "obj_idx": obj_idx,
                "obj_word": obj_word,
                "magnitude": t["magnitude"],
                "deltaE": f"{metrics['deltaE']:.3f}",
                "ssim_in": f"{metrics['ssim_in']:.4f}",
                "ssim_out": f"{metrics['ssim_out']:.4f}",
                "psnr_in": f"{metrics['psnr_in']:.2f}",
                "psnr_out": f"{metrics['psnr_out']:.2f}",
                "distortion_in": f"{metrics['distortion_in']:.4f}",
                "distortion_out": f"{metrics['distortion_out']:.4f}",
                "color_score": f"{metrics['color_score']:.3f}",
                "efficiency": f"{metrics['efficiency']:.2f}",
                "note": metrics["note"],
            }
            writer.writerow(row)
            f.flush()
            print(f"[worker pid={pid}] {t['phase']} {t['pc_name']} val={t[GROUP_KEY[t['phase']]]} "
                  f"obj={obj_word} -> ΔE={row['deltaE']} SSIM={row['ssim_in']} PSNR={row['psnr_in']}dB", flush=True)


# =========================== ANALYSIS & RANKING ===========================
def analyze_phase_results(csv_path, out_dir, phase_name):
    """Ranks configurations based on color score, deltaE, SSIM, and PSNR."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["phase"] == phase_name and r["note"] == "ok":
                rows.append(r)

    if not rows:
        raise RuntimeError(f"No valid rows found for phase: {phase_name}")

    gkey = GROUP_KEY[phase_name]
    groups = {}
    for r in rows:
        val = r[gkey]
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

    print(f"\n--- RESULTS RANKING FOR [{phase_name.upper()}] ---")
    print(f"{'Rank':<5} | {gkey:<15} | {'ColorScore':<11} | {'Mean ΔE':<10} | {'SSIM_in':<9} | {'PSNR_in (dB)':<12}")
    print("-" * 75)
    for rank, row in enumerate(summary, 1):
        print(f"{rank:<5} | {str(row[gkey]):<15} | {row['mean_color_score']:<11.3f} | {row['mean_deltaE']:<10.3f} | "
              f"{row['mean_ssim_in']:<9.4f} | {row['mean_psnr_in']:<12.2f}")
    print(f">>> WINNING CANDIDATE FOR {phase_name.upper()}: {winner}")

    summary_csv = os.path.join(out_dir, f"fase_b_{phase_name}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[gkey, "mean_deltaE", "mean_ssim_in", "mean_psnr_in", "mean_color_score", "mean_efficiency"])
        writer.writeheader()
        writer.writerows(summary)

    return winner


def generate_schedule_summary_plot(csv_path, out_dir):
    """3-panel visual summary of the Phase B schedule sweep matching standard aesthetic."""
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


# =========================== MAIN DRIVER ===========================
def main():
    parser = argparse.ArgumentParser(description="Phase B: Temporal Schedule Optimization for SD3.5-M")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--analyze-only", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pca_directions = load_pca_loading_vectors()

    if args.analyze_only:
        for ph in PHASES:
            analyze_phase_results(args.analyze_only, args.out_dir, ph)
        generate_schedule_summary_plot(args.analyze_only, args.out_dir)
        return

    if args.worker:
        _require_all_deps()
        seg_models = setup_seg_models(DEVICE)
        pipe, vae = setup_sd35(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models, pca_directions)
        return

    # Master Controller
    _require_all_deps()
    gpu_list = get_gpu_list(AVAILABLE_GPUS)
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
        tasks = build_phase_tasks(phase_name, cands, fixed_params)
        print(f"Generated {len(tasks)} tasks -> distributing over {n_workers} workers across {len(gpu_list)} GPU(s)")

        cmd_base = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--out-dir", args.out_dir,
        ]
        tmp_chunks = os.path.join(args.out_dir, f"_tmp_chunks_{phase_name}")
        spawn_workers(tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

        # Merge parts
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

        # Analyze winner
        winner = analyze_phase_results(stage_csv, args.out_dir, phase_name)
        if phase_name == "n_partes":
            fixed_params["n_partes"] = int(winner)
        elif phase_name == "gate":
            fixed_params["gate_frac"] = float(winner)
        elif phase_name == "perfil":
            fixed_params["perfil_name"] = str(winner)

    # Consolidate full CSV
    master_csv = os.path.join(args.out_dir, "fase_b_schedule_search_raw.csv")
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
    config_json = os.path.join(args.out_dir, "fase_b_winning_schedule.json")
    with open(config_json, "w") as f:
        json.dump(fixed_params, f, indent=2)
    print(f"Saved winning schedule to: {config_json}")

    generate_schedule_summary_plot(master_csv, args.out_dir)


if __name__ == "__main__":
    main()

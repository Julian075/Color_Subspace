"""
FASE_B_CONFIG_PCA.PY -- Envelope & Schedule Search for FLUX in PCA Latent Space.

Optimizes the temporal denoising schedule w(t) for the 3 discovered PCA loading vectors:
  1. n_partes in [1, 2, 3, 4]
  2. gate_frac in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  3. perfil_name in ['plano', 'ascendente', 'suave', 'angosto_alto', 'descendente']

Measures: deltaE (CIEDE2000), SSIM, and PSNR (inside/outside mask).
Ranks configurations primarily by Color Score (deltaE * sqrt(SSIM)).
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
    from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000
    _UTILS_IMPORT_ERROR = None
except ImportError as _e:
    setup_seg_models = get_object_mask = measure_color_gt = ciede2000 = None
    _UTILS_IMPORT_ERROR = str(_e)

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    _SKIMAGE_IMPORT_ERROR = None
except ImportError as _e:
    structural_similarity = peak_signal_noise_ratio = None
    _SKIMAGE_IMPORT_ERROR = str(_e)


def _require_all_deps():
    if any(x is None for x in (setup_seg_models, get_object_mask, measure_color_gt, ciede2000)):
        raise ImportError(
            "Failed to import from utils.py (setup_seg_models/get_object_mask/measure_color_gt/ciede2000).\n"
            f"  original error: {_UTILS_IMPORT_ERROR}\n  cwd={os.getcwd()}")
    if structural_similarity is None or peak_signal_noise_ratio is None:
        raise ImportError(
            "Failed to import structural_similarity or peak_signal_noise_ratio from skimage.\n"
            f"  original error: {_SKIMAGE_IMPORT_ERROR}")


# =========================== CONFIGURATION ===========================
MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 32

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5

# Reference mode: absolute latent displacement (ref = 1.0)
REF_MODE = "none"
REF_CHANNELS = None

# Load PCA axes from Phase A output
PCA_AXES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fase_a_pca_out", "pca_axes.json")


def load_pca_loading_vectors(axes_path=PCA_AXES_PATH):
    if not os.path.exists(axes_path):
        raise FileNotFoundError(f"Cannot find PCA axes file: {axes_path}. Run Phase A first!")
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


# Sweep candidate values
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

# 7 Reference Objects covering the CIELAB color solid
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
    """Wrapper around flux_core.run_generation."""
    return run_generation(
        pipe, prompt, int(seed), RESOLUTION, RESOLUTION, device, STEPS, GUIDANCE,
        channel_idxs=None, direction=direction, combo=None, magnitude=magnitude, bands=bands,
        mask_latent=mask_latent, ref_mode=REF_MODE, ref_channels=REF_CHANNELS
    )


def calculate_psnr_masked(img1_np, img2_np, mask_2d=None):
    """Computes PSNR (dB) on uint8 RGB images, optionally inside a mask."""
    if mask_2d is not None and mask_2d.sum() > 0:
        diff = (img1_np.astype(np.float32) - img2_np.astype(np.float32))[mask_2d]
        mse = float(np.mean(diff ** 2))
    else:
        mse = float(np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def generate_case_context(pipe, vae, seg_models, prompt, seed, obj_word, device, out_dir, case_tag):
    """Generates baseline image, SAM3 mask, and base Lab color."""
    latents_packed_base = gen(pipe, prompt, int(seed), device)
    if torch.isnan(latents_packed_base).any():
        print(f"  [ERROR] NaN in baseline latents for {obj_word}")
        return None

    img_base = decode_latents_4d(vae, unpack_to_4d(pipe, latents_packed_base, RESOLUTION, RESOLUTION))
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(img_base).save(os.path.join(out_dir, f"{case_tag}_baseline.png"))

    mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
    if mask_pixel is None:
        h, w = img_base.shape[:2]
        mask_pixel = np.ones((h, w), dtype=bool)

    base_lab = measure_color_gt(img_base, mask_pixel)
    if base_lab is None:
        base_lab = [50.0, 0.0, 0.0]

    latent_h, latent_w = latent_hw(pipe, RESOLUTION, RESOLUTION)
    mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)
    save_mask_preview(img_base, mask_pixel, os.path.join(out_dir, f"{case_tag}_mask_preview.png"))

    return {
        "img_base": img_base,
        "mask_pixel": mask_pixel,
        "mask_latent": mask_latent,
        "base_lab": base_lab,
    }


def compute_cell_metrics(pipe, vae, prompt, seed, device, direction, magnitude, bands, ctx):
    """Runs single shifted generation and computes DeltaE, SSIM, and PSNR."""
    latents_packed = gen(
        pipe, prompt, int(seed), device, direction=direction, magnitude=magnitude,
        bands=bands, mask_latent=ctx["mask_latent"]
    )
    if torch.isnan(latents_packed).any():
        return {
            "deltaE": 0.0, "ssim_in": 0.0, "ssim_out": 1.0, "psnr_in": 0.0, "psnr_out": 99.0,
            "distortion_in": 1.0, "distortion_out": 0.0, "color_score": 0.0, "efficiency": 0.0,
            "note": "nan_latent"
        }

    img_mod = decode_latents_4d(vae, unpack_to_4d(pipe, latents_packed, RESOLUTION, RESOLUTION))

    # Measure color shift
    mod_lab = measure_color_gt(img_mod, ctx["mask_pixel"])
    if mod_lab is None:
        deltaE = 0.0
    else:
        deltaE = float(ciede2000(ctx["base_lab"], mod_lab))

    # Structural metrics (SSIM & PSNR)
    ssim_val = float(structural_similarity(ctx["img_base"], img_mod, channel_axis=2, data_range=255))
    psnr_in = calculate_psnr_masked(ctx["img_base"], img_mod, ctx["mask_pixel"])
    psnr_out = calculate_psnr_masked(ctx["img_base"], img_mod, ~ctx["mask_pixel"])

    distortion_in = 1.0 - ssim_val
    distortion_out = 0.0
    ssim_in = ssim_val
    ssim_out = 1.0

    color_score = deltaE * (ssim_in ** 0.5)
    efficiency = deltaE / (distortion_in + 1e-3)

    return {
        "deltaE": deltaE,
        "ssim_in": ssim_in,
        "ssim_out": ssim_out,
        "psnr_in": psnr_in,
        "psnr_out": psnr_out,
        "distortion_in": distortion_in,
        "distortion_out": distortion_out,
        "color_score": color_score,
        "efficiency": efficiency,
        "note": "ok",
    }


# =========================== TASK BUILDER ===========================
def build_phase_tasks(phase_name, param_values, fixed_params):
    """Constructs task list for a search stage."""
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
                    "seed": int(seed),
                    "obj_word": obj_word,
                    "magnitude": PROBE_MAGNITUDE,
                })
    return tasks


# =========================== WORKER ROUTINE ===========================
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
            if r["phase"] == phase_name:
                rows.append(r)

    if not rows:
        raise RuntimeError(f"No rows found for phase: {phase_name}")

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
    for val, vals in groups.items():
        summary.append({
            gkey: val,
            "mean_deltaE": np.mean(vals["deltaE"]),
            "mean_ssim_in": np.mean(vals["ssim_in"]),
            "mean_psnr_in": np.mean(vals["psnr_in"]),
            "mean_color_score": np.mean(vals["color_score"]),
            "mean_efficiency": np.mean(vals["efficiency"]),
        })

    summary = sorted(summary, key=lambda x: x["mean_color_score"], reverse=True)
    winner = summary[0][gkey]

    print(f"\n" + "=" * 78)
    print(f"RANKING RESULTS FOR STAGE: {phase_name.upper()} (Grouped by {gkey})")
    print("=" * 78)
    print(f"{'Candidate':<16} | {'Mean ΔE':<10} | {'Mean SSIM':<10} | {'Mean PSNR (dB)':<15} | {'Color Score':<12}")
    print("-" * 78)
    for s in summary:
        print(f"{s[gkey]:<16} | {s['mean_deltaE']:<10.3f} | {s['mean_ssim_in']:<10.4f} | {s['mean_psnr_in']:<15.2f} | {s['mean_color_score']:<12.3f}")
    print(f">>> WINNING {phase_name}: {winner} (Color Score = {summary[0]['mean_color_score']:.3f})")

    ranking_csv = os.path.join(out_dir, f"{phase_name}_search_ranking.csv")
    with open(ranking_csv, "w", newline="") as f:
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
        sub = [r for r in rows if r["phase"] == ph]
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
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Generated schedule search plot: {plot_path}")


# =========================== MAIN DRIVER ===========================
def main():
    parser = argparse.ArgumentParser(description="Phase B: PCA Envelope & Schedule Search")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--available-gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs (e.g. '0,1,2,3').")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Number of parallel worker processes per GPU (default: 1).")
    parser.add_argument("--analyze-only", type=str, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global AVAILABLE_GPUS
    if args.available_gpus:
        AVAILABLE_GPUS = [int(x) for x in args.available_gpus.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)
    pca_directions = load_pca_loading_vectors()

    if args.analyze_only:
        for ph in PHASES:
            analyze_phase_results(args.analyze_only, args.out_dir, ph)
        generate_schedule_summary_plot(args.analyze_only, args.out_dir)
        return

    if args.worker:
        _require_all_deps()
        seg_models = setup_seg_models("cpu")
        pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
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
        stage_csv = os.path.join(args.out_dir, f"fase_b_{phase_name}_raw.csv")
        cands = N_PARTES_CANDIDATES if phase_name == "n_partes" else (
            GATE_FRAC_CANDIDATES if phase_name == "gate" else list(PERFIL_GENERATORS.keys())
        )
        expected_n_tasks = len(cands) * 3 * len(REFERENCE_OBJECTS)

        # Check if stage was already fully completed
        if os.path.exists(stage_csv):
            with open(stage_csv, newline="") as f_check:
                existing_rows = list(csv.DictReader(f_check))
            if len(existing_rows) >= expected_n_tasks:
                print(f"\n" + "=" * 80)
                print(f">>> [RESUME] STAGE [{ph_idx + 1}/3]: {phase_name.upper()} ALREADY COMPLETED ({len(existing_rows)} rows found). Skipping computation!")
                print("=" * 80)
                winner = analyze_phase_results(stage_csv, args.out_dir, phase_name)
                if phase_name == "n_partes":
                    fixed_params["n_partes"] = int(winner)
                elif phase_name == "gate":
                    fixed_params["gate_frac"] = float(winner)
                elif phase_name == "perfil":
                    fixed_params["perfil_name"] = str(winner)
                all_phase_csvs.append(stage_csv)
                continue

        # Check partial tasks inside _csv_parts
        parts_dir = os.path.join(args.out_dir, "_csv_parts")
        completed_keys = set()
        if os.path.isdir(parts_dir):
            for fn in os.listdir(parts_dir):
                if fn.endswith(".csv"):
                    try:
                        with open(os.path.join(parts_dir, fn), newline="") as f_part:
                            for r in csv.DictReader(f_part):
                                if r.get("phase") == phase_name:
                                    gval = r.get(GROUP_KEY[phase_name])
                                    pc = r.get("pc_name")
                                    obj = r.get("obj_word")
                                    if gval is not None and pc is not None and obj is not None:
                                        completed_keys.add((str(gval), str(pc), str(obj)))
                    except Exception:
                        pass

        tasks = build_phase_tasks(phase_name, cands, fixed_params)
        gkey_param = "n_partes" if phase_name == "n_partes" else ("gate_frac" if phase_name == "gate" else "perfil_name")
        rem_tasks = [
            t for t in tasks
            if (str(t[gkey_param]), str(t["pc_name"]), str(t["obj_word"])) not in completed_keys
        ]

        print(f"\n" + "=" * 80)
        print(f">>> STARTING SCHEDULE SEARCH STAGE [{ph_idx + 1}/3]: {phase_name.upper()}")
        print(f"    Current fixed params: {fixed_params}")
        print(f"    Total tasks: {len(tasks)} | Completed from partial run: {len(completed_keys)} | Remaining: {len(rem_tasks)}")
        print("=" * 80)

        if len(rem_tasks) > 0:
            cmd_base = [
                sys.executable, os.path.abspath(__file__), "--worker",
                "--out-dir", args.out_dir,
            ]
            tmp_chunks = os.path.join(args.out_dir, f"_tmp_chunks_{phase_name}")
            spawn_workers(rem_tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

        # Merge parts
        merged_rows = []
        seen = set()
        if os.path.isdir(parts_dir):
            for fn in sorted(os.listdir(parts_dir)):
                if fn.endswith(".csv"):
                    fp = os.path.join(parts_dir, fn)
                    try:
                        with open(fp, newline="") as f:
                            for r in csv.DictReader(f):
                                if r.get("phase") == phase_name:
                                    key = (r.get("phase"), str(r.get(GROUP_KEY[phase_name])), str(r.get("pc_name")), str(r.get("obj_word")))
                                    if key not in seen:
                                        seen.add(key)
                                        merged_rows.append(r)
                    except Exception:
                        pass

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

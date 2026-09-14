"""
COLECCION_DATOS_MLP_PCA.PY -- High-Performance Phase C Dataset Generator for PixArt.

Generates 25,000 paired training samples (3,125 baselines x 8 variants) for learning
the continuous color transfer map in PCA latent subspace:
  Input:  (base_lab, target_lab) -> Output: (m1, m2, m3)

Optimizations:
  - 100% GPU-resident in FP16 (activates Turing Tensor Cores for ~7s/generation)
  - 1 Baseline Generation + 1 Segmentation per 8 variants (87.5% reduction in segmentation overhead)
  - Multi-axial 3D spherical direction sampling on S^2 (Cardinals, Bisections, Diagonals, Fibonacci)
  - Calibrated magnitude distribution (80% Zone 1 [0.05, 0.45], 20% Zone 2 [0.45, 1.25])
  - Per-baseline checkpointing and atomic resume
"""

import os
import sys
import gc
import json
import math
import random
import argparse
import subprocess
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Local imports
from pixart_core import (
    MODEL_ID_ALPHA, MODEL_ID_SIGMA, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION,
    STEPS, GUIDANCE, MAX_SEQ_LEN_ALPHA, MAX_SEQ_LEN_SIGMA,
    setup_pixart, decode_latents_4d, build_envelope_bands, build_mask_latent,
    latent_hw, run_generation, PERFIL_GENERATORS, spawn_workers, get_available_gpus
)
import utils

N_BASELINES_TOTAL = 3125
K_VARIANTS_PER_BASELINE = 8
MASTER_SEED = 20260825

P_ZONE1_LINEAR = 0.80
ZONE1_RANGE = (0.05, 0.45)
ZONE2_RANGE = (0.45, 1.25)

# 100 Diverse Object Categories for Broad Semantic Generalization
OBJECT_POOL = [
    ("sphere", "a smooth matte gray sphere on a plain white background, studio lighting"),
    ("cube", "a smooth matte gray cube on a plain white background, studio lighting"),
    ("cylinder", "a smooth matte gray cylinder on a plain white background, studio lighting"),
    ("mug", "a clean ceramic coffee mug on a neutral background, product photo"),
    ("cup", "a simple porcelain cup on a white surface, studio lighting"),
    ("teapot", "a modern ceramic teapot on a white surface, studio lighting"),
    ("vase", "an elegant ceramic vase on a plain white background, studio photo"),
    ("bowl", "a smooth ceramic bowl on a plain white background, studio lighting"),
    ("plate", "a ceramic dinner plate on a plain white background, product photo"),
    ("bottle", "a glass water bottle on a clean white background, studio lighting"),
    ("chair", "a modern wooden dining chair on a plain studio background"),
    ("table", "a minimalist wooden side table on a neutral background"),
    ("lamp", "a contemporary desk lamp on a plain background, studio lighting"),
    ("cushion", "a soft fabric cushion on a plain white background, product photo"),
    ("pillow", "a plush bed pillow on a plain white background, studio lighting"),
    ("backpack", "a sleek urban backpack on a neutral background, product photo"),
    ("handbag", "a stylish leather handbag on a clean studio background"),
    ("shoe", "a classic casual sneaker on a plain white background, product photo"),
    ("boot", "a sturdy leather ankle boot on a neutral studio background"),
    ("hat", "a classic fedora hat on a clean white background, product photo"),
    ("cap", "a sporty baseball cap on a neutral studio background"),
    ("jacket", "a stylish zip-up jacket on a plain white background, product photo"),
    ("sweater", "a cozy knitted sweater on a neutral background, studio photo"),
    ("scarf", "a soft wool scarf neatly folded on a plain white background"),
    ("glove", "a pair of leather gloves on a clean studio background"),
    ("umbrella", "a modern compact umbrella on a plain white background"),
    ("clock", "a minimalist modern wall clock on a plain neutral background"),
    ("watch", "a classic wrist watch on a clean studio background, macro photo"),
    ("headphones", "a pair of sleek over-ear headphones on a plain background"),
    ("speaker", "a compact portable bluetooth speaker on a neutral background"),
    ("camera", "a vintage film camera on a plain white background, studio photo"),
    ("telephone", "a classic rotary telephone on a clean studio background"),
    ("toaster", "a modern stainless steel toaster on a plain white background"),
    ("kettle", "an electric tea kettle on a neutral background, product photo"),
    ("blender", "a modern kitchen blender on a plain white background, studio photo"),
    ("pot", "a ceramic cooking pot with lid on a clean background"),
    ("pan", "a non-stick frying pan on a plain white background, product photo"),
    ("candlestick", "an elegant metal candlestick on a neutral studio background"),
    ("mirror", "a small round tabletop mirror on a plain white background"),
    ("picture_frame", "a simple rectangular picture frame on a neutral studio background"),
    ("book", "a hardcover book with plain cover on a clean white background"),
    ("notebook", "a spiral notebook with plain cover on a neutral background"),
    ("pen", "a sleek metal ballpoint pen on a plain white background, studio photo"),
    ("pencil_holder", "a ceramic desk pencil holder on a neutral background"),
    ("scissors", "a pair of stainless steel scissors on a plain white background"),
    ("stapler", "a modern desktop stapler on a neutral background, studio photo"),
    ("calculator", "a compact desk calculator on a plain white background"),
    ("paperweight", "a smooth glass paperweight on a neutral studio background"),
    ("globe", "a desk educational globe on a plain white background, studio photo"),
    ("hourglass", "a classic sand hourglass on a neutral studio background"),
    ("apple", "a fresh crisp apple on a plain white background, studio lighting"),
    ("banana", "a ripe banana on a neutral studio background, product photo"),
    ("orange", "a fresh whole orange on a plain white background, studio photo"),
    ("pear", "a sweet juicy pear on a clean neutral background, studio photo"),
    ("lemon", "a bright fresh lemon on a plain white background, studio lighting"),
    ("tomato", "a ripe red tomato on a plain white background, studio photo"),
    ("bell_pepper", "a fresh bell pepper on a neutral background, studio lighting"),
    ("carrot", "a single fresh carrot on a plain white background, studio photo"),
    ("eggplant", "a fresh whole eggplant on a clean studio background"),
    ("pumpkin", "a small decorative pumpkin on a plain white background, studio photo"),
    ("car", "a miniature scale model toy car on a plain neutral background"),
    ("truck", "a miniature toy pickup truck on a plain white background"),
    ("bus", "a miniature toy city bus on a clean studio background"),
    ("train", "a miniature toy locomotive on a plain white background"),
    ("airplane", "a small scale model commercial airplane on a neutral background"),
    ("boat", "a miniature wooden sailboat model on a plain white background"),
    ("bicycle", "a classic city commuter bicycle on a plain studio background"),
    ("motorcycle", "a scale model motorcycle on a clean neutral background"),
    ("skateboard", "a modern skateboard deck on a plain white background, studio photo"),
    ("surfboard", "a sleek surfboard standing upright on a neutral background"),
    ("guitar", "an acoustic guitar body on a plain white background, studio photo"),
    ("violin", "a classic wooden violin on a neutral studio background"),
    ("trumpet", "a brass musical trumpet on a plain white background, studio photo"),
    ("flute", "a polished metal flute on a clean neutral background"),
    ("drum", "a small acoustic snare drum on a plain white background, studio photo"),
    ("teddy_bear", "a plush stuffed teddy bear on a clean neutral background"),
    ("doll", "a classic porcelain doll on a plain white background, studio photo"),
    ("robot_toy", "a retro metal toy robot on a neutral studio background"),
    ("wooden_block", "a smooth wooden building block on a plain white background"),
    ("chess_piece", "a carved wooden chess knight piece on a clean studio background"),
    ("football", "a standard leather football on a plain white background, studio photo"),
    ("basketball", "a textured leather basketball on a neutral background"),
    ("tennis_ball", "a standard tennis ball on a plain white background, studio photo"),
    ("baseball", "an official leather baseball on a clean studio background"),
    ("helmet", "a sleek modern bicycle helmet on a plain white background"),
    ("sunglasses", "a stylish pair of sunglasses on a neutral studio background"),
    ("perfume_bottle", "an elegant glass perfume bottle on a plain white background"),
    ("soap_dispenser", "a modern ceramic soap dispenser on a clean studio background"),
    ("toothbrush_holder", "a simple ceramic toothbrush holder on a plain white background"),
    ("towel", "a neatly folded cotton bath towel on a neutral studio background"),
    ("bucket", "a sturdy utility bucket on a plain white background, studio photo"),
    ("watering_can", "a classic gardening watering can on a neutral studio background"),
    ("flowerpot", "a terracotta ceramic flowerpot on a plain white background"),
    ("birdhouse", "a small wooden garden birdhouse on a neutral studio background"),
    ("toolbox", "a compact metal portable toolbox on a plain white background"),
    ("flashlight", "a durable aluminum flashlight on a clean studio background"),
    ("lantern", "a classic metal camping lantern on a plain white background"),
    ("thermos", "an insulated stainless steel thermos on a neutral background"),
    ("lunchbox", "a modern compact lunchbox on a plain white background, studio photo"),
    ("suitcase", "a sleek hard-shell travel suitcase on a neutral studio background"),
]


def load_pca_basis(axes_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(axes_path):
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        print(f"[INFO] PCA axes file not found at {axes_path}. Using standard basis.")
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return u1, u2, u3


def sample_direction_and_magnitude(rng: random.Random,
                                   zone1_range: Tuple[float, float] = ZONE1_RANGE,
                                   zone2_range: Tuple[float, float] = ZONE2_RANGE,
                                   p_zone1: float = P_ZONE1_LINEAR) -> Tuple[float, float, float]:
    """Samples multi-axial 3D unit direction and calibrated magnitude."""
    mode = rng.random()
    if mode < 0.25:
        axis = rng.randint(0, 2)
        sign = rng.choice([-1.0, 1.0])
        d = np.zeros(3, dtype=np.float32)
        d[axis] = sign
    elif mode < 0.50:
        axes = rng.sample([0, 1, 2], 2)
        signs = [rng.choice([-1.0, 1.0]), rng.choice([-1.0, 1.0])]
        d = np.zeros(3, dtype=np.float32)
        d[axes[0]] = signs[0]
        d[axes[1]] = signs[1]
        d = d / np.linalg.norm(d)
    elif mode < 0.75:
        signs = [rng.choice([-1.0, 1.0]) for _ in range(3)]
        d = np.array(signs, dtype=np.float32)
        d = d / np.linalg.norm(d)
    else:
        vec = np.array([rng.gauss(0, 1) for _ in range(3)], dtype=np.float32)
        norm = np.linalg.norm(vec)
        d = vec / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if rng.random() < p_zone1:
        mag = rng.uniform(zone1_range[0], zone1_range[1])
    else:
        mag = rng.uniform(zone2_range[0], zone2_range[1])

    m_vec = d * mag
    return float(m_vec[0]), float(m_vec[1]), float(m_vec[2])


def run_worker(args: argparse.Namespace) -> None:
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\n[WORKER {args.worker_id}] Initializing on {device} ({args.model_type.upper()})...")
    print(f"[WORKER {args.worker_id}] Task Range: Baselines {args.task_start} to {args.task_end} ({args.task_end - args.task_start} baselines x {K_VARIANTS_PER_BASELINE} variants = {(args.task_end - args.task_start) * K_VARIANTS_PER_BASELINE} samples)")

    u1, u2, u3 = load_pca_basis(args.axes_path)

    winning_schedule = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}
    if os.path.exists(args.sched_path):
        try:
            with open(args.sched_path) as f:
                winning_schedule = json.load(f)
        except Exception:
            pass

    gate_frac = winning_schedule["gate_frac"]
    n_partes = winning_schedule["n_partes"]
    perfil_name = winning_schedule["perfil_name"]
    profile_vals = PERFIL_GENERATORS[perfil_name](n_partes)
    bands = build_envelope_bands(gate_frac, profile_vals, transition_mode="ramp_down")

    max_seq_len = MAX_SEQ_LEN_SIGMA if args.model_type == "sigma" else MAX_SEQ_LEN_ALPHA
    pipe, vae = setup_pixart(
        model_type=args.model_type,
        device=device,
        dtype=DTYPE,
        num_latent_channels=NUM_LATENT_CHANNELS,
        use_cpu_offload=False
    )
    seg_models = utils.setup_seg_models(device)

    out_csv = os.path.join(args.out_dir, f"dataset_part_worker_{args.worker_id}.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    completed_baselines = set()
    if os.path.exists(out_csv):
        try:
            prev_df = pd.read_csv(out_csv)
            for _, row in prev_df.iterrows():
                results.append(row.to_dict())
                if "baseline_id" in row:
                    completed_baselines.add(int(row["baseline_id"]))
                elif "sample_id" in row and "_" in str(row["sample_id"]):
                    completed_baselines.add(int(str(row["sample_id"]).split("_")[0]))
            print(f"[WORKER {args.worker_id}] Resumed {len(results)} samples ({len(completed_baselines)} complete baselines) from {out_csv}")
        except Exception as e:
            print(f"[WORKER {args.worker_id}] Note: could not load previous CSV: {e}")

    total_baselines = args.task_end - args.task_start

    for b_idx, baseline_id in enumerate(range(args.task_start, args.task_end)):
        if baseline_id in completed_baselines:
            continue

        obj_name, prompt = OBJECT_POOL[baseline_id % len(OBJECT_POOL)]
        sample_seed = (args.seed + baseline_id * 17) % 1000000

        # 1. Baseline generation (m = 0)
        latents_base = run_generation(
            pipe, prompt, sample_seed, RESOLUTION, RESOLUTION, device, args.steps, args.guidance,
            max_sequence_length=max_seq_len
        )
        img_base = decode_latents_4d(vae, latents_base)

        mask_pixel = utils.get_object_mask(Image.fromarray(img_base), obj_name, seg_models)
        if mask_pixel is None:
            h, w = img_base.shape[:2]
            yy, xx = np.ogrid[:h, :w]
            mask_pixel = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= (min(h, w) * 0.35) ** 2

        lab_base = utils.measure_color_gt(img_base, mask_pixel)
        if lab_base is None:
            lab_base = (50.0, 0.0, 0.0)

        latent_h, latent_w = latent_hw(RESOLUTION, RESOLUTION)
        mask_latent = build_mask_latent(mask_pixel, latent_h, latent_w, device)

        # 2. Generate K variants with continuous PCA latent steering
        rng = random.Random(sample_seed + 1000)
        for var_idx in range(K_VARIANTS_PER_BASELINE):
            m1, m2, m3 = sample_direction_and_magnitude(rng, ZONE1_RANGE, ZONE2_RANGE, P_ZONE1_LINEAR)

            latents_mod = run_generation(
                pipe, prompt, sample_seed, RESOLUTION, RESOLUTION, device, args.steps, args.guidance,
                max_sequence_length=max_seq_len,
                pca_basis=(u1, u2, u3), m_vector=(m1, m2, m3), bands=bands, mask_latent=mask_latent
            )
            img_mod = decode_latents_4d(vae, latents_mod)
            lab_mod = utils.measure_color_gt(img_mod, mask_pixel)
            if lab_mod is None:
                lab_mod = lab_base

            dE00 = float(utils.ciede2000(lab_base, lab_mod))

            results.append({
                "sample_id": f"{baseline_id}_{var_idx}",
                "baseline_id": baseline_id,
                "variant_id": var_idx,
                "object": obj_name,
                "prompt": prompt,
                "seed": sample_seed,
                "base_L": lab_base[0], "base_a": lab_base[1], "base_b": lab_base[2],
                "target_L": lab_mod[0], "target_a": lab_mod[1], "target_b": lab_mod[2],
                "m1": m1, "m2": m2, "m3": m3,
                "deltaE00": dE00,
            })

        completed_baselines.add(baseline_id)

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_baselines:
            pd.DataFrame(results).to_csv(out_csv, index=False)
            done_cnt = len(completed_baselines)
            print(f"[WORKER {args.worker_id}] Progress: {done_cnt}/{total_baselines} baselines done ({len(results)} samples).")
            gc.collect()
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[WORKER {args.worker_id}] Finished! Total saved samples: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Phase C Dataset Collection for PixArt (25k Samples)")
    parser.add_argument("--model-type", choices=["alpha", "sigma"], default="alpha")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--axes-path", type=str, default=None)
    parser.add_argument("--sched-path", type=str, default=None)
    parser.add_argument("--n-baselines", type=int, default=N_BASELINES_TOTAL)
    parser.add_argument("--k-variants", type=int, default=K_VARIANTS_PER_BASELINE)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=N_BASELINES_TOTAL)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.join(base_dir, f"coleccion_datos_mlp_pca_{args.model_type}_out")
    if args.axes_path is None:
        args.axes_path = os.path.join(base_dir, f"fase_a_{args.model_type}_out", "pca_axes.json")
    if args.sched_path is None:
        args.sched_path = os.path.join(base_dir, f"fase_b_{args.model_type}_out", "fase_b_winning_schedule.json")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.worker_id is not None:
        run_worker(args)
        return

    gpus = get_available_gpus()
    print(f"\n" + "="*80)
    print(f"   PIXART ({args.model_type.upper()}) 25,000 DATASET COLLECTOR (FP16 RESIDENT)")
    print(f"   Baselines: {args.n_baselines} | Variants/Baseline: {args.k_variants} | Total: {args.n_baselines * args.k_variants} samples")
    print(f"   Available GPUs ({len(gpus)}): {gpus}")
    print(f"   Output Directory: {args.out_dir}")
    print("="*80 + "\n")

    num_workers = len(gpus) * args.workers_per_gpu
    chunk_size = math.ceil(args.n_baselines / num_workers)

    worker_commands = []
    w_id = 0
    for gpu_idx, gpu_id in enumerate(gpus):
        for _ in range(args.workers_per_gpu):
            task_start = w_id * chunk_size
            task_end = min(task_start + chunk_size, args.n_baselines)
            if task_start >= args.n_baselines:
                break

            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker-id", str(w_id),
                "--gpu", "0",
                "--task-start", str(task_start),
                "--task-end", str(task_end),
                "--model-type", args.model_type,
                "--out-dir", args.out_dir,
                "--axes-path", args.axes_path,
                "--sched-path", args.sched_path,
                "--seed", str(args.seed),
                "--steps", str(args.steps),
                "--guidance", str(args.guidance),
            ]
            worker_commands.append((cmd, gpu_id, w_id))
            w_id += 1

    print(f"[ORCHESTRATOR] Spawning {len(worker_commands)} workers ({args.workers_per_gpu} per GPU) for {args.n_baselines} baselines...")
    processes = []
    for cmd, gpu_id, worker_id in worker_commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        p = subprocess.Popen(cmd, env=env)
        processes.append((p, worker_id))

    for p, worker_id in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[ERROR] Worker {worker_id} exited with returncode {p.returncode}")

    print("[ORCHESTRATOR] All worker processes finished.")

    part_files = sorted([os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.startswith("dataset_part_worker_") and f.endswith(".csv")])
    if part_files:
        dfs = [pd.read_csv(f) for f in part_files if os.path.getsize(f) > 0]
        if dfs:
            master_df = pd.concat(dfs, ignore_index=True)
            master_csv = os.path.join(args.out_dir, "dataset_mlp_pca.csv")
            master_df.to_csv(master_csv, index=False)
            print(f"\n[DATASET] Aggregated {len(master_df)} training pairs into {master_csv}")


if __name__ == "__main__":
    main()

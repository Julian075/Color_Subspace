"""
RUN_GENCOLORBENCH_PCA.PY -- GenColorBench Batch Generator with FLUX + PCA Shift MLP.

Generates benchmark images for GenColorBench (e.g. Numerical Color Precision: ncu_*.csv,
or all benchmark tasks: cna_*, coa_*, ica_*, moc_*, ncu_*) across multiple GPUs & Nodes.

Features:
  - Multi-GPU & Multi-Node Sharding & Parallel Worker Orchestration
  - Atomic Checkpointing & Resume Support
  - Continuous PCA Shift via trained ResMLP_256
  - Exports Generated PNGs + Detailed Generation Manifest CSV (for offline evaluation)
"""

import os
import sys
import gc
import json
import glob
import math
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure current directory is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from flux2_core import (
    build_envelope_bands, envelope_weight,
    latent_hw, unpack_to_4d, pack_from_4d, decode_latents_4d, build_mask_latent,
    PERFIL_GENERATORS, setup_flux,
)
import utils
from iscc_nbs import find_nearest_iscc_l2, find_nearest_iscc_l1
from model_pca import load_mlp_pca

MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 32
RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5

PCA_AXES_PATH = os.path.join(os.path.dirname(__file__), "fase_a_pca_out", "pca_axes.json")
WINNING_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "fase_b_pca_out", "fase_b_winning_schedule.json")
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "mlp_training_out", "mlp_shift_pca_best.pt")

# Load PCA basis vectors
if os.path.exists(PCA_AXES_PATH):
    with open(PCA_AXES_PATH) as f:
        pca_data = json.load(f)["axes"]
    U1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
    U2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
    U3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
else:
    raise FileNotFoundError(f"PCA axes file not found at: {PCA_AXES_PATH}")

# Load winning schedule
if os.path.exists(WINNING_SCHEDULE_PATH):
    with open(WINNING_SCHEDULE_PATH) as f:
        sched_cfg = json.load(f)
else:
    sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

BANDS = build_envelope_bands(
    sched_cfg["gate_frac"],
    PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
    "ramp_down"
)


# =============================================================================
# Helper Utilities & Shift Engine
# =============================================================================
def shift_pca_4d(latents_4d, m1, m2, m3, mask=None):
    out = latents_4d.clone()
    m_mask = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in range(NUM_LATENT_CHANNELS):
        delta_c = float(m1 * U1[c] + m2 * U2[c] + m3 * U3[c])
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


@torch.no_grad()
def generate_gencolorbench_image(pipe, vae, mlp_shift, seg_models, device,
                                 prompt_text, obj_word, target_lab, seed,
                                 height=RESOLUTION, width=RESOLUTION):
    """
    Generates a benchmark image conditioning FLUX on prompt_text and shifting
    the segmented object to target_lab using our trained PCA Shift MLP.
    """
    latent_h, latent_w = latent_hw(pipe, height, width)
    state = {
        "locked": False, "mask_latent": None, "mask_pixel": None,
        "init_lab": None, "m_pred": None, "img_x0": None, "failed": False, "reason": None
    }
    captured = {}

    orig_step = pipe.scheduler.step

    def patched_step(model_output, timestep, sample, *a, **k):
        if not state["locked"]:
            try:
                sched = pipe.scheduler
                idx = sched.index_for_timestep(timestep) if hasattr(sched, "index_for_timestep") else getattr(sched, "_step_index", None)
                sigma = sched.sigmas[idx] if idx is not None else None
                if sigma is not None:
                    captured["pending_x0"] = (sample.detach() - sigma * model_output.detach()).clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def cb(pipe_, step_index, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        frac = step_index / max(STEPS - 1, 1)
        w = envelope_weight(frac, BANDS)

        if not state["locked"]:
            if w == 0.0:
                return callback_kwargs

            pred_x0 = captured.get("pending_x0", lat)
            img_x0 = decode_latents_4d(vae, unpack_to_4d(pipe_, pred_x0, height, width))
            state["img_x0"] = img_x0

            mask_pixel = utils.get_object_mask(Image.fromarray(img_x0), obj_word, seg_models)
            state["locked"] = True

            if mask_pixel is None or mask_pixel.sum() < 50:
                state["failed"] = True
                state["reason"] = "Mask detection failed at gate step"
                return callback_kwargs

            init_lab = utils.measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Color measurement failed at gate step"
                return callback_kwargs

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            # Predict PCA shift vector m = (m1, m2, m3)
            m_pred = mlp_shift.predict_m(init_lab, target_lab)
            state["m_pred"] = m_pred

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return callback_kwargs

        unpacked = unpack_to_4d(pipe_, lat, height, width)
        m = state["mask_latent"]
        if m.shape[-2:] != unpacked.shape[-2:]:
            m = F.interpolate(m, size=unpacked.shape[-2:], mode="nearest")

        m1, m2, m3 = state["m_pred"]
        shifted = shift_pca_4d(unpacked, m1 * w, m2 * w, m3 * w, mask=m)
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents = pipe(
            prompt=prompt_text,
            height=height,
            width=width,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="latent",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
    finally:
        pipe.scheduler.step = orig_step

    if state["failed"]:
        return None, None, None, state["reason"]

    img_final = decode_latents_4d(vae, unpack_to_4d(pipe, latents, height, width))
    return img_final, state["init_lab"], state["m_pred"], None


# =============================================================================
# Defensive Column Parsing
# =============================================================================
def make_get_col(row):
    def get_col(*candidatos, default=None):
        for c in candidatos:
            if c in row.index and pd.notna(row[c]):
                val = str(row[c]).strip()
                if val:
                    return val
        return default
    return get_col


def read_target_color(row, get_col):
    raw = get_col("target_color", "color", "hex")
    if raw is not None:
        return raw
    if all(c in row.index for c in ("r", "g", "b")) and all(pd.notna(row[c]) for c in ("r", "g", "b")):
        return f"rgb({int(float(row['r']))},{int(float(row['g']))},{int(float(row['b']))})"
    return None


# =============================================================================
# Checkpointing
# =============================================================================
def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"completed_csvs": [], "current_csv": None, "current_row": 0, "total_images": 0}


def save_checkpoint(checkpoint_path: Path, state: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, checkpoint_path)


# =============================================================================
# Process a Single Benchmark CSV
# =============================================================================
def process_csv(csv_path: Path, output_dir: Path, pipe, vae, mlp_shift, seg_models,
                images_per_prompt: int, checkpoint_state: Dict[str, Any], checkpoint_path: Path,
                shard_id: int = 0, num_shards: int = 1, device: str = "cuda",
                checkpoint_interval: int = 5) -> None:
    df = pd.read_csv(csv_path)
    csv_name = csv_path.stem
    csv_output_dir = output_dir / csv_name
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = csv_output_dir / f"manifest_shard_{shard_id}.csv"
    manifest_rows = []
    if manifest_path.exists():
        try:
            manifest_rows = pd.read_csv(manifest_path).to_dict(orient="records")
        except Exception:
            manifest_rows = []

    print(f"\n{'='*70}")
    print(f"[{device}] Processing: {csv_path.name}")
    print(f"Total Rows: {len(df)} | Shard: row % {num_shards} == {shard_id}")
    print(f"Output: {csv_output_dir}")
    print(f"{'='*70}")

    for idx in range(len(df)):
        if idx % num_shards != shard_id:
            continue

        row = df.iloc[idx]
        get_col = make_get_col(row)

        prompt_id = get_col("id", "prompt_id", default=str(idx + 1))
        base_seed = int(float(get_col("seed", default="42")))
        obj_word = get_col("object", "obj_word", "category", default="object")
        raw_prompt = get_col("prompt", default=f"a photo of a {obj_word}")

        target_color_raw = read_target_color(row, get_col)
        if target_color_raw is None:
            print(f"  [{device}] [Row {idx}] Skipped: no target color found.")
            continue

        try:
            target_lab = utils.parse_target_color(target_color_raw)
        except Exception as e:
            print(f"  [{device}] [Row {idx}] Color parse error: {e}")
            continue

        # Check existing images
        existing = list(csv_output_dir.glob(f"{prompt_id}_*.png"))
        if len(existing) >= images_per_prompt:
            continue

        # Color-conditioned semantic prompt using official ISCC-NBS Level 2 intermediate hues
        color_name_selected, dist_to_name = find_nearest_iscc_l2(target_lab)
        
        # If the benchmark prompt contains hex/rgb codes, build an enhanced semantic prompt
        if "#" in raw_prompt or "rgb(" in raw_prompt:
            prompt_text = f"a photo of a {color_name_selected} {obj_word}, studio lighting, high quality, 8k"
        else:
            prompt_text = raw_prompt

        print(f"  [{device}] [{idx+1}/{len(df)}] Prompt {prompt_id}: {obj_word} -> {target_color_raw} ('{color_name_selected}' ISCC-L2)")

        saved_count = 0
        for img_idx in range(images_per_prompt):
            img_num = img_idx + 1
            seed = base_seed + img_idx * 1000 + int(prompt_id)
            img_path = csv_output_dir / f"{prompt_id}_{img_num}.png"
            if img_path.exists():
                continue

            try:
                img_final, init_lab, m_pred, err = generate_gencolorbench_image(
                    pipe=pipe, vae=vae, mlp_shift=mlp_shift, seg_models=seg_models,
                    device=device, prompt_text=prompt_text, obj_word=obj_word,
                    target_lab=target_lab, seed=seed
                )

                if img_final is None:
                    print(f"      [Seed {seed}] Generation failed: {err}")
                    continue

                Image.fromarray(img_final).save(img_path)
                saved_count += 1

                manifest_rows.append({
                    "benchmark": csv_name,
                    "prompt_id": prompt_id,
                    "img_num": img_num,
                    "img_path": str(img_path),
                    "prompt_used": prompt_text,
                    "raw_prompt": raw_prompt,
                    "object": obj_word,
                    "target_color_raw": target_color_raw,
                    "color_system": "iscc_l2",
                    "color_name_selected": color_name_selected,
                    "target_L": f"{target_lab[0]:.2f}",
                    "target_a": f"{target_lab[1]:.2f}",
                    "target_b": f"{target_lab[2]:.2f}",
                    "init_L": f"{init_lab[0]:.2f}" if init_lab is not None else "",
                    "init_a": f"{init_lab[1]:.2f}" if init_lab is not None else "",
                    "init_b": f"{init_lab[2]:.2f}" if init_lab is not None else "",
                    "m1": f"{m_pred[0]:.4f}" if m_pred is not None else "",
                    "m2": f"{m_pred[1]:.4f}" if m_pred is not None else "",
                    "m3": f"{m_pred[2]:.4f}" if m_pred is not None else "",
                    "m_norm": f"{np.linalg.norm(m_pred):.4f}" if m_pred is not None else "",
                    "seed": seed,
                })

            except Exception as e:
                print(f"      [Seed {seed}] Unexpected error: {e}")
                traceback.print_exc()
                continue

        checkpoint_state["total_images"] += saved_count
        checkpoint_state["current_csv"] = str(csv_path)
        checkpoint_state["current_row"] = idx + 1
        save_checkpoint(checkpoint_path, checkpoint_state)

        if (idx + 1) % checkpoint_interval == 0:
            if manifest_rows:
                pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            gc.collect()
            torch.cuda.empty_cache()

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)


# =============================================================================
# Run Sharded Generation
# =============================================================================
def run_benchmark_generation(prompts_dir: Path, output_dir: Path, mlp_shift_ckpt: str,
                              device: str = "cuda", images_per_prompt: int = 4,
                              pattern: str = "ncu_*.csv", shard_id: int = 0, num_shards: int = 1) -> None:
    prompts_dir = Path(prompts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_shard_{shard_id}_of_{num_shards}.json"
    checkpoint_state = load_checkpoint(checkpoint_path)

    if prompts_dir.is_file() and prompts_dir.suffix == ".csv":
        csv_files = [prompts_dir]
    else:
        csv_files = sorted(prompts_dir.glob(pattern))

    if not csv_files:
        print(f"No CSVs found in: {prompts_dir} matching pattern '{pattern}'")
        return

    print(f"Found {len(csv_files)} CSV files matching '{pattern}':")
    for f in csv_files:
        print(f"  - {f.name}")

    pipe, vae = setup_flux(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models("cpu")
    mlp_shift = load_mlp_pca(mlp_shift_ckpt, device)

    for csv_path in csv_files:
        process_csv(
            csv_path=csv_path, output_dir=output_dir, pipe=pipe, vae=vae,
            mlp_shift=mlp_shift, seg_models=seg_models,
            images_per_prompt=images_per_prompt, checkpoint_state=checkpoint_state,
            checkpoint_path=checkpoint_path, shard_id=shard_id, num_shards=num_shards,
            device=device
        )

    del pipe, vae, seg_models, mlp_shift
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Auto Multi-GPU Launcher (Supports Single-Node or Multi-Node Offsets)
# =============================================================================
def launch_auto_multi_gpu(args: argparse.Namespace) -> None:
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        worker_gpus = [x.strip() for x in parent_cvd.split(",") if x.strip()]
    else:
        total = torch.cuda.device_count()
        worker_gpus = [str(i) for i in range(total)]

    num_local_workers = len(worker_gpus)
    total_shards = args.total_shards if args.total_shards > 0 else num_local_workers
    shard_offset = args.shard_offset

    print(f"\n{'='*70}")
    print(f">>> LAUNCHING GENCOLORBENCH MULTI-GPU GENERATOR")
    print(f"    Local Worker GPUs ({num_local_workers}): {worker_gpus}")
    print(f"    Shard Range: Shards {shard_offset} to {shard_offset + num_local_workers - 1} (of {total_shards} total)")
    print(f"    Prompts Dir: {args.prompts_dir}")
    print(f"    Pattern:     {args.pattern}")
    print(f"    Output Dir:  {args.output_dir}")
    print(f"    Checkpoint:  {args.mlp_shift_ckpt}")
    print(f"{'='*70}\n")

    processes: List[subprocess.Popen] = []
    for local_idx, gpu_id in enumerate(worker_gpus):
        global_shard_id = shard_offset + local_idx
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--prompts-dir", args.prompts_dir,
            "--output-dir", args.output_dir,
            "--mlp-shift-ckpt", args.mlp_shift_ckpt,
            "--images-per-prompt", str(args.images_per_prompt),
            "--num-shards", str(total_shards),
            "--shard-id", str(global_shard_id),
            "--pattern", args.pattern,
            "--device", "cuda:0"
        ]

        log_path = Path(args.output_dir) / f"worker_shard_{global_shard_id}_gpu_{gpu_id}.log"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Launching Worker for Shard {global_shard_id}/{total_shards} on Local GPU {gpu_id} -> {log_path.name}")
        log_file = open(log_path, "w")
        processes.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))

    print("\nAll local workers running. Waiting for completion...")
    for idx, p in enumerate(processes):
        p.wait()
        global_shard_id = shard_offset + idx
        print(f"Worker for Shard {global_shard_id} (GPU {worker_gpus[idx]}) exited with code {p.returncode}.")
    print("\n>>> Local Multi-GPU Batch Finished!")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GenColorBench Generation with FLUX PCA MLP")
    parser.add_argument("--prompts-dir", type=str, default="/leonardo_work/AIFAC_S07_004/jsantamaria/projects/colorspace/gencolorbench/mini_bench_prompts")
    parser.add_argument("--output-dir", type=str, default="./gencolorbench_flux_pca_out")
    parser.add_argument("--mlp-shift-ckpt", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--pattern", type=str, default="ncu_*.csv", help="Glob pattern for benchmark tasks (e.g. 'ncu_*.csv' or '*.csv')")
    parser.add_argument("--images-per-prompt", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auto-multi-gpu", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-offset", type=int, default=0, help="Offset for global shard ID in multi-node runs")
    parser.add_argument("--total-shards", type=int, default=0, help="Total number of shards across all nodes (0 = autodetect local)")
    args = parser.parse_args()

    if args.auto_multi_gpu:
        launch_auto_multi_gpu(args)
    else:
        run_benchmark_generation(
            prompts_dir=Path(args.prompts_dir),
            output_dir=Path(args.output_dir),
            mlp_shift_ckpt=args.mlp_shift_ckpt,
            device=args.device,
            images_per_prompt=args.images_per_prompt,
            pattern=args.pattern,
            shard_id=args.shard_id,
            num_shards=args.num_shards
        )

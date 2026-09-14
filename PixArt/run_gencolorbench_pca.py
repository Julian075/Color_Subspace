"""
RUN_GENCOLORBENCH_PCA.PY -- GenColorBench Batch Generator with PixArt + PCA Shift MLP.

Generates benchmark images for GenColorBench (e.g. Numerical Color Precision: ncu_*.csv,
or all benchmark tasks: cna_*, coa_*, ica_*, moc_*, ncu_*) on PixArt (PixArt-alpha & PixArt-Sigma).

Features:
  - Multi-GPU Sharding & Parallel Worker Orchestration
  - Atomic Checkpointing & Resume Support
  - Continuous PCA Shift via trained ResMLP_256
  - Exports Generated PNGs ({prompt_id}_{img_num}.png) + Detailed Generation Manifest CSV
  - Uses ISCC-NBS Level 2 color baseline naming in prompts for all numeric color tasks
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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Local imports
from pixart_core import (
    MODEL_ID_ALPHA, MODEL_ID_SIGMA, DTYPE, NUM_LATENT_CHANNELS, RESOLUTION,
    STEPS, GUIDANCE, MAX_SEQ_LEN_ALPHA, MAX_SEQ_LEN_SIGMA,
    setup_pixart, decode_latents_4d, build_envelope_bands, envelope_weight, build_mask_latent,
    latent_hw, run_generation, PERFIL_GENERATORS,
)
import utils
from iscc_nbs import find_nearest_iscc_l2, get_iscc_l2_centroid
from model_pca import load_mlp_pca

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pca_basis(axes_path: str):
    if os.path.exists(axes_path):
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        print(f"[INFO] PCA axes file not found at {axes_path}. Using placeholder vectors.")
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return u1, u2, u3


def load_winning_schedule(sched_path: str):
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            return json.load(f)
    return {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_csv": "", "current_row": 0, "total_images": 0}


def save_checkpoint(checkpoint_path: Path, state: Dict[str, Any]) -> None:
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(state, f, indent=2)
    temp_path.replace(checkpoint_path)


def make_get_col(row: pd.Series):
    def get_col(*candidates: str, default: str = "") -> str:
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return str(row[c]).strip()
        return default
    return get_col


def parse_target_lab(row: pd.Series) -> Tuple[float, float, float]:
    """Extracts CIELAB target color using ISCC-NBS Level 2 as canonical basis."""
    if "target_L" in row and pd.notna(row["target_L"]):
        return (float(row["target_L"]), float(row["target_a"]), float(row["target_b"]))
    elif "r" in row and "g" in row and "b" in row and pd.notna(row["r"]):
        return utils.rgb_to_lab_single_np((int(float(row["r"])), int(float(row["g"])), int(float(row["b"]))))
    elif "target_hex" in row and pd.notna(row["target_hex"]):
        return utils.rgb_to_lab_single_np(utils.hex_to_rgb(str(row["target_hex"])))
    elif "hex" in row and pd.notna(row["hex"]):
        return utils.rgb_to_lab_single_np(utils.hex_to_rgb(str(row["hex"])))
    for col in ["color_name", "color", "color_word", "target_color"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            try:
                return get_iscc_l2_centroid(str(row[col]))
            except Exception:
                pass
    return (50.0, 0.0, 0.0)


@torch.no_grad()
def generate_gencolorbench_image(pipe, vae, mlp_shift, seg_models, device,
                                 prompt_text, obj_word, target_lab, seed,
                                 u1, u2, u3, bands,
                                 height=RESOLUTION, width=RESOLUTION,
                                 steps=STEPS, guidance=GUIDANCE,
                                 max_seq_len=MAX_SEQ_LEN_ALPHA):
    """
    Single-pass inference: ONE continuous denoising trajectory.
    - NO separate pre-generation baseline pass.
    - At the gate step (first step where envelope weight w > 0), x̂₀ is predicted
      via DDPM epsilon-prediction: x̂₀ = (xₜ - √(1-ᾱₜ)·ε) / √ᾱₜ
      (using pipe.scheduler.alphas_cumprod). Decoded once with VAE, segmented
      once with SAM. MLP predicts (m1,m2,m3). State is then locked.
    - Remaining steps apply PCA latent shift smoothly via the envelope weight.
    Gate: Alpha gate_frac=0.5 (step ~10/20), Sigma gate_frac=0.75 (step ~15/20).
    """
    latent_h, latent_w = latent_hw(height, width)
    generator = torch.Generator(device=device).manual_seed(seed)

    state = {
        "locked": False,
        "mask_latent": None,
        "mask_pixel": None,
        "init_lab": None,
        "m_pred": None,
        "failed": False,
        "reason": None,
    }
    captured = {}

    orig_step = pipe.scheduler.step

    def patched_step(model_output, timestep, sample, *a, **k):
        if not state["locked"]:
            try:
                sched = pipe.scheduler
                # DDPM: x̂₀ = (xₜ - √(1-ᾱₜ)·ε) / √ᾱₜ
                t = int(timestep)
                alpha_prod_t = sched.alphas_cumprod[t].to(sample.device)
                sqrt_alpha = alpha_prod_t ** 0.5
                sqrt_one_minus_alpha = (1.0 - alpha_prod_t) ** 0.5
                pred_x0 = (sample.detach() - sqrt_one_minus_alpha * model_output.detach()) / sqrt_alpha
                captured["pending_x0"] = pred_x0.clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def callback_fn(step_index, timestep, latents):
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)

        if not state["locked"]:
            if w == 0.0:
                return

            pred_x0 = captured.get("pending_x0", latents)
            img_x0 = decode_latents_4d(vae, pred_x0)

            mask_pixel = utils.get_object_mask(Image.fromarray(img_x0), obj_word, seg_models)
            state["locked"] = True

            if mask_pixel is None or mask_pixel.sum() < 50:
                h_img, w_img = img_x0.shape[:2]
                yy, xx = np.ogrid[:h_img, :w_img]
                mask_pixel = ((xx - w_img / 2) ** 2 + (yy - h_img / 2) ** 2) <= (min(h_img, w_img) * 0.35) ** 2

            init_lab = utils.measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Color measurement failed at gate step"
                return

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            m1, m2, m3 = mlp_shift(init_lab, target_lab)
            state["m_pred"] = (m1, m2, m3)

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return

        m1, m2, m3 = state["m_pred"]
        m_mask = state["mask_latent"][:, 0].to(latents.dtype)
        shifted = latents.clone()
        for c in range(NUM_LATENT_CHANNELS):
            delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c]) * w
            shifted[:, c] += delta_c * m_mask
        latents.copy_(shifted)

    pipe.scheduler.step = patched_step
    try:
        latents = pipe(
            prompt_text,
            height=height, width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=max_seq_len,
            clean_caption=False,
            generator=generator,
            output_type="latent",
            callback=callback_fn,
            callback_steps=1,
        ).images
    finally:
        pipe.scheduler.step = orig_step

    if state["failed"]:
        return None, None, None, (0.0, 0.0, 0.0)

    img_final = decode_latents_4d(vae, latents)
    final_lab = utils.measure_color_gt(img_final, state["mask_pixel"]) if state["mask_pixel"] is not None else None

    return img_final, state["init_lab"], final_lab, state["m_pred"]


def process_csv(csv_path: Path, output_dir: Path, pipe, vae, mlp_shift, seg_models,
                u1, u2, u3, bands, max_seq_len: int,
                images_per_prompt: int, checkpoint_state: Dict[str, Any],
                checkpoint_path: Path, shard_id: int, num_shards: int,
                device: str, checkpoint_interval: int = 10):
    csv_name = csv_path.stem
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[{device}] Error reading CSV {csv_path}: {e}")
        return

    csv_output_dir = output_dir / csv_name
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = csv_output_dir / f"manifest_shard_{shard_id}.csv"
    manifest_rows = []
    if manifest_path.exists():
        try:
            manifest_rows = pd.read_csv(manifest_path).to_dict(orient="records")
        except Exception:
            manifest_rows = []

    print(f"\n" + "="*70)
    print(f"[{device}] Processing: {csv_path.name}")
    print(f"Total Rows: {len(df)} | Shard: row % {num_shards} == {shard_id}")
    print(f"Output: {csv_output_dir}")
    print("="*70)

    for idx in range(len(df)):
        if idx % num_shards != shard_id:
            continue

        row = df.iloc[idx]
        get_col = make_get_col(row)

        prompt_id = get_col("id", "prompt_id", default=str(idx + 1))
        base_seed = int(float(get_col("seed", default="42")))
        obj_word = get_col("object", "obj_word", "category", default="object")
        raw_prompt = get_col("prompt", default=f"a photo of a {obj_word}")

        target_lab = parse_target_lab(row)

        existing = list(csv_output_dir.glob(f"{prompt_id}_*.png"))
        if len(existing) >= images_per_prompt:
            continue

        color_name_selected, dist_to_name = find_nearest_iscc_l2(target_lab)
        color_name_clean = color_name_selected.replace("_", " ")

        if "#" in raw_prompt or "rgb(" in raw_prompt or "hex" in raw_prompt.lower():
            prompt_text = f"a photo of a {color_name_clean} {obj_word}, studio lighting, high quality, 8k"
        else:
            prompt_text = raw_prompt

        print(f"  [{device}] [{idx+1}/{len(df)}] Prompt {prompt_id}: {obj_word} -> Lab({target_lab[0]:.1f},{target_lab[1]:.1f},{target_lab[2]:.1f}) (ISCC-NBS L2: '{color_name_clean}')")

        saved_count = 0
        for img_idx in range(images_per_prompt):
            img_num = img_idx + 1
            seed = base_seed + img_idx * 1000 + int(prompt_id)
            img_path = csv_output_dir / f"{prompt_id}_{img_num}.png"
            if img_path.exists():
                continue

            try:
                img_final, init_lab, final_lab, m_pred = generate_gencolorbench_image(
                    pipe=pipe, vae=vae, mlp_shift=mlp_shift, seg_models=seg_models,
                    device=device, prompt_text=prompt_text, obj_word=obj_word,
                    target_lab=target_lab, seed=seed,
                    u1=u1, u2=u2, u3=u3, bands=bands, max_seq_len=max_seq_len
                )

                Image.fromarray(img_final).save(img_path)
                saved_count += 1

                dE_final = float(utils.ciede2000(final_lab if final_lab is not None else target_lab, target_lab))
                manifest_rows.append({
                    "benchmark": csv_name,
                    "prompt_id": prompt_id,
                    "img_num": img_num,
                    "img_path": str(img_path),
                    "prompt_used": prompt_text,
                    "raw_prompt": raw_prompt,
                    "object": obj_word,
                    "target_color_raw": f"rgb({row.get('r', '')},{row.get('g', '')},{row.get('b', '')})" if "r" in row else str(row.get("target_hex", "")),
                    "color_name_selected": color_name_clean,
                    "target_L": f"{target_lab[0]:.2f}",
                    "target_a": f"{target_lab[1]:.2f}",
                    "target_b": f"{target_lab[2]:.2f}",
                    "init_L": f"{init_lab[0]:.2f}" if init_lab is not None else "",
                    "init_a": f"{init_lab[1]:.2f}" if init_lab is not None else "",
                    "init_b": f"{init_lab[2]:.2f}" if init_lab is not None else "",
                    "final_L": f"{final_lab[0]:.2f}" if final_lab is not None else "",
                    "final_a": f"{final_lab[1]:.2f}" if final_lab is not None else "",
                    "final_b": f"{final_lab[2]:.2f}" if final_lab is not None else "",
                    "deltaE00": f"{dE_final:.2f}",
                    "m1": f"{m_pred[0]:.4f}" if m_pred is not None else "",
                    "m2": f"{m_pred[1]:.4f}" if m_pred is not None else "",
                    "m3": f"{m_pred[2]:.4f}" if m_pred is not None else "",
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


def run_benchmark_generation(prompts_dir: Path, output_dir: Path, model_type: str,
                             mlp_shift_ckpt: str, axes_path: str, sched_path: str,
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

    max_seq_len = MAX_SEQ_LEN_SIGMA if model_type == "sigma" else MAX_SEQ_LEN_ALPHA
    u1, u2, u3 = load_pca_basis(axes_path)
    sched = load_winning_schedule(sched_path)
    bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")

    pipe, vae = setup_pixart(model_type=model_type, device=device, dtype=DTYPE, num_latent_channels=NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)
    mlp_shift = load_mlp_pca(mlp_shift_ckpt, device)

    for csv_path in csv_files:
        process_csv(
            csv_path=csv_path, output_dir=output_dir, pipe=pipe, vae=vae,
            mlp_shift=mlp_shift, seg_models=seg_models,
            u1=u1, u2=u2, u3=u3, bands=bands, max_seq_len=max_seq_len,
            images_per_prompt=images_per_prompt, checkpoint_state=checkpoint_state,
            checkpoint_path=checkpoint_path, shard_id=shard_id, num_shards=num_shards,
            device=device
        )

    del pipe, vae, seg_models, mlp_shift
    gc.collect()
    torch.cuda.empty_cache()


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

    print(f"\n" + "="*70)
    print(f">>> LAUNCHING PIXART ({args.model_type.upper()}) GENCOLORBENCH MULTI-GPU GENERATOR")
    print(f"    Local Worker GPUs ({num_local_workers}): {worker_gpus}")
    print(f"    Shard Range: Shards {shard_offset} to {shard_offset + num_local_workers - 1} (of {total_shards} total)")
    print(f"    Prompts Dir: {args.prompts_dir}")
    print(f"    Pattern:     {args.pattern}")
    print(f"    Output Dir:  {args.output_dir}")
    print(f"    Checkpoint:  {args.mlp_shift_ckpt}")
    print("="*70 + "\n")

    processes: List[subprocess.Popen] = []
    for local_idx, gpu_id in enumerate(worker_gpus):
        global_shard_id = shard_offset + local_idx
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--model-type", args.model_type,
            "--prompts-dir", str(args.prompts_dir),
            "--output-dir", str(args.output_dir),
            "--mlp-shift-ckpt", str(args.mlp_shift_ckpt),
            "--axes-path", str(args.axes_path),
            "--sched-path", str(args.sched_path),
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
    print(f"\n>>> PixArt {args.model_type.upper()} Multi-GPU Generation Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenColorBench Runner for PixArt PCA MLP")
    parser.add_argument("--model-type", choices=["alpha", "sigma"], default="alpha")
    parser.add_argument("--prompts-dir", type=str, default="/data/140-1/users/jsantamaria/vae_exploration/gencolorbench/mini_bench_prompt")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mlp-shift-ckpt", type=str, default=None)
    parser.add_argument("--axes-path", type=str, default=None)
    parser.add_argument("--sched-path", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="ncu_*.csv", help="Glob pattern for benchmark tasks")
    parser.add_argument("--images-per-prompt", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auto-multi-gpu", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=0)
    args = parser.parse_args()

    if args.axes_path is None:
        args.axes_path = os.path.join(BASE_DIR, f"fase_a_{args.model_type}_out", "pca_axes.json")
    if args.sched_path is None:
        args.sched_path = os.path.join(BASE_DIR, f"fase_b_{args.model_type}_out", "fase_b_winning_schedule.json")
    if args.mlp_shift_ckpt is None:
        args.mlp_shift_ckpt = os.path.join(BASE_DIR, f"mlp_training_{args.model_type}_out", "mlp_shift_pca_best.pt")
    if args.output_dir is None:
        args.output_dir = f"/data/140-1/users/jsantamaria/vae_exploration/results_paper/pixart_{args.model_type}/gencolorbench_out"

    if args.auto_multi_gpu:
        launch_auto_multi_gpu(args)
    else:
        run_benchmark_generation(
            prompts_dir=Path(args.prompts_dir),
            output_dir=Path(args.output_dir),
            model_type=args.model_type,
            mlp_shift_ckpt=args.mlp_shift_ckpt,
            axes_path=args.axes_path,
            sched_path=args.sched_path,
            device=args.device,
            images_per_prompt=args.images_per_prompt,
            pattern=args.pattern,
            shard_id=args.shard_id,
            num_shards=args.num_shards
        )

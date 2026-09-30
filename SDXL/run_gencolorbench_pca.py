"""
RUN_GENCOLORBENCH_PCA.PY -- GenColorBench Single-Forward In-Flight Evaluation for SDXL.

Methodology:
  - Single Forward Pass: ONE continuous 30-step diffusion trajectory per image.
  - In-Flight Gate: At gate_frac (step ~12 of 30, when envelope opens), x̂₀ is captured
    from the denoiser / scheduler step:
      x̂₀ = sample - σ · model_output (Euler/DDIM),
    decoded once with SDXL VAE, and segmented with SAM-3 to obtain the object mask and initial Lab color.
  - Closed-Loop Steering: The trained MLP predicts (m1, m2, m3) from (init_lab, target_lab).
  - Steering is applied in-place during the remaining steps of the SAME generation trajectory.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
from sdxl_core import (
    build_envelope_bands, envelope_weight,
    latent_hw, decode_latents_4d, build_mask_latent,
    PERFIL_GENERATORS, setup_sdxl, get_gpu_list, spawn_workers,
)
import utils
from model_pca import load_mlp_pca

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4
RESOLUTION = 1024
STEPS = 30
GUIDANCE = 5.0

PCA_AXES_PATH = os.path.join(os.path.dirname(__file__), "fase_a_pca_out", "pca_axes.json")
WINNING_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "fase_b_pca_out", "fase_b_winning_schedule.json")
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "mlp_training_out", "mlp_shift_pca_best.pt")


def load_pca_basis(axes_path=PCA_AXES_PATH):
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


def load_winning_schedule(sched_path=WINNING_SCHEDULE_PATH):
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            return json.load(f)
    return {"gate_frac": 0.40, "n_partes": 1, "perfil_name": "plano"}


def parse_target_lab(row: pd.Series) -> Tuple[float, float, float]:
    """Robustly extracts CIELAB target color."""
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
                from iscc_nbs import get_iscc_l2_centroid
                return get_iscc_l2_centroid(str(row[col]))
            except Exception:
                pass
    return (50.0, 0.0, 0.0)


def parse_object_word(row: pd.Series) -> str:
    """Robustly extracts target object name."""
    for col in ["target_object", "obj_word", "object", "main_obj", "object1", "obj_name", "category"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "object"


@torch.no_grad()
def generate_gencolorbench_single_forward(
    pipe, vae, mlp_shift, seg_models, device,
    prompt_text: str, obj_word: str, target_lab: Tuple[float, float, float], seed: int,
    u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, bands: list,
    height: int = RESOLUTION, width: int = RESOLUTION
) -> Dict[str, Any]:
    """
    Single-Forward Closed-Loop Inference:
    - Exactly ONE continuous 30-step diffusion trajectory.
    - Captures predicted clean x̂₀ at gate step.
    - Segments with SAM-3 once, predicts MLP shift m*, and steers in-place.
    """
    latent_h, latent_w = latent_hw(height, width)

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
                # EulerDiscreteScheduler: x̂₀ = sample - σ · model_output
                if hasattr(sched, "sigmas"):
                    idx = getattr(sched, "step_index", getattr(sched, "_step_index", None))
                    if idx is None and hasattr(sched, "index_for_timestep"):
                        idx = sched.index_for_timestep(timestep)
                    if idx is not None and idx < len(sched.sigmas):
                        sigma = sched.sigmas[idx].to(sample.device)
                        captured["pending_x0"] = (sample.detach() - sigma * model_output.detach()).clone()
                elif hasattr(sched, "alphas_cumprod"):
                    t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
                    alpha_prod_t = sched.alphas_cumprod[t].to(sample.device)
                    sqrt_alpha = alpha_prod_t ** 0.5
                    sqrt_one_minus_alpha = (1.0 - alpha_prod_t) ** 0.5
                    captured["pending_x0"] = ((sample.detach() - sqrt_one_minus_alpha * model_output.detach()) / sqrt_alpha).clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def cb(pipe_, step_index, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        frac = step_index / max(STEPS - 1, 1)
        w = envelope_weight(frac, bands)

        if not state["locked"]:
            if w == 0.0:
                return callback_kwargs

            pred_x0 = captured.get("pending_x0", lat)
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
                return callback_kwargs

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            m1, m2, m3 = mlp_shift(init_lab, target_lab)
            state["m_pred"] = (m1, m2, m3)

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return callback_kwargs

        m1, m2, m3 = state["m_pred"]
        m_mask = state["mask_latent"][:, 0].to(lat.dtype)
        for c in range(NUM_LATENT_CHANNELS):
            delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c]) * w
            lat[:, c] += delta_c * m_mask
        callback_kwargs["latents"] = lat
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents = pipe(
            prompt_text,
            height=height, width=width,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
    finally:
        pipe.scheduler.step = orig_step

    img_final = decode_latents_4d(vae, latents)
    if state["failed"] or state["init_lab"] is None:
        return {
            "image": Image.fromarray(img_final),
            "base_lab": (50.0, 0.0, 0.0),
            "final_lab": (50.0, 0.0, 0.0),
            "m_pred": (0.0, 0.0, 0.0),
        }

    final_lab = utils.measure_color_gt(img_final, state["mask_pixel"]) if state["mask_pixel"] is not None else None

    return {
        "image": Image.fromarray(img_final),
        "base_lab": state["init_lab"],
        "final_lab": final_lab if final_lab is not None else state["init_lab"],
        "m_pred": state["m_pred"],
    }


def merge_manifests(out_dir: str):
    for task_dir in Path(out_dir).iterdir():
        if not task_dir.is_dir():
            continue
        all_csvs = sorted(set(list(task_dir.glob("manifest_shard_*.csv")) + list(task_dir.glob("manifest*.csv"))))
        if all_csvs:
            dfs = []
            for sf in all_csvs:
                try:
                    df_ = pd.read_csv(sf)
                    if len(df_) > 0:
                        dfs.append(df_)
                except Exception:
                    pass
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                if "task_id" in merged.columns and "seed_idx" in merged.columns:
                    merged["_tid"] = pd.to_numeric(merged["task_id"], errors="coerce")
                    merged["_sidx"] = pd.to_numeric(merged["seed_idx"], errors="coerce")
                    merged = merged.sort_values(by=["_tid", "_sidx"]).drop_duplicates(subset=["_tid", "_sidx"]).drop(columns=["_tid", "_sidx"])
                merged_path = task_dir / "manifest.csv"
                merged.to_csv(merged_path, index=False)
                print(f"[MERGE] Merged {len(dfs)} manifest files into {merged_path} ({len(merged)} rows)", flush=True)


def launch_auto_multi_gpu(args: argparse.Namespace) -> None:
    if args.gpus:
        worker_gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    else:
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_cvd:
            worker_gpus = [x.strip() for x in parent_cvd.split(",") if x.strip()]
        else:
            total = torch.cuda.device_count()
            worker_gpus = [str(i) for i in range(total)]

    num_local_workers = len(worker_gpus)
    print(f"\n" + "="*70)
    print(f">>> LAUNCHING SDXL MULTI-GPU GENCOLORBENCH RUNNER")
    print(f"    Worker GPUs ({num_local_workers}): {worker_gpus}")
    print(f"    Task:       {args.task}")
    print(f"    Out Dir:    {args.out_dir}")
    print("="*70 + "\n", flush=True)

    processes: List[subprocess.Popen] = []
    for local_idx, gpu_id in enumerate(worker_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            sys.executable, "-u", os.path.abspath(__file__),
            "--prompt-dir", str(args.prompt_dir),
            "--task", str(args.task),
            "--out-dir", str(args.out_dir),
            "--ckpt-path", str(args.ckpt_path),
            "--num-shards", str(num_local_workers),
            "--shard-id", str(local_idx),
            "--gpu", "0",
            "--seeds", *[str(s) for s in args.seeds]
        ]
        if args.benchmark_csv:
            cmd.extend(["--benchmark-csv", str(args.benchmark_csv)])

        log_path = Path(args.out_dir) / f"worker_gpu_{gpu_id}_shard_{local_idx}.log"
        print(f"Launching Worker for Shard {local_idx}/{num_local_workers} on GPU {gpu_id} -> {log_path.name}", flush=True)
        log_file = open(log_path, "w", buffering=1)
        processes.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))

    for idx, p in enumerate(processes):
        p.wait()
        print(f"Worker {idx} on GPU {worker_gpus[idx]} exited with code {p.returncode}.", flush=True)

    merge_manifests(args.out_dir)
    print(f"\n>>> Multi-GPU Generation Finished & Manifests Merged!", flush=True)


def run_benchmark_task(csv_path: str, out_dir: str, ckpt_path: str,
                       gpu: int = 0, seeds: List[int] = [1, 2, 3, 4],
                       shard_id: int = 0, num_shards: int = 1,
                       pipe=None, vae=None, mlp_shift=None, seg_models=None,
                       u1=None, u2=None, u3=None, bands=None):
    task_name = Path(csv_path).stem
    task_out_dir = os.path.join(out_dir, task_name)
    os.makedirs(task_out_dir, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if pipe is None or vae is None or mlp_shift is None or seg_models is None or u1 is None or bands is None:
        print(f"\n[{task_name.upper()}] Loading SDXL models on {device} (Shard {shard_id}/{num_shards})...", flush=True)
        u1, u2, u3 = load_pca_basis()
        sched = load_winning_schedule()
        bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")
        mlp_shift = load_mlp_pca(ckpt_path, device=device)
        pipe, vae = setup_sdxl(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
        seg_models = utils.setup_seg_models(device)

    df = pd.read_csv(csv_path)
    total_in_shard = sum(1 for idx in range(len(df)) if (num_shards <= 1 or (idx % num_shards) == shard_id))
    print(f"[{task_name}] Loaded {len(df)} prompts from {csv_path}. Shard {shard_id}/{num_shards} processing {total_in_shard} prompts ({total_in_shard * len(seeds)} images)...")

    manifest_csv = os.path.join(task_out_dir, f"manifest_shard_{shard_id}.csv" if num_shards > 1 else "manifest.csv")
    results = []
    if os.path.exists(manifest_csv):
        try:
            results = pd.read_csv(manifest_csv).to_dict("records")
        except Exception:
            results = []

    done_keys = set()
    for r in results:
        if "task_id" in r and "seed_idx" in r:
            try:
                done_keys.add(f"{int(float(r['task_id']))}_{int(float(r['seed_idx']))}")
            except Exception:
                done_keys.add(f"{str(r['task_id']).strip()}_{str(r['seed_idx']).strip()}")

    # Check all existing shard manifests and main manifest
    all_manifest_files = list(Path(task_out_dir).glob("manifest_shard_*.csv")) + list(Path(task_out_dir).glob("manifest*.csv"))
    for mf in all_manifest_files:
        if mf.exists():
            try:
                for _, r in pd.read_csv(mf).iterrows():
                    if "task_id" in r and "seed_idx" in r:
                        try:
                            done_keys.add(f"{int(float(r['task_id']))}_{int(float(r['seed_idx']))}")
                        except Exception:
                            done_keys.add(f"{str(r['task_id']).strip()}_{str(r['seed_idx']).strip()}")
            except Exception:
                pass

    shard_idx = 0
    for idx, row in df.iterrows():
        if num_shards > 1 and (idx % num_shards) != shard_id:
            continue
        shard_idx += 1

        task_id = row.get("id", idx + 1)
        raw_prompt = str(row.get("prompt", row.get("text", "")))
        obj_word = parse_object_word(row)
        target_lab = parse_target_lab(row)

        from iscc_nbs import find_nearest_iscc_l2
        color_name_selected, _ = find_nearest_iscc_l2(target_lab)
        color_name_clean = color_name_selected.replace("_", " ")

        if "#" in raw_prompt or "rgb(" in raw_prompt or "hex" in raw_prompt.lower():
            prompt_text = f"a photo of a {color_name_clean} {obj_word}, studio lighting, high quality, 8k"
        else:
            prompt_text = raw_prompt

        for seed_idx, s in enumerate(seeds, start=1):
            try:
                key = f"{int(float(task_id))}_{int(float(seed_idx))}"
            except Exception:
                key = f"{str(task_id).strip()}_{str(seed_idx).strip()}"
            img_filename = f"{task_id}_{seed_idx}.png"
            img_path = os.path.join(task_out_dir, img_filename)

            if key in done_keys and os.path.exists(img_path):
                continue

            seed_val = int(s * 1000 + int(task_id) * 17) % 1000000

            res = generate_gencolorbench_single_forward(
                pipe, vae, mlp_shift, seg_models, device,
                prompt_text, obj_word, target_lab, seed_val,
                u1, u2, u3, bands
            )
            res["image"].save(img_path)

            dE = float(utils.ciede2000(res["final_lab"], target_lab))
            record = {
                "task_id": task_id,
                "seed_idx": seed_idx,
                "seed": seed_val,
                "prompt": prompt_text,
                "raw_prompt": raw_prompt,
                "object": obj_word,
                "target_L": target_lab[0], "target_a": target_lab[1], "target_b": target_lab[2],
                "base_L": res["base_lab"][0], "base_a": res["base_lab"][1], "base_b": res["base_lab"][2],
                "final_L": res["final_lab"][0], "final_a": res["final_lab"][1], "final_b": res["final_lab"][2],
                "deltaE_final": dE,
                "m1": res["m_pred"][0], "m2": res["m_pred"][1], "m3": res["m_pred"][2],
                "image_filename": img_filename
            }
            results.append(record)
            done_keys.add(key)
            del res

            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(manifest_csv, index=False)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if (shard_idx % 10 == 0) or (shard_idx == total_in_shard):
            print(f"[{task_name}][Shard {shard_id}] Progress: {shard_idx}/{total_in_shard} prompts completed.", flush=True)

    pd.DataFrame(results).to_csv(manifest_csv, index=False)
    print(f"\n>>> [{task_name}][Shard {shard_id}] Complete. Saved manifest to: {manifest_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GenColorBench Single-Forward In-Flight Evaluation for SDXL")
    parser.add_argument("--prompt-dir", default="./gencolorbench/mini_bench_prompts")
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--task", default="ncu", choices=["ncu", "all", "cna", "coa", "ica", "moc", "iscc_l2"])
    parser.add_argument("--out-dir", default="./gencolorbench_sdxl_out")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g. '0,1,2,3,4,5')")
    parser.add_argument("--auto-multi-gpu", action="store_true", help="Launch multi-GPU worker pool")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.auto_multi_gpu:
        launch_auto_multi_gpu(args)
        return

    if args.benchmark_csv:
        csv_list = [args.benchmark_csv]
    elif args.task == "all":
        csv_list = sorted(glob.glob(os.path.join(args.prompt_dir, "*.csv")))
    elif args.task == "iscc_l2":
        csv_list = sorted(glob.glob(os.path.join(args.prompt_dir, "*_l2.csv")))
    else:
        csv_list = sorted(glob.glob(os.path.join(args.prompt_dir, f"{args.task}_*.csv")))

    if not csv_list:
        raise FileNotFoundError(f"No task CSVs found matching task '{args.task}' in {args.prompt_dir}")

    print(f"Found {len(csv_list)} task CSVs for task '{args.task}': {[Path(p).name for p in csv_list]}", flush=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading SDXL models on {device} (Shard {args.shard_id}/{args.num_shards})...", flush=True)
    u1, u2, u3 = load_pca_basis()
    sched = load_winning_schedule()
    bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")
    mlp_shift = load_mlp_pca(args.ckpt_path, device=device)
    pipe, vae = setup_sdxl(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)

    for csv_file in csv_list:
        run_benchmark_task(
            csv_file, args.out_dir, args.ckpt_path,
            gpu=args.gpu, seeds=args.seeds,
            shard_id=args.shard_id, num_shards=args.num_shards,
            pipe=pipe, vae=vae, mlp_shift=mlp_shift, seg_models=seg_models,
            u1=u1, u2=u2, u3=u3, bands=bands
        )

    if args.num_shards <= 1:
        merge_manifests(args.out_dir)


if __name__ == "__main__":
    main()

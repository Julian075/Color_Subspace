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
from sd35_core import (
    build_envelope_bands, envelope_weight,
    latent_hw, decode_latents_4d, build_mask_latent,
    PERFIL_GENERATORS, setup_sd35,
)
import utils
from model_pca import load_mlp_pca

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
RESOLUTION = 1024
STEPS = 28
GUIDANCE = 4.5

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
    return {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}


def parse_target_lab(row: pd.Series) -> Tuple[float, float, float]:
    """Robustly extracts CIELAB target color using ISCC-NBS Level 2 as canonical color basis."""
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
def generate_gencolorbench_image(pipe, vae, mlp_shift, seg_models, device,
                                 prompt_text, obj_word, target_lab, seed,
                                 u1, u2, u3, bands,
                                 height=RESOLUTION, width=RESOLUTION):
    """
    Single-pass inference: ONE continuous 28-step diffusion trajectory.
    - NO separate pre-generation baseline pass.
    - At the gate step (first step where envelope weight w > 0), x̂₀ is predicted
      via flow-matching: x̂₀ = xₜ - σₜ·vₜ, decoded once with VAE, segmented
      once with SAM. MLP predicts (m1,m2,m3). State is then locked.
    - Remaining steps apply PCA latent shift smoothly via the envelope weight.
    Gate for SD3.5-M: gate_frac=0.60 (step ~17 of 28).
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
                idx = getattr(sched, "_step_index", None)
                if idx is None and hasattr(sched, "index_for_timestep"):
                    idx = sched.index_for_timestep(timestep)
                sigma = sched.sigmas[idx] if idx is not None else None
                if sigma is not None:
                    captured["pending_x0"] = (sample.detach() - sigma * model_output.detach()).clone()
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
        ).images
    finally:
        pipe.scheduler.step = orig_step

    img_final = decode_latents_4d(vae, latents)
    if state["failed"]:
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
        shard_files = sorted(task_dir.glob("manifest_shard_*.csv"))
        if shard_files:
            dfs = []
            for sf in shard_files:
                try:
                    dfs.append(pd.read_csv(sf))
                except Exception:
                    pass
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                if "task_id" in merged.columns and "seed_idx" in merged.columns:
                    merged = merged.sort_values(by=["task_id", "seed_idx"]).drop_duplicates(subset=["task_id", "seed_idx"])
                merged.to_csv(task_dir / "manifest.csv", index=False)
                print(f"[MERGE] Merged {len(shard_files)} shards into {task_dir / 'manifest.csv'} ({len(merged)} rows)")


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
    print(f">>> LAUNCHING SD3.5-M MULTI-GPU GENCOLORBENCH RUNNER")
    print(f"    Worker GPUs ({num_local_workers}): {worker_gpus}")
    print(f"    Task:       {args.task}")
    print(f"    Out Dir:    {args.out_dir}")
    print("="*70 + "\n")

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
        log_file = open(log_path, "w")
        processes.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))

    for idx, p in enumerate(processes):
        p.wait()
        print(f"Worker {idx} on GPU {worker_gpus[idx]} exited with code {p.returncode}.")

    merge_manifests(args.out_dir)
    print(f"\n>>> Multi-GPU Generation Finished & Manifests Merged!")


def run_benchmark_task(csv_path: str, out_dir: str, ckpt_path: str,
                       gpu: int = 0, seeds: List[int] = [1, 2, 3, 4],
                       shard_id: int = 0, num_shards: int = 1,
                       pipe=None, vae=None, mlp_shift=None, seg_models=None,
                       u1=None, u2=None, u3=None, bands=None):
    task_name = Path(csv_path).stem
    task_out_dir = os.path.join(out_dir, task_name)
    os.makedirs(task_out_dir, exist_ok=True)

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if pipe is None or vae is None or mlp_shift is None or seg_models is None or u1 is None or bands is None:
        print(f"\n[{task_name.upper()}] Loading SD3.5-M models on {device} (Shard {shard_id}/{num_shards})...", flush=True)
        u1, u2, u3 = load_pca_basis()
        sched = load_winning_schedule()
        bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")
        mlp_shift = load_mlp_pca(ckpt_path, device=device)
        pipe, vae = setup_sd35(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS, use_cpu_offload=False)
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

    done_keys = {f"{r['task_id']}_{r['seed_idx']}" for r in results if "task_id" in r and "seed_idx" in r}

    main_manifest = os.path.join(task_out_dir, "manifest.csv")
    if os.path.exists(main_manifest):
        try:
            main_results = pd.read_csv(main_manifest).to_dict("records")
            for r in main_results:
                if "task_id" in r and "seed_idx" in r:
                    done_keys.add(f"{r['task_id']}_{r['seed_idx']}")
        except Exception:
            pass

    shard_idx = 0
    for idx, row in df.iterrows():
        if num_shards > 1 and (idx % num_shards) != shard_id:
            continue
        shard_idx += 1

        task_id = row.get("id", idx + 1)
        raw_prompt = row["prompt"]
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
            key = f"{task_id}_{seed_idx}"
            img_filename = f"{task_id}_{seed_idx}.png"
            img_path = os.path.join(task_out_dir, img_filename)

            if key in done_keys and os.path.exists(img_path):
                continue

            seed_val = int(s * 1000 + int(task_id) * 17) % 1000000

            out = generate_gencolorbench_image(
                pipe, vae, mlp_shift, seg_models, device,
                prompt_text, obj_word, target_lab, seed_val,
                u1, u2, u3, bands
            )

            out["image"].save(img_path)
            dE_final = float(utils.ciede2000(out["final_lab"], target_lab))

            record = {
                "task_id": task_id,
                "seed_idx": seed_idx,
                "seed": seed_val,
                "prompt": prompt_text,
                "raw_prompt": raw_prompt,
                "object": obj_word,
                "target_L": target_lab[0], "target_a": target_lab[1], "target_b": target_lab[2],
                "final_L": out["final_lab"][0], "final_a": out["final_lab"][1], "final_b": out["final_lab"][2],
                "deltaE_final": dE_final,
                "m1": out["m_pred"][0], "m2": out["m_pred"][1], "m3": out["m_pred"][2],
                "image_filename": img_filename,
            }
            results.append(record)
            done_keys.add(key)

            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(manifest_csv, index=False)

        if (shard_idx % 10 == 0) or (shard_idx == total_in_shard):
            print(f"[{task_name}][Shard {shard_id}] Progress: {shard_idx}/{total_in_shard} prompts completed.", flush=True)

    pd.DataFrame(results).to_csv(manifest_csv, index=False)
    print(f"\n>>> [{task_name}][Shard {shard_id}] Complete. Saved manifest to: {manifest_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GenColorBench Evaluation Runner for SD3.5-M")
    parser.add_argument("--prompt-dir", default="/data/140-1/users/jsantamaria/vae_exploration/gencolorbench/mini_bench_prompt")
    parser.add_argument("--benchmark-csv", default=None, help="Path to single GenColorBench task CSV")
    parser.add_argument("--task", default="ncu", choices=["ncu", "all", "cna", "coa", "ica", "moc", "iscc_l2"], help="Task group filter")
    parser.add_argument("--out-dir", default="/data/140-1/users/jsantamaria/vae_exploration/results_paper/sd3.5/gencolorbench_out")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g. '4,2')")
    parser.add_argument("--auto-multi-gpu", action="store_true", help="Launch multi-GPU worker pool")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
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

    print(f"Found {len(csv_list)} task CSVs for task '{args.task}': {[Path(p).name for p in csv_list]}")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SD3.5-M models on {device} (Shard {args.shard_id}/{args.num_shards})...", flush=True)
    u1, u2, u3 = load_pca_basis()
    sched = load_winning_schedule()
    bands = build_envelope_bands(sched["gate_frac"], PERFIL_GENERATORS[sched["perfil_name"]](sched["n_partes"]), "ramp_down")
    mlp_shift = load_mlp_pca(args.ckpt_path, device=device)
    pipe, vae = setup_sd35(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS, use_cpu_offload=False)
    seg_models = utils.setup_seg_models(device)

    for csv_path in csv_list:
        run_benchmark_task(
            csv_path, args.out_dir, args.ckpt_path,
            gpu=args.gpu, seeds=args.seeds,
            shard_id=args.shard_id, num_shards=args.num_shards,
            pipe=pipe, vae=vae, mlp_shift=mlp_shift, seg_models=seg_models,
            u1=u1, u2=u2, u3=u3, bands=bands
        )


if __name__ == "__main__":
    main()

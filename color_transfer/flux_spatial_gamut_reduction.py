#!/usr/bin/env python3
"""Spatially adaptive (pixel-wise chroma-aware) gamut reduction and desaturation in FLUX.1-dev.

Rather than applying a uniform global latent shift, this method:
1. Decodes predicted clean latent x_0 at gate step to obtain spatial CIELAB maps: L(y, x), a(y, x), b(y, x).
2. Contracts a* and b* towards zero per pixel according to reduction ratio r:
   a_target(y, x) = a(y, x) * (1 - r)
   b_target(y, x) = b(y, x) * (1 - r)
   L_target(y, x) = L(y, x) (constant lightness).
3. Evaluates the MLP vectorially across the latent grid to predict displacement m1(y, x), m2(y, x).
4. Injects spatial perturbation modulated by temporal envelope w(t).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

SCRIPT_DIR = Path(__file__).resolve().parent
FLUX_DIR = SCRIPT_DIR.parent / "Flux"
if str(FLUX_DIR) not in sys.path:
    sys.path.insert(0, str(FLUX_DIR))

from flux_core import (
    build_envelope_bands,
    envelope_weight,
    latent_hw,
    unpack_to_4d,
    pack_from_4d,
    decode_latents_4d,
    setup_flux,
)
from model_pca import load_mlp_pca

PCA_AXES_PATH = FLUX_DIR / "fase_a_pca_out" / "pca_axes.json"
WINNING_SCHEDULE_PATH = FLUX_DIR / "fase_b_pca_out" / "fase_b_winning_schedule.json"
DEFAULT_CKPT_PATH = FLUX_DIR / "mlp_training_out" / "mlp_shift_pca_best.pt"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5


def compute_spatial_mlp_shifts(
    mlp_wrapper,
    L_base: np.ndarray,
    a_base: np.ndarray,
    b_base: np.ndarray,
    reduction_ratio: float,
    fix_l: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the ResMLP vectorially across the spatial latent grid (H, W).
    Returns displacement maps m1(H, W), m2(H, W), m3(H, W).
    """
    H, W = L_base.shape
    factor = 1.0 - reduction_ratio

    L_mod = L_base
    a_mod = a_base * factor
    b_mod = b_base * factor

    delta_L = L_mod - L_base
    delta_a = a_mod - a_base
    delta_b = b_mod - b_base

    c_base = np.hypot(a_base, b_base)
    c_mod = np.hypot(a_mod, b_mod)
    delta_c = c_mod - c_base

    h_base_rad = np.arctan2(b_base, a_base)
    h_mod_rad = np.arctan2(b_mod, a_mod)

    sin_h_base, cos_h_base = np.sin(h_base_rad), np.cos(h_base_rad)
    sin_dh, cos_dh = np.sin(h_mod_rad - h_base_rad), np.cos(h_mod_rad - h_base_rad)

    feat = np.stack([
        L_base.flatten(), a_base.flatten(), b_base.flatten(),
        delta_L.flatten(), delta_a.flatten(), delta_b.flatten(),
        L_mod.flatten(), a_mod.flatten(), b_mod.flatten(),
        c_base.flatten(), delta_c.flatten(),
        sin_h_base.flatten(), cos_h_base.flatten(),
        sin_dh.flatten(), cos_dh.flatten(),
    ], axis=-1).astype(np.float32)

    feat_norm = (feat - mlp_wrapper.x_mean) / mlp_wrapper.x_std
    t_in = torch.from_numpy(feat_norm).to(mlp_wrapper.device)

    with torch.no_grad():
        m_out = mlp_wrapper.model(t_in).view(H, W, 3).cpu().numpy()

    m1 = m_out[:, :, 0]
    m2 = m_out[:, :, 1]
    m3 = np.zeros((H, W), dtype=np.float32) if fix_l else m_out[:, :, 2]

    return m1, m2, m3


def run_spatial_gamut_reduction(
    prompt: str,
    reduction_ratio: float,
    seed: int = 42,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "gamut_reduction_out",
    save_baseline: bool = True,
    mlp_ckpt_path: Optional[str] = None,
    preloaded_models: Optional[Dict[str, Any]] = None,
    fix_l: bool = True,
    prefix: str = "van_gogh_spatial",
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tag = f"{prefix}_seed{seed}_red{int(reduction_ratio * 100)}"

    print("=" * 80)
    print("FLUX SPATIAL / CHROMA-AWARE GAMUT REDUCTION")
    print(f"Prompt:            \"{prompt}\"")
    print(f"Reduction:         -{reduction_ratio * 100:.1f}% Chroma Pixel-by-Pixel (Sign-Aware, Fixed L*)")
    print(f"Seed: {seed} | Steps: {steps} | Guidance: {guidance} | Device: {device}")
    print("=" * 80)

    # 1. Load models and configurations
    if preloaded_models is not None:
        pipe = preloaded_models["pipe"]
        vae = preloaded_models["vae"]
        mlp_shift = preloaded_models["mlp_shift"]
        u1 = preloaded_models["u1"]
        u2 = preloaded_models["u2"]
        u3 = preloaded_models["u3"]
        bands = preloaded_models["bands"]
    else:
        print("[1/3] Loading FLUX, VAE, and ResMLP...")
        pipe, vae = setup_flux(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)

        ckpt = mlp_ckpt_path or str(DEFAULT_CKPT_PATH)
        mlp_shift = load_mlp_pca(ckpt, device=device)

        with open(PCA_AXES_PATH, "r") as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)

        from flux_core import PERFIL_GENERATORS
        if WINNING_SCHEDULE_PATH.exists():
            with open(WINNING_SCHEDULE_PATH, "r") as f:
                sched_cfg = json.load(f)
        else:
            sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

        bands = build_envelope_bands(
            sched_cfg["gate_frac"],
            PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
            "ramp_down",
        )

    # 2. Baseline generation (if ratio is 0)
    if reduction_ratio <= 1e-6:
        print("\n[Step] Generating unsteered baseline image (0% reduction)...")
        gen = torch.Generator(device=device).manual_seed(seed)
        pipe.scheduler.sigmas = pipe.scheduler.sigmas.to(device)
        with torch.no_grad():
            out_img = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=gen,
                height=resolution,
                width=resolution,
            ).images[0]

        img_path = out_path / f"{prefix}_seed{seed}_red0.png"
        out_img.save(img_path)
        arr = np.array(out_img) / 255.0
        lab = rgb2lab(arr)
        C_map = np.hypot(lab[:, :, 1], lab[:, :, 2])
        print(f"  -> Saved baseline: {img_path.name}")
        print(f"     Mean L*={lab[:,:,0].mean():.1f} | Mean Chroma={C_map.mean():.2f} (P90={np.percentile(C_map, 90):.2f})")
        return {
            "tag": tag,
            "path": str(img_path),
            "ratio": 0.0,
            "C_pixel_mean": float(C_map.mean()),
            "L_mean": float(lab[:, :, 0].mean()),
        }

    # 3. Generation with Chroma-Aware Spatial Modulation
    print(f"\n[Step] Generating image with Spatial Gamut Reduction (-{reduction_ratio * 100:.1f}%)...")
    lat_h = resolution // 8
    lat_w = resolution // 8

    state = {
        "locked": False,
        "delta_latent_spatial": None,
    }
    captured = {}
    orig_step = pipe.scheduler.step

    def patched_step(model_output, timestep, sample, *a, **k):
        if not state["locked"]:
            try:
                sched = pipe.scheduler
                idx = (sched.index_for_timestep(timestep)
                       if hasattr(sched, "index_for_timestep")
                       else getattr(sched, "_step_index", None))
                sigma = sched.sigmas[idx] if idx is not None else None
                if sigma is not None:
                    captured["pending_x0"] = (sample.detach() - sigma * model_output.detach()).clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def cb(pipe_, step_index, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)

        if not state["locked"]:
            if w == 0.0:
                return callback_kwargs

            # === GATE STEP (t=14, frac=0.5) ===
            pred_x0 = captured.get("pending_x0", lat)
            img_x0_np = decode_latents_4d(vae, unpack_to_4d(pipe_, pred_x0, resolution, resolution))

            # Extract spatial CIELAB maps
            arr_x0 = img_x0_np.astype(np.float32) / 255.0
            lab_x0 = rgb2lab(arr_x0)
            L_img = lab_x0[:, :, 0]
            a_img = lab_x0[:, :, 1]
            b_img = lab_x0[:, :, 2]

            # Downsample to latent resolution (128x128) via area interpolation
            t_L = torch.from_numpy(L_img).unsqueeze(0).unsqueeze(0).float()
            t_a = torch.from_numpy(a_img).unsqueeze(0).unsqueeze(0).float()
            t_b = torch.from_numpy(b_img).unsqueeze(0).unsqueeze(0).float()

            L_lat = F.interpolate(t_L, size=(lat_h, lat_w), mode="area").squeeze().numpy()
            a_lat = F.interpolate(t_a, size=(lat_h, lat_w), mode="area").squeeze().numpy()
            b_lat = F.interpolate(t_b, size=(lat_h, lat_w), mode="area").squeeze().numpy()

            # Predict spatial latent displacements m1(y, x), m2(y, x)
            m1_grid, m2_grid, m3_grid = compute_spatial_mlp_shifts(
                mlp_shift, L_lat, a_lat, b_lat, reduction_ratio, fix_l=fix_l
            )

            # Construct spatial latent perturbation tensor: (1, 16, lat_h, lat_w)
            u1_t = torch.tensor(u1, device=device, dtype=torch.float32).view(1, NUM_LATENT_CHANNELS, 1, 1)
            u2_t = torch.tensor(u2, device=device, dtype=torch.float32).view(1, NUM_LATENT_CHANNELS, 1, 1)

            m1_t = torch.from_numpy(m1_grid).to(device=device, dtype=torch.float32).view(1, 1, lat_h, lat_w)
            m2_t = torch.from_numpy(m2_grid).to(device=device, dtype=torch.float32).view(1, 1, lat_h, lat_w)

            delta_spatial = m1_t * u1_t + m2_t * u2_t  # (1, 16, lat_h, lat_w)
            state["delta_latent_spatial"] = delta_spatial.to(DTYPE)
            state["locked"] = True

            print(f"  [GATE STEP @ Step {step_index} | frac={frac:.2f}]")
            print(f"    Latent m1 range: [{m1_grid.min():+.4f}, {m1_grid.max():+.4f}] (Adaptive Yellow-Blue)")
            print(f"    Latent m2 range: [{m2_grid.min():+.4f}, {m2_grid.max():+.4f}] (Adaptive Green-Red)")
            print(f"    Latent m3: strictly 0.0 (Fixed L*)")

        if w == 0.0 or state["delta_latent_spatial"] is None:
            return callback_kwargs

        unpacked = unpack_to_4d(pipe_, lat, resolution, resolution)
        out = unpacked + (w * state["delta_latent_spatial"])
        callback_kwargs["latents"] = pack_from_4d(pipe_, out)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    gen = torch.Generator(device=device).manual_seed(seed)
    pipe.scheduler.sigmas = pipe.scheduler.sigmas.to(device)

    try:
        with torch.no_grad():
            out_img = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=gen,
                height=resolution,
                width=resolution,
                callback_on_step_end=cb,
            ).images[0]
    finally:
        pipe.scheduler.step = orig_step

    img_path = out_path / f"{prefix}_seed{seed}_red{int(reduction_ratio * 100)}.png"
    out_img.save(img_path)

    arr = np.array(out_img) / 255.0
    lab = rgb2lab(arr)
    C_map = np.hypot(lab[:, :, 1], lab[:, :, 2])
    print(f"  -> Saved output: {img_path.name}")
    print(f"     Mean L*={lab[:,:,0].mean():.1f} | Mean Chroma={C_map.mean():.2f} (P90={np.percentile(C_map, 90):.2f})")

    return {
        "tag": tag,
        "path": str(img_path),
        "ratio": reduction_ratio,
        "C_pixel_mean": float(C_map.mean()),
        "L_mean": float(lab[:, :, 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Chroma-aware spatial gamut reduction with FLUX.1-dev")
    parser.add_argument("--prompt", type=str, default="A painting in the style of Van Gogh", help="Text prompt")
    parser.add_argument("--reduction", type=float, default=0.50, help="Chroma reduction factor (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--out-dir", type=str, default="gamut_reduction_out", help="Output directory")
    parser.add_argument("--prefix", type=str, default="van_gogh_spatial", help="Output filename prefix")
    args = parser.parse_args()

    run_spatial_gamut_reduction(
        prompt=args.prompt,
        reduction_ratio=args.reduction,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

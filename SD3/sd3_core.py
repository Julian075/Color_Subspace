"""
SD3_CORE.PY -- Core Latent Subspace and Temporal Engine for Stable Diffusion 3 Medium (SD3).

Implements:
  1. SD3 Pipeline Loader (with CPU offloading for 24GB GPUs)
  2. Direct 4D Latent Channel Steering in VAE latent space (16 channels)
  3. Temporal Profile Scheduling (Step-wise Denoising Weight Envelopes)
  4. SAM-3 Spatial Mask Latent Downsampling & Overlay
  5. Multi-GPU Subprocess Orchestration
"""

import os
import sys
import math
import subprocess
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, AutoencoderKL

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
RESOLUTION = 1024
STEPS = 28
GUIDANCE = 7.0

# =========================== TEMPORAL ENVELOPE GENERATORS ===========================

def build_envelope_bands(gate_frac: float,
                         profile_vals: List[float],
                         transition_mode: str = "ramp_down") -> List[Tuple[float, float, float, float]]:
    """
    Builds piecewise-linear temporal envelope bands w(t) for Flow-Matching denoising.
    gate_frac: fraction of trajectory where color control is active [0, gate_frac].
    """
    n = len(profile_vals)
    if n == 0 or gate_frac <= 0.0:
        return []

    band_len = gate_frac / n
    bands = []
    for i, val in enumerate(profile_vals):
        t_start = i * band_len
        t_end = (i + 1) * band_len
        bands.append((t_start, t_end, val, val))

    if transition_mode == "ramp_down" and bands:
        last_s, last_e, _, _ = bands[-1]
        bands[-1] = (last_s, last_e, profile_vals[-1], 0.0)

    return bands


def envelope_weight(frac: float, bands: List[Tuple[float, float, float, float]]) -> float:
    for t_s, t_e, w_s, w_e in bands:
        if t_s <= frac <= t_e:
            if t_e == t_s:
                return float(w_s)
            alpha = (frac - t_s) / (t_e - t_s)
            return float(w_s + alpha * (w_e - w_s))
    return 0.0


PERFIL_GENERATORS: Dict[str, Callable[[int], List[float]]] = {
    "plano": lambda n: [1.0] * n,
    "ascendente": lambda n: [float(i + 1) / n for i in range(n)],
    "descendente": lambda n: [float(n - i) / n for i in range(n)],
    "triangular": lambda n: [1.0 - abs(2.0 * i / (n - 1) - 1.0) for i in range(n)] if n > 1 else [1.0],
}


# =========================== 4D LATENT STEERING OPERATORS ===========================

def shift_channels(latents_4d: torch.Tensor,
                   channel_idxs: List[int],
                   magnitude: float,
                   weight: float = 1.0,
                   m_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies additive shift delta along specific latent channels in 4D (B, C, H, W) tensor.
    """
    out = latents_4d.clone()
    delta = float(magnitude * weight)
    for c in channel_idxs:
        if m_mask is not None:
            out[:, c] += delta * m_mask
        else:
            out[:, c] += delta
    return out


def shift_direction(latents_4d: torch.Tensor,
                    direction: List[Tuple[int, float]],
                    magnitude: float,
                    weight: float = 1.0,
                    m_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = latents_4d.clone()
    for c, d in direction:
        delta = float(d * magnitude * weight)
        if m_mask is not None:
            out[:, c] += delta * m_mask
        else:
            out[:, c] += delta
    return out


def shift_pca_4d(latents_4d: torch.Tensor,
                 u1: np.ndarray, u2: np.ndarray, u3: np.ndarray,
                 m1: float, m2: float, m3: float,
                 weight: float = 1.0,
                 m_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Shifts 4D latent tensor along the orthonormal PCA color subspace spanned by (u1, u2, u3).
    """
    out = latents_4d.clone()
    num_ch = latents_4d.shape[1]
    for c in range(num_ch):
        delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c]) * weight
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


# =========================== SD3 SETUP & LATENT MANAGEMENT ===========================
def setup_sd3(model_id="stabilityai/stable-diffusion-3-medium-diffusers", device="cuda", dtype=torch.bfloat16, num_latent_channels=16, use_cpu_offload=False):
    print(f"Loading SD 3 Medium ({model_id})...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    if use_cpu_offload and str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    print(f"  vae.config.latent_channels={vae.config.latent_channels} "
          f"scaling_factor={vae.config.scaling_factor} shift_factor={vae.config.shift_factor}")
    assert vae.config.latent_channels == num_latent_channels, (
        f"Expected {num_latent_channels} channels, but loaded VAE has {vae.config.latent_channels}"
    )
    return pipe, vae


def latent_hw(height, width, vae_scale_factor=8):
    return int(height) // vae_scale_factor, int(width) // vae_scale_factor


@torch.no_grad()
def decode_latents_4d(vae, latents_4d):
    """
    Decodes (B, 16, H_lat, W_lat) latents in raw units to uint8 numpy RGB image.
    Uses SD3 VAE shift_factor and scaling_factor.
    """
    shift = getattr(vae.config, "shift_factor", 0.0609)
    scale = getattr(vae.config, "scaling_factor", 1.5305)
    x = (latents_4d.to(torch.float32) / scale) + shift
    dec = (vae.decode(x).sample / 2 + 0.5).clamp(0, 1)
    return (dec[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# =========================== SPATIAL MASK PROJECTION ===========================
def build_mask_latent(mask_pixel, latent_h, latent_w, device):
    mask_pil = Image.fromarray((mask_pixel.astype(np.uint8)) * 255).resize(
        (latent_w, latent_h), Image.NEAREST
    )
    arr = (np.array(mask_pil) > 127).astype(np.float32)
    return torch.from_numpy(arr).to(device)[None, None]


def save_mask_preview(img_uint8, mask_pixel, out_path):
    overlay = img_uint8.astype(np.float32).copy()
    red = np.array([255.0, 40.0, 40.0])
    overlay[mask_pixel] = 0.45 * overlay[mask_pixel] + 0.55 * red
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


# =========================== SD3 GENERATION PIPELINE RUNNER ===========================
@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps=STEPS, guidance=GUIDANCE,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None,
                   pca_basis=None, m_vector=None):
    """
    Executes SD 3 generation with optional latent steering during the denoising trajectory.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    is_controlled = (
        (channel_idxs is not None or combo is not None or direction is not None or pca_basis is not None) and
        (magnitude is not None or m_vector is not None) and
        bands is not None
    )

    if not is_controlled:
        latents = pipe(
            prompt,
            height=height, width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
            output_type="latent",
        ).images
        return latents

    m_mask = mask_latent[:, 0].to(DTYPE) if mask_latent is not None else None

    def callback_fn(pipe_, step_index, timestep, callback_kwargs):
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)
        if w == 0.0:
            return callback_kwargs

        lat = callback_kwargs["latents"]

        if pca_basis is not None and m_vector is not None:
            u1, u2, u3 = pca_basis
            m1, m2, m3 = m_vector
            lat = shift_pca_4d(lat, u1, u2, u3, m1, m2, m3, weight=w, m_mask=m_mask)
        elif direction is not None and magnitude is not None:
            lat = shift_direction(lat, direction, magnitude, weight=w, m_mask=m_mask)
        elif combo is not None and direction is not None:
            lat = shift_direction(lat, list(zip(combo, direction)), magnitude, weight=w, m_mask=m_mask)
        elif channel_idxs is not None:
            lat = shift_channels(lat, channel_idxs, magnitude, weight=w, m_mask=m_mask)

        callback_kwargs["latents"] = lat
        return callback_kwargs

    latents = pipe(
        prompt,
        height=height, width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
        output_type="latent",
        callback_on_step_end=callback_fn,
    ).images
    return latents


# =========================== MULTI-GPU WORKER ORCHESTRATION ===========================
def get_available_gpus() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    count = torch.cuda.device_count()
    return list(range(count)) if count > 0 else [0]


def spawn_workers(script_path: str, n_tasks: int, extra_args: List[str] = None):
    """
    Splits n_tasks evenly across available GPUs and spawns worker subprocesses.
    """
    gpus = get_available_gpus()
    n_gpus = len(gpus)
    print(f"[ORCHESTRATOR] Spawning {n_gpus} workers for {n_tasks} tasks across GPUs: {gpus}")

    chunk_size = math.ceil(n_tasks / n_gpus)
    procs = []

    for worker_id, gpu_id in enumerate(gpus):
        start_idx = worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, n_tasks)
        if start_idx >= n_tasks:
            break

        cmd = [
            sys.executable, script_path,
            "--worker-id", str(worker_id),
            "--gpu", "0",
            "--task-start", str(start_idx),
            "--task-end", str(end_idx),
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    for p in procs:
        p.wait()
    print("[ORCHESTRATOR] All worker processes finished.")

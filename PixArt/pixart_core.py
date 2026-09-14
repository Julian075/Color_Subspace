"""
PIXART_CORE.PY -- Core Latent Subspace and Temporal Engine for PixArt Models (PixArt-alpha & PixArt-Sigma).

Implements:
  1. PixArt Pipeline Loader (Supports PixArt-alpha [120 tokens] and PixArt-Sigma [300 tokens])
  2. Direct 4D Latent Channel Steering in VAE latent space (4 channels)
  3. Temporal Profile Scheduling (Step-wise Denoising Weight Envelopes)
  4. SAM-3 Spatial Mask Latent Downsampling & Overlay
  5. Multi-GPU Subprocess Orchestration
"""

import os
import sys
import math
import subprocess
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from PIL import Image
from diffusers import PixArtAlphaPipeline, PixArtSigmaPipeline, AutoencoderKL

MODEL_ID_ALPHA = "PixArt-alpha/PixArt-XL-2-1024-MS"
MODEL_ID_SIGMA = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4
RESOLUTION = 1024
STEPS = 20
GUIDANCE = 4.5
MAX_SEQ_LEN_ALPHA = 120
MAX_SEQ_LEN_SIGMA = 300

# =========================== TEMPORAL ENVELOPE GENERATORS ===========================

def build_envelope_bands(gate_frac: float,
                         profile_vals: List[float],
                         transition_mode: str = "ramp_down") -> List[Tuple[float, float, float, float]]:
    """
    Builds piecewise-linear temporal envelope bands w(t) for denoising.
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
    Applies additive shift delta along specific latent channels in 4D (B, 4, H, W) tensor.
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
                    combo: Tuple[int, ...],
                    direction: Tuple[float, ...],
                    magnitude: float,
                    weight: float = 1.0,
                    m_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = latents_4d.clone()
    for c, d in zip(combo, direction):
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


# =========================== PIXART SETUP & LATENT MANAGEMENT ===========================
def setup_pixart(model_type="alpha", model_id=None, device="cuda", dtype=torch.float16, num_latent_channels=4, use_cpu_offload=False):
    if model_id is None:
        model_id = MODEL_ID_ALPHA if model_type == "alpha" else MODEL_ID_SIGMA

    print(f"Loading PixArt ({model_type.upper()}: {model_id})...")
    pipe_cls = PixArtAlphaPipeline if model_type == "alpha" else PixArtSigmaPipeline
    pipe = pipe_cls.from_pretrained(model_id, torch_dtype=dtype)

    if use_cpu_offload and str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    print(f"  vae.config.latent_channels={vae.config.latent_channels} scaling_factor={vae.config.scaling_factor}")
    assert vae.config.latent_channels == num_latent_channels, (
        f"Expected {num_latent_channels} channels, but loaded VAE has {vae.config.latent_channels}"
    )
    return pipe, vae


def latent_hw(height, width, vae_scale_factor=8):
    return int(height) // vae_scale_factor, int(width) // vae_scale_factor


@torch.no_grad()
def decode_latents_4d(vae, latents_4d):
    """
    Decodes (B, 4, H_lat, W_lat) latents in raw units to uint8 numpy RGB image.
    Uses PixArt VAE scaling_factor (0.18215).
    """
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    x = latents_4d.to(torch.float32) / scale
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


# =========================== PIXART GENERATION PIPELINE RUNNER ===========================
@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps=STEPS, guidance=GUIDANCE,
                   max_sequence_length=MAX_SEQ_LEN_ALPHA,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None,
                   pca_basis=None, m_vector=None):
    """
    Executes PixArt generation with optional latent steering during the denoising trajectory.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    is_controlled = (
        (channel_idxs is not None or combo is not None or pca_basis is not None) and
        (magnitude is not None or m_vector is not None) and
        bands is not None
    )

    if not is_controlled:
        latents = pipe(
            prompt,
            height=height, width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=max_sequence_length,
            clean_caption=False,
            generator=generator,
            output_type="latent",
        ).images
        return latents

    m_mask = mask_latent[:, 0].to(DTYPE) if mask_latent is not None else None

    def callback_fn(step_index, timestep, latents):
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)
        if w == 0.0:
            return

        if pca_basis is not None and m_vector is not None:
            u1, u2, u3 = pca_basis
            m1, m2, m3 = m_vector
            shifted = shift_pca_4d(latents, u1, u2, u3, m1, m2, m3, weight=w, m_mask=m_mask)
        elif combo is not None and direction is not None:
            shifted = shift_direction(latents, combo, direction, magnitude, weight=w, m_mask=m_mask)
        elif channel_idxs is not None:
            shifted = shift_channels(latents, channel_idxs, magnitude, weight=w, m_mask=m_mask)
        else:
            shifted = latents

        latents.copy_(shifted)

    latents = pipe(
        prompt,
        height=height, width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        clean_caption=False,
        generator=generator,
        output_type="latent",
        callback=callback_fn,
        callback_steps=1,
    ).images
    return latents


# =========================== MULTI-GPU WORKER ORCHESTRATION ===========================
def get_available_gpus() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    count = torch.cuda.device_count()
    return list(range(count)) if count > 0 else [0]


def spawn_workers(script_path: str, n_tasks: int, extra_args: List[str] = None, workers_per_gpu: int = 1):
    """
    Splits n_tasks evenly across available GPUs (with workers_per_gpu processes per GPU)
    and spawns worker subprocesses.
    """
    gpus = get_available_gpus()
    gpu_slots = []
    for g in gpus:
        gpu_slots.extend([g] * workers_per_gpu)

    n_workers = len(gpu_slots)
    print(f"[ORCHESTRATOR] Spawning {n_workers} workers ({workers_per_gpu} per GPU) for {n_tasks} tasks across GPUs: {gpus}")

    chunk_size = math.ceil(n_tasks / n_workers)
    procs = []

    for worker_id, gpu_id in enumerate(gpu_slots):
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
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    for p in procs:
        p.wait()
    print("[ORCHESTRATOR] All worker processes finished.")

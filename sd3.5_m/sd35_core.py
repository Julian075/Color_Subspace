"""
SD35_CORE.PY -- Low-level mechanisms and core engine for Stable Diffusion 3.5 Medium (SD3.5-M).

Features:
- SD3.5-M Model loading (16-channel VAE with scaling_factor=1.5305, shift_factor=0.0609).
- Direct 4D latent operations without patch-packing.
- Dynamic temporal scheduling envelopes (cosine, ramp-down, ascending, smooth).
- Spatial mask projection from pixel space (SAM3) to latent space.
- Multi-channel, directional, and PCA basis latent steering.
- Robust multi-GPU subprocessing with physical GPU isolation.
"""

import os
import json
import subprocess
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline, AutoencoderKL
from PIL import Image


# =========================== SCHEDULE & ENVELOPES ===========================
def build_envelope_bands(gate_frac, chrono, mode="ramp_down"):
    """
    Partitions [gate_frac, 1.0] into len(chrono) equal bands, each with peak magnitude chrono[i].
    """
    n = len(chrono)
    edges = np.linspace(gate_frac, 1.0, n + 1)
    return [(float(edges[i]), float(edges[i + 1]), chrono[i], mode) for i in range(n)]


def shift_schedule(frac, lo, hi, peak, mode="ramp_down"):
    if frac < lo or frac > hi:
        return 0.0
    p = (frac - lo) / max(hi - lo, 1e-8)
    p = float(min(max(p, 0.0), 1.0))
    w = (1.0 - p) if mode == "ramp_down" else 1.0
    return peak * w


def envelope_weight(frac, bands):
    return sum(shift_schedule(frac, lo, hi, peak, mode) for lo, hi, peak, mode in bands)


PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}


# =========================== LATENT SHIFTING OPERATORS ===========================
def compute_ref(latents_4d, channel_idxs, ref_mode, ref_channels, mask=None):
    """
    ref_mode: 'none' (absolute magnitude) | 'self' (normalized by mean magnitude of steered channels)
    """
    if ref_mode == "none":
        return 1.0
    cols = channel_idxs if ref_mode == "self" else ref_channels
    sub = latents_4d[:, cols]
    if mask is not None:
        m = mask.expand(-1, sub.shape[1], -1, -1).bool()
        vals = sub[m]
        if vals.numel() > 0:
            return float(vals.abs().mean().item())
    return float(sub.abs().mean().item())


def shift_channels(latents_4d, channel_idxs, magnitude, mask=None):
    """
    Applies uniform scalar shift to specified channel indices.
    mask: None -> whole image; (1, 1, H_lat, W_lat) -> within mask boundary.
    """
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in channel_idxs:
        out[:, c] += magnitude * m if m is not None else magnitude
    return out


def shift_direction(latents_4d, direction, magnitude, mask=None):
    """
    direction: list of (channel_idx, weight_or_sign).
    Normalized to unit L2 norm to ensure magnitude comparability.
    """
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    norm = float(np.sqrt(sum(w * w for _, w in direction))) or 1.0
    for c, w in direction:
        delta = magnitude * (w / norm)
        out[:, c] += delta * m if m is not None else delta
    return out


def shift_multi_direction(latents_4d, direction_magnitude_pairs, mask=None):
    """
    Applies multiple directional shifts simultaneously (e.g. L + a + b).
    """
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for direction, magnitude in direction_magnitude_pairs:
        norm = float(np.sqrt(sum(w * w for _, w in direction))) or 1.0
        for c, w in direction:
            delta = magnitude * (w / norm)
            out[:, c] += delta * m if m is not None else delta
    return out


def shift_pca_4d(latents_4d, u1, u2, u3, m1, m2, m3, mask=None):
    """
    Direct steering along top 3 PCA orthonormal eigenvectors (U1, U2, U3).
    """
    out = latents_4d.clone()
    m_mask = mask[:, 0].to(out.dtype) if mask is not None else None
    num_channels = latents_4d.shape[1]
    for c in range(num_channels):
        delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c])
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


# =========================== SD3.5 SETUP & LATENT MANAGEMENT ===========================
def setup_sd35(model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda", dtype=torch.bfloat16, num_latent_channels=16, use_cpu_offload=False):
    print(f"Loading SD 3.5 Medium ({model_id})...")
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
    Uses SD3.5 VAE shift_factor and scaling_factor.
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


# =========================== SD3.5 GENERATION PIPELINE RUNNER ===========================
@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps, guidance,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None,
                   pca_basis=None, m_vector=None):
    """
    Executes SD 3.5 generation with optional latent steering during the denoising trajectory.
    """
    apply_shift = bands is not None and (
        combo is not None or
        (pca_basis is not None and m_vector is not None) or
        (magnitude is not None and (channel_idxs is not None or direction is not None))
    )

    if combo is not None:
        ref_cols = ref_channels
    else:
        ref_cols = channel_idxs if channel_idxs is not None else ([c for c, _ in direction] if direction is not None else None)

    def cb(pipe_, step_index, timestep, callback_kwargs):
        if not apply_shift:
            return callback_kwargs
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)
        if w == 0.0:
            return callback_kwargs

        latents = callback_kwargs["latents"]  # (B, 16, H_lat, W_lat)
        ref = compute_ref(latents, ref_cols, ref_mode, ref_channels, mask=mask_latent)

        if pca_basis is not None and m_vector is not None:
            u1, u2, u3 = pca_basis
            m1, m2, m3 = m_vector
            shifted = shift_pca_4d(latents, u1, u2, u3, m1 * w * ref, m2 * w * ref, m3 * w * ref, mask=mask_latent)
        elif combo is not None:
            pairs = [(d, mag * w * ref) for d, mag in combo]
            shifted = shift_multi_direction(latents, pairs, mask=mask_latent)
        elif direction is not None:
            shifted = shift_direction(latents, direction, magnitude * w * ref, mask=mask_latent)
        else:
            shifted = shift_channels(latents, channel_idxs, magnitude * w * ref, mask=mask_latent)

        callback_kwargs["latents"] = shifted
        return callback_kwargs

    latents = pipe(
        prompt, height=height, width=width,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="latent",
        callback_on_step_end=cb if apply_shift else None,
    ).images
    return latents


# =========================== MULTI-GPU PARALLELIZATION ===========================
def get_gpu_list(available_gpus=None):
    """
    Returns physical GPU indices respecting parent CUDA_VISIBLE_DEVICES.
    """
    if available_gpus is not None:
        return [str(g) for g in available_gpus]
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        physical = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
        if physical:
            return physical
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No GPUs detected.")
    return [str(i) for i in range(n)]


def chunk_list(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def spawn_workers(tasks, n_workers, gpu_list, workers_per_gpu, tmp_dir, worker_cmd_base):
    os.makedirs(tmp_dir, exist_ok=True)
    chunks = chunk_list(tasks, n_workers)
    procs = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        chunk_path = os.path.join(tmp_dir, f"chunk{i}.json")
        with open(chunk_path, "w") as f:
            json.dump(chunk, f)
        gpu_id = gpu_list[i // workers_per_gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = worker_cmd_base + ["--chunk", chunk_path]
        print(f"  Worker {i}: Physical GPU {gpu_id}, {len(chunk)} tasks")
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]

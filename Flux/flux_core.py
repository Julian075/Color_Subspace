"""FLUX_CORE: Core latent manipulation, temporal schedule modulation, and multi-GPU execution for FLUX.1."""

import os
import json
import subprocess
import numpy as np
import torch
from diffusers import FluxPipeline, AutoencoderKL
from PIL import Image


# =========================== schedule (ramps within bands) ===========================
def build_envelope_bands(gate_frac, chrono, mode="ramp_down"):
    """Splits [gate_frac, 1.0] into equal-width bands with corresponding peak magnitude weights."""
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


# Envelope profile generators
PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}


# =========================== shift scaling (ref) ===========================
def compute_ref(latents_4d, channel_idxs, ref_mode, ref_channels, mask=None):
    """Computes reference scale: 'none' (1.0), 'self' (active channels), or 'channels' (fixed subset)."""
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
    """Applies uniform shift to specified channels inside optional spatial mask."""
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in channel_idxs:
        out[:, c] += magnitude * m if m is not None else magnitude
    return out


def shift_direction(latents_4d, direction, magnitude, mask=None):
    """Applies a normalized composite direction vector to latents inside optional spatial mask."""
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    norm = float(np.sqrt(sum(sign * sign for _, sign in direction))) or 1.0
    for c, sign in direction:
        delta = magnitude * (sign / norm)
        out[:, c] += delta * m if m is not None else delta
    return out


def shift_multi_direction(latents_4d, direction_magnitude_pairs, mask=None):
    """Applies multiple directional shifts simultaneously to latents inside optional spatial mask."""
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for direction, magnitude in direction_magnitude_pairs:
        norm = float(np.sqrt(sum(sign * sign for _, sign in direction))) or 1.0
        for c, sign in direction:
            delta = magnitude * (sign / norm)
            out[:, c] += delta * m if m is not None else delta
    return out


# =========================== FLUX: setup, pack/unpack, decode ===========================
def setup_flux(model_id, device, dtype, num_latent_channels):
    print(f"Loading FLUX ({model_id})...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    # VAE in float32 for clean decoding
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    print(f"  vae.config.latent_channels={vae.config.latent_channels} "
          f"scaling_factor={vae.config.scaling_factor} shift_factor={vae.config.shift_factor}")
    assert vae.config.latent_channels == num_latent_channels, (
        f"expected {num_latent_channels} channels, loaded VAE has {vae.config.latent_channels}")
    return pipe, vae


def latent_hw(pipe, height, width):
    """Height/width of unpacked latent (before 2x2 patchification) as expected by pipe._pack_latents."""
    vsf = pipe.vae_scale_factor
    return 2 * (int(height) // (vsf * 2)), 2 * (int(width) // (vsf * 2))


def unpack_to_4d(pipe, latents_packed, height, width):
    return pipe._unpack_latents(latents_packed, height, width, pipe.vae_scale_factor)


def pack_from_4d(pipe, latents_4d, latent_h=None, latent_w=None):
    b, c, h, w = latents_4d.shape
    lh = latent_h if latent_h is not None else h
    lw = latent_w if latent_w is not None else w
    return pipe._pack_latents(latents_4d, b, c, lh, lw)


@torch.no_grad()
def decode_latents_4d(vae, latents_4d):
    """Decode raw latent tensor (B, 16, H_lat, W_lat) into uint8 RGB image."""
    x = latents_4d.to(torch.float32) / vae.config.scaling_factor + vae.config.shift_factor
    dec = (vae.decode(x).sample / 2 + 0.5).clamp(0, 1)
    return (dec[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_mask_latent(mask_pixel, latent_h, latent_w, device):
    mask_pil = Image.fromarray((mask_pixel.astype(np.uint8)) * 255).resize(
        (latent_w, latent_h), Image.NEAREST)
    arr = (np.array(mask_pil) > 127).astype(np.float32)
    return torch.from_numpy(arr).to(device)[None, None]


def save_mask_preview(img_uint8, mask_pixel, out_path):
    overlay = img_uint8.astype(np.float32).copy()
    red = np.array([255.0, 40.0, 40.0])
    overlay[mask_pixel] = 0.45 * overlay[mask_pixel] + 0.55 * red
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


# =========================== Generation with optional latent steering ===========================
@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps, guidance,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None):
    """Generates latents with optional latent directional shifts applied during denoising steps."""
    apply_shift = bands is not None and (combo is not None or
                                         (magnitude is not None and (channel_idxs is not None or direction is not None)))
    latent_h, latent_w = latent_hw(pipe, height, width)
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
        latents = callback_kwargs["latents"]
        unpacked = unpack_to_4d(pipe_, latents, height, width)
        ref = compute_ref(unpacked, ref_cols, ref_mode, ref_channels, mask=mask_latent)
        if combo is not None:
            pairs = [(d, mag * w * ref) for d, mag in combo]
            shifted = shift_multi_direction(unpacked, pairs, mask=mask_latent)
        elif direction is not None:
            shifted = shift_direction(unpacked, direction, magnitude * w * ref, mask=mask_latent)
        else:
            shifted = shift_channels(unpacked, channel_idxs, magnitude * w * ref, mask=mask_latent)
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    latents_packed = pipe(
        prompt, height=height, width=width,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        output_type="latent",
        callback_on_step_end=cb if apply_shift else None,
    ).images
    return latents_packed


# =========================== multi-GPU parallelization ===========================
def get_gpu_list(available_gpus):
    """Resolves physical GPU device IDs from CUDA_VISIBLE_DEVICES or PyTorch."""
    if available_gpus is not None:
        return [str(g) for g in available_gpus]
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        physical = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
        if physical:
            return physical
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No GPU detected -- specify available GPUs explicitly.")
    return [str(i) for i in range(n)]


def chunk_list(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def spawn_workers(tasks, n_workers, gpu_list, workers_per_gpu, tmp_dir, worker_cmd_base):
    """Partitions tasks across worker subprocesses pinned to available GPUs."""
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
        print(f"  worker {i}: GPU fisica {gpu_id}, {len(chunk)} tareas")
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]
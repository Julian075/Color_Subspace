"""
SDXL_CORE.PY -- Low-level mechanisms and core engine for Stable Diffusion XL (SDXL Base 1.0).

Features:
- SDXL Model loading (4-channel VAE with scaling_factor=0.13025, variant='fp16').
- Direct 4D latent operations (B, 4, H_lat, W_lat) without patch-packing.
- Dynamic temporal scheduling envelopes (plano, ascendente, suave, angosto_alto, descendente).
- Spatial mask projection from pixel space (SAM-3) to latent space.
- Multi-channel, directional, and PCA basis latent steering.
- Robust multi-GPU subprocessing with physical GPU isolation.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4
RESOLUTION = 1024
STEPS = 30
GUIDANCE = 5.0


# =========================== SCHEDULE & ENVELOPES ===========================
def build_envelope_bands(gate_frac: float,
                         profile_vals: List[float],
                         transition_mode: str = "ramp_down") -> List[Tuple[float, float, float, str]]:
    """
    Partitions [gate_frac, 1.0] into len(profile_vals) equal bands, each with peak magnitude profile_vals[i].
    """
    n = len(profile_vals)
    if n == 0 or gate_frac >= 1.0:
        return []
    edges = np.linspace(gate_frac, 1.0, n + 1)
    return [(float(edges[i]), float(edges[i + 1]), float(profile_vals[i]), transition_mode) for i in range(n)]


def shift_schedule(frac: float, lo: float, hi: float, peak: float, mode: str = "ramp_down") -> float:
    if frac < lo or frac > hi:
        return 0.0
    p = (frac - lo) / max(hi - lo, 1e-8)
    p = float(min(max(p, 0.0), 1.0))
    w = (1.0 - p) if mode == "ramp_down" else 1.0
    return float(peak * w)


def envelope_weight(frac: float, bands: List[Tuple[float, float, float, str]]) -> float:
    return sum(shift_schedule(frac, lo, hi, peak, mode) for lo, hi, peak, mode in bands)


PERFIL_GENERATORS: Dict[str, Callable[[int], List[float]]] = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
    "triangular":   lambda n: [1.0 - abs(2.0 * i / (n - 1) - 1.0) for i in range(n)] if n > 1 else [1.0],
}


# =========================== LATENT SHIFTING OPERATORS ===========================
def compute_ref(latents_4d: torch.Tensor, channel_idxs, ref_mode: str, ref_channels=None, mask=None) -> float:
    """
    ref_mode: 'none' (absolute magnitude) | 'self' (normalized by mean magnitude of steered channels)
    """
    if ref_mode == "none":
        return 1.0
    cols = channel_idxs if ref_mode == "self" else ref_channels
    if cols is None:
        return 1.0
    sub = latents_4d[:, cols]
    if mask is not None:
        m = mask.expand(-1, sub.shape[1], -1, -1).bool()
        vals = sub[m]
        if vals.numel() > 0:
            return float(vals.abs().mean().item())
    return float(sub.abs().mean().item())


def shift_channels(latents_4d: torch.Tensor, channel_idxs: List[int], magnitude: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies uniform scalar shift to specified channel indices in 4D tensor (B, 4, H_lat, W_lat).
    mask: None -> whole image; (1, 1, H_lat, W_lat) -> within mask boundary.
    """
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in channel_idxs:
        out[:, c] += magnitude * m if m is not None else magnitude
    return out


def shift_direction(latents_4d: torch.Tensor, direction: List[Tuple[int, float]], magnitude: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def shift_multi_direction(latents_4d: torch.Tensor, direction_magnitude_pairs, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def shift_pca_4d(latents_4d: torch.Tensor, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray,
                 m1: float, m2: float, m3: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Direct steering along top 3 PCA orthonormal eigenvectors (U1, U2, U3) in R^4.
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


# =========================== SDXL SETUP & LATENT MANAGEMENT ===========================
def setup_sdxl(model_id: str = MODEL_ID, device: str = "cuda", dtype=DTYPE, num_latent_channels: int = NUM_LATENT_CHANNELS, use_cpu_offload: bool = False):
    print(f"Loading SDXL Pipeline ({model_id})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True
    )
    if use_cpu_offload and str(device).startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)

    print(f"  vae.config.latent_channels={vae.config.latent_channels} scaling_factor={vae.config.scaling_factor}")
    assert vae.config.latent_channels == num_latent_channels, (
        f"Expected {num_latent_channels} channels, but loaded VAE has {vae.config.latent_channels}"
    )
    return pipe, vae


def latent_hw(height: int, width: int, vae_scale_factor: int = 8) -> Tuple[int, int]:
    return int(height) // vae_scale_factor, int(width) // vae_scale_factor


@torch.no_grad()
def decode_latents_4d(vae: AutoencoderKL, latents_4d: torch.Tensor) -> np.ndarray:
    """
    Decodes (B, 4, H_lat, W_lat) latents in raw units to uint8 numpy RGB image.
    Uses SDXL VAE scaling_factor (0.13025).
    """
    scale = getattr(vae.config, "scaling_factor", 0.13025)
    x = latents_4d.to(torch.float32) / scale
    dec = (vae.decode(x).sample / 2 + 0.5).clamp(0, 1)
    return (dec[0].float().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# =========================== SPATIAL MASK PROJECTION ===========================
def build_mask_latent(mask_pixel: np.ndarray, latent_h: int, latent_w: int, device: str = "cuda", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    mask_pil = Image.fromarray((mask_pixel.astype(np.uint8)) * 255).resize(
        (latent_w, latent_h), Image.NEAREST
    )
    arr = (np.array(mask_pil) > 127).astype(np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)[None, None]


def save_mask_preview(img_uint8: np.ndarray, mask_pixel: np.ndarray, out_path: str):
    overlay = img_uint8.astype(np.float32).copy()
    red = np.array([255.0, 40.0, 40.0])
    overlay[mask_pixel] = 0.45 * overlay[mask_pixel] + 0.55 * red
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


# =========================== SDXL GENERATION PIPELINE RUNNER ===========================
@torch.no_grad()
def run_generation(pipe: StableDiffusionXLPipeline, prompt: str, seed: int,
                   height: int = RESOLUTION, width: int = RESOLUTION,
                   device: str = "cuda", steps: int = STEPS, guidance: float = GUIDANCE,
                   channel_idxs: Optional[List[int]] = None,
                   direction: Optional[List[Tuple[int, float]]] = None,
                   combo=None,
                   magnitude: Optional[float] = None,
                   bands: Optional[List[Tuple[float, float, float, str]]] = None,
                   mask_latent: Optional[torch.Tensor] = None,
                   ref_mode: str = "none",
                   ref_channels: Optional[List[int]] = None,
                   pca_basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                   m_vector: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
    """
    Executes SDXL generation with optional latent steering during the denoising trajectory.
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

        latents = callback_kwargs["latents"]  # (B, 4, H_lat, W_lat)
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

    generator = torch.Generator(device=device).manual_seed(seed) if str(device).startswith("cuda") else torch.Generator().manual_seed(seed)

    output = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
        output_type="latent",
        callback_on_step_end=cb if apply_shift else None,
        callback_on_step_end_tensor_inputs=["latents"] if apply_shift else None,
    )
    return output.images


# =========================== MULTI-GPU PARALLELIZATION ===========================
def get_gpu_list(available_gpus: Optional[List[int]] = None) -> List[str]:
    """
    Returns physical GPU indices respecting parent CUDA_VISIBLE_DEVICES,
    or autodetects free GPUs leaving at least 1 GPU free if all GPUs are free.
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
    if n > 1:
        # Default policy: leave 1 GPU free if unconstrained
        return [str(i) for i in range(n - 1)]
    return ["0"]


def chunk_list(items: list, n_chunks: int) -> list:
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def spawn_workers(tasks: list, n_workers: int, gpu_list: List[str], workers_per_gpu: int, tmp_dir: str, worker_cmd_base: List[str]) -> List[int]:
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
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = worker_cmd_base + ["--chunk", chunk_path]
        print(f"  Worker {i}: Physical GPU {gpu_id}, {len(chunk)} tasks")
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]

"""
FLUX2_CORE.PY -- Core pipeline operations, flow-matching scheduler hooks,
latent pack/unpacking, and schedule envelope generators for FLUX.2.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import subprocess
import numpy as np
import torch
from PIL import Image

try:
    from diffusers import Flux2Pipeline, AutoencoderKLFlux2
except ImportError:
    from diffusers import FluxPipeline as Flux2Pipeline, AutoencoderKL as AutoencoderKLFlux2

DEFAULT_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")


# =========================== SCHEDULE ENVELOPE ===========================
def build_envelope_bands(gate_frac, chrono, mode="ramp_down"):
    """
    Splits [gate_frac, 1.0] into len(chrono) equal bands, each with peak magnitude chrono[i].
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


# =========================== PROFILE GENERATORS ===========================
PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}


def compute_ref(latents_4d, channel_idxs, ref_mode, ref_channels, mask=None):
    if ref_mode == "none":
        return 1.0
    cols = channel_idxs if ref_mode == "self" else ref_channels
    sub = latents_4d[:, cols]
    if mask is not None:
        sub_mask = mask.expand(-1, len(cols), -1, -1)
        return float(sub[sub_mask > 0.5].abs().mean().item()) + 1e-6
    return float(sub.abs().mean().item()) + 1e-6


# =========================== LATENT PACK / UNPACK ===========================
def latent_hw(pipe, height=1024, width=1024):
    scale = getattr(pipe, "vae_scale_factor", 16)
    return int(height) // scale, int(width) // scale


def unpack_to_4d(pipe, latents_3d, height=1024, width=1024):
    """
    Unpacks FLUX.2 patchified latents (B, (H/16)*(W/16), 128) -> (B, 32, H/8, W/8).
    """
    if latents_3d.ndim == 4:
        return latents_3d
    b, s, c_patch = latents_3d.shape
    h_tok = height // 16
    w_tok = width // 16
    latents_4d_patch = latents_3d.permute(0, 2, 1).reshape(b, c_patch, h_tok, w_tok)
    c = c_patch // 4  # 32
    latents = latents_4d_patch.reshape(b, c, 2, 2, h_tok, w_tok)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(b, c, h_tok * 2, w_tok * 2)
    return latents


def pack_from_4d(pipe, latents_4d, height_latent=None, width_latent=None):
    """
    Packs FLUX.2 latents (B, 32, H/8, W/8) -> patchified (B, (H/16)*(W/16), 128).
    """
    if latents_4d.ndim == 3:
        return latents_4d
    b, c, h_lat, w_lat = latents_4d.shape
    h_tok = h_lat // 2
    w_tok = w_lat // 2
    latents = latents_4d.reshape(b, c, h_tok, 2, w_tok, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(b, c * 4, h_tok, w_tok)
    latents_3d = latents.reshape(b, c * 4, h_tok * w_tok).permute(0, 2, 1)
    return latents_3d


def decode_latents_4d(vae, latents_4d, device=None):
    """
    Decodes FLUX.2 (B, 32, H/8, W/8) into uint8 RGB numpy array (H_pixel, W_pixel, 3).
    """
    with torch.no_grad():
        if device is None:
            if hasattr(vae, "_hf_hook") and hasattr(vae._hf_hook, "execution_device") and vae._hf_hook.execution_device is not None:
                device = vae._hf_hook.execution_device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        z = latents_4d.to(device=device, dtype=vae.dtype)
        if hasattr(vae, "bn") and hasattr(vae.bn, "running_mean"):
            # vae.bn operates on patchified latents (B, 128, H/16, W/16)
            b, c, h_lat, w_lat = z.shape
            h_tok = h_lat // 2
            w_tok = w_lat // 2
            z_patch = z.reshape(b, c, h_tok, 2, w_tok, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h_tok, w_tok)
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=z.dtype)
            bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + getattr(vae.config, "batch_norm_eps", 1e-5)).to(device=device, dtype=z.dtype)
            z_patch = z_patch * bn_std + bn_mean
            z = z_patch.reshape(b, c, 2, 2, h_tok, w_tok).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h_tok * 2, w_tok * 2)
        elif hasattr(vae.config, "scaling_factor"):
            scale = getattr(vae.config, "scaling_factor", 0.3611)
            shift = getattr(vae.config, "shift_factor", 0.0609)
            z = (z / scale) + shift

        img = vae.decode(z, return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        arr = img[0].permute(1, 2, 0).cpu().float().numpy()
        return (arr * 255).round().astype(np.uint8)


def build_mask_latent(mask_pixel_np, latent_h, latent_w, device):
    """
    Converts a 2D boolean mask into a latent tensor of shape (1, 1, latent_h, latent_w).
    """
    pil = Image.fromarray((mask_pixel_np.astype(np.uint8)) * 255).resize(
        (latent_w, latent_h), resample=Image.NEAREST
    )
    t = torch.from_numpy(np.array(pil) > 127).float().unsqueeze(0).unsqueeze(0).to(device)
    return t


def save_mask_preview(img_rgb_np, mask_pixel_np, out_path):
    overlay = img_rgb_np.copy().astype(np.float32)
    red = np.array([255.0, 40.0, 40.0])
    overlay[mask_pixel_np] = 0.5 * overlay[mask_pixel_np] + 0.5 * red
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


def shift_channels(latents_4d, channel_idxs, magnitude, mask=None):
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in channel_idxs:
        out[:, c] += magnitude * m if m is not None else magnitude
    return out


def shift_direction(latents_4d, direction, magnitude, mask=None):
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    norm = float(np.sqrt(sum(sign * sign for _, sign in direction))) or 1.0
    for c, sign in direction:
        delta = magnitude * (sign / norm)
        out[:, c] += delta * m if m is not None else delta
    return out


def shift_multi_direction(latents_4d, direction_magnitude_pairs, mask=None):
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for direction, magnitude in direction_magnitude_pairs:
        norm = float(np.sqrt(sum(sign * sign for _, sign in direction))) or 1.0
        for c, sign in direction:
            delta = magnitude * (sign / norm)
            out[:, c] += delta * m if m is not None else delta
    return out


@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps, guidance,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None):
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
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="latent",
        callback_on_step_end=cb if apply_shift else None,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images
    return latents_packed


# =========================== PIPELINE SETUP ===========================
def resolve_local_model_path(model_id):
    if os.path.exists(model_id):
        return model_id

    repo_folder = f"models--{model_id.replace('/', '--')}"
    candidate_cache_dirs = []
    if "HF_HOME" in os.environ:
        candidate_cache_dirs.append(os.environ["HF_HOME"])
    candidate_cache_dirs.append(os.path.expanduser("~/.cache/huggingface"))

    for hf_home in candidate_cache_dirs:
        hub_dir = os.path.join(hf_home, "hub")
        repo_path = os.path.join(hub_dir, repo_folder)
        if os.path.exists(repo_path):
            snapshots_dir = os.path.join(repo_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snaps = [
                    os.path.join(snapshots_dir, s)
                    for s in os.listdir(snapshots_dir)
                    if not s.startswith(".") and os.path.isfile(os.path.join(snapshots_dir, s, "model_index.json"))
                ]
                if snaps:
                    snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return snaps[0]
    return model_id


def setup_flux(model_id=None, device="cuda", dtype=torch.bfloat16, num_latent_channels=32):
    """
    Loads FLUX.2 pipeline components in bfloat16 directly from local cache or repo ID.
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    resolved_path = resolve_local_model_path(model_id)
    print(f"Loading Pipeline from: {resolved_path} on {device} [{dtype}]...")

    pipe = None
    from diffusers import Flux2Pipeline

    # 1. Try local snapshot path first if it exists
    if os.path.exists(resolved_path):
        try:
            pipe = Flux2Pipeline.from_pretrained(resolved_path, torch_dtype=dtype, local_files_only=True)
        except Exception as e_local:
            print(f"Notice: Loading Flux2Pipeline with local_files_only from {resolved_path} failed: {e_local}")

    # 2. Fall back to loading via model_id / hub (uses HF cache or downloads missing shards)
    if pipe is None:
        try:
            pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        except Exception as e_hub:
            print(f"Flux2Pipeline.from_pretrained({model_id}) failed: {e_hub}. Trying AutoPipelineForText2Image...")
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                resolved_path if os.path.exists(resolved_path) else model_id,
                torch_dtype=dtype
            )

    # Offload configuration
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        dev_idx = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        except Exception:
            free_bytes = 0

        # FLUX.2 transformer alone is ~61GB in bfloat16. If free VRAM < 65GB, model_cpu_offload will OOM.
        if free_bytes < 65 * (1024**3):
            free_gb = free_bytes / (1024**3)
            print(f"Notice: Free VRAM on cuda:{dev_idx} is {free_gb:.1f} GB (< 65 GB required for full-model residency).")
            print("Enabling sequential CPU offload (layer-by-layer offload to prevent CUDA OOM)...")
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            try:
                pipe.enable_model_cpu_offload(device=device)
                print("Enabled full-speed model CPU offload (32B transformer in VRAM during denoising).")
            except Exception as e_offload:
                print(f"Model CPU offload notice: {e_offload}. Enabling sequential CPU offload...")
                pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    vae = pipe.vae
    return pipe, vae


# =========================== WORKER & GPU UTILITIES ===========================
def get_gpu_list(available_gpus=None):
    if available_gpus is not None:
        return [str(g) for g in available_gpus]
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        physical = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
        if physical:
            return physical
    n = torch.cuda.device_count()
    if n == 0:
        return ["0"]
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
        print(f"  worker {i}: GPU {gpu_id}, {len(chunk)} tasks")
        procs.append(subprocess.Popen(cmd, env=env))
    return [p.wait() for p in procs]

"""
FLUX_CORE.PY -- mecanismo de bajo nivel de FLUX, compartido entre
explorar_canales_flux.py y fase_a_screening_flux.py (y cualquier script
nuevo que se sume despues, ej. la Fase C de calibracion).

NO tiene CONFIG propio ni orquestacion (nada de main(), nada de
REFERENCE_OBJECTS/MAGNITUDES) -- eso es responsabilidad de CADA script,
que sigue siendo corrible solo (`python explorar_canales_flux.py`,
`python fase_a_screening_flux.py`). Ac vive SOLO lo que antes estaba
duplicado literal entre los dos: pack/unpack del latente, decode, aplicar
el shift, el schedule con rampas, run_generation, y la paralelizacion
multi-GPU. Por eso todas las funciones reciben sus parametros explicitos
(gate_frac, steps, resolution, etc.) en vez de leerlos de variables
globales del modulo que las llama -- separado de un script, no puede
"adivinar" el CONFIG de otro.

Motivo del split (para que quede registrado): antes cada script tenia su
propia copia de esta mecanica. Cuando aparecio el bug de
CUDA_VISIBLE_DEVICES (pisaba la restriccion del proceso padre), hubo que
arreglarlo en los dos lugares a mano -- exactamente el tipo de error que
este archivo evita de ac en mas.
"""

import os
import json
import subprocess
import numpy as np
import torch
from diffusers import FluxPipeline, AutoencoderKL
from PIL import Image


# =========================== schedule (rampas dentro de bandas) ===========================
def build_envelope_bands(gate_frac, chrono, mode="ramp_down"):
    """[gate_frac, 1.0] partido en len(chrono) bandas iguales, cada una
    con su techo de magnitud (chrono[i], en orden lejos->cerca porque
    fraccion creciente = mas cerca del final del denoising)."""
    n = len(chrono)
    edges = np.linspace(gate_frac, 1.0, n + 1)
    return [(float(edges[i]), float(edges[i + 1]), chrono[i], mode) for i in range(n)]


def shift_schedule(frac, lo, hi, peak, mode="ramp_down"):
    if frac < lo or frac > hi:
        return 0.0
    p = (frac - lo) / max(hi - lo, 1e-8)   # 0 al abrir esta banda, 1 al cerrarla
    p = float(min(max(p, 0.0), 1.0))
    w = (1.0 - p) if mode == "ramp_down" else 1.0
    return peak * w


def envelope_weight(frac, bands):
    return sum(shift_schedule(frac, lo, hi, peak, mode) for lo, hi, peak, mode in bands)


# =========================== generadores de perfil (forma del envelope) ===========================
# Cada uno recibe n_partes y devuelve la lista de picos por banda (chrono),
# normalizada a techo 1.0 (el techo real lo pone la dosis, no esta lista).
# Vive ac (no en cada script) porque fase_b_config.py (la busca) y
# fase_c_*.py (la usa ya fija) necesitan reconstruir el MISMO chrono a
# partir de un nombre de perfil -- si viviera duplicado en cada script,
# un cambio en uno y no en el otro haria que el schedule "confirmado" de
# Fase B no sea reproducible exactamente en Fase C.
PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}


# =========================== escala del shift (ref) ===========================
def compute_ref(latents_4d, channel_idxs, ref_mode, ref_channels, mask=None):
    """ref_mode: 'none' (magnitud absoluta) | 'self' (escala por los
    propios canales que se shiftean) | 'channels' (escala por un grupo
    fijo, ref_channels). Con mask: promedia SOLO dentro del objeto."""
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
    """mask: None -> shift a TODA la imagen. (1,1,H_lat,W_lat) 0/1 ->
    shift SOLO donde mask==1 (el objeto segmentado)."""
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in channel_idxs:
        out[:, c] += magnitude * m if m is not None else magnitude
    return out


def shift_direction(latents_4d, direction, magnitude, mask=None):
    """direction: lista de (channel_idx, signo +1/-1) -- una direccion
    COMBINADA (varios canales agrupados en Fase B, algunos con signo
    invertido, ej. el eje 'a' = [(2,+1),(11,+1),(14,+1),(5,-1),(10,-1),
    (13,-1)]). El vector de signos se NORMALIZA a norma unitaria antes de
    aplicar la magnitud (norma = sqrt(cantidad de canales), ya que cada
    signo es +-1) -- sin esto, una direccion de 6 canales empuja 6x mas
    fuerte que una de 2 canales con la MISMA magnitude, haciendo que
    comparar 'm' entre direcciones de distinto tamaño (ej. en el ranking
    de fase_b_config.py) no sea una comparacion justa. Con
    normalizacion, 'magnitude' representa "un paso del mismo tamaño"
    sin importar cuantos canales tenga el grupo."""
    out = latents_4d.clone()
    m = mask[:, 0].to(out.dtype) if mask is not None else None
    norm = float(np.sqrt(sum(sign * sign for _, sign in direction))) or 1.0
    for c, sign in direction:
        delta = magnitude * (sign / norm)
        out[:, c] += delta * m if m is not None else delta
    return out


def shift_multi_direction(latents_4d, direction_magnitude_pairs, mask=None):
    """Aplica VARIAS direcciones a la vez, cada una con su PROPIA
    magnitud -- para Fase C parte 2 (combinaciones L+a+b simultaneas, cada
    eje con la magnitud que le calibro parte 1). Simplemente suma el
    shift_direction de cada par sobre el MISMO latente -- los canales no
    se pisan entre direcciones distintas (cada direccion confirmada usa
    su propio subconjunto de canales, ver GRUPOS_CONFIRMADOS), asi que
    aplicar todas juntas es una suma directa, no hace falta resolver
    conflictos."""
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
    print(f"Cargando FLUX ({model_id})...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    # VAE aparte en float32 para decode limpio.
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
    print(f"  vae.config.latent_channels={vae.config.latent_channels} "
         f"scaling_factor={vae.config.scaling_factor} shift_factor={vae.config.shift_factor}")
    assert vae.config.latent_channels == num_latent_channels, (
        f"esperaba {num_latent_channels} canales, el VAE cargado tiene {vae.config.latent_channels}")
    return pipe, vae


def latent_hw(pipe, height, width):
    """Alto/ancho del latente YA desempaquetado (antes de parches 2x2) --
    lo que espera pipe._pack_latents. Distinto de lo que espera
    pipe._unpack_latents, que quiere pixeles crudos."""
    vsf = pipe.vae_scale_factor
    return 2 * (int(height) // (vsf * 2)), 2 * (int(width) // (vsf * 2))


def unpack_to_4d(pipe, latents_packed, height, width):
    return pipe._unpack_latents(latents_packed, height, width, pipe.vae_scale_factor)


def pack_from_4d(pipe, latents_4d, latent_h, latent_w):
    b, c = latents_4d.shape[0], latents_4d.shape[1]
    return pipe._pack_latents(latents_4d, b, c, latent_h, latent_w)


@torch.no_grad()
def decode_latents_4d(vae, latents_4d):
    """(B,16,H_lat,W_lat) YA en unidades de latente crudo -> imagen uint8.
    Aplica scaling_factor+shift_factor (los DOS -- FLUX no es solo
    scaling_factor como SDXL)."""
    x = latents_4d.to(torch.float32) / vae.config.scaling_factor + vae.config.shift_factor
    dec = (vae.decode(x).sample / 2 + 0.5).clamp(0, 1)
    return (dec[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# =========================== mascara: pixeles -> espacio latente ===========================
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


# =========================== generacion, con o sin shift durante el denoising ===========================
@torch.no_grad()
def run_generation(pipe, prompt, seed, height, width, device, steps, guidance,
                   channel_idxs=None, direction=None, combo=None, magnitude=None, bands=None,
                   mask_latent=None, ref_mode="none", ref_channels=None):
    """Sin nada de shift + magnitude + bands: genera limpio (baseline).
    channel_idxs: shift uniforme (exploracion/Fase A). direction: UNA
    direccion combinada con signo por canal (Fase B en adelante).
    combo: VARIAS direcciones a la vez, cada una con su PROPIA magnitud
    -- lista de (direction, magnitude), ignora el parametro 'magnitude'
    (Fase C parte 2, L+a+b simultaneos). Mutuamente excluyentes entre si.
    Devuelve el latente final EMPAQUETADO."""
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


# =========================== paralelizacion multi-GPU (subprocesos reales) ===========================
def get_gpu_list(available_gpus):
    """Devuelve los IDs FISICOS reales a poner en el CUDA_VISIBLE_DEVICES
    de cada worker -- si el proceso PADRE ya corrio con un
    CUDA_VISIBLE_DEVICES restringido, respeta ESA lista de fisicas en vez
    de reemplazarla por 0..n-1 (bug real, ya corregido una vez -- ver
    docstring del modulo)."""
    if available_gpus is not None:
        return [str(g) for g in available_gpus]
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_cvd:
        physical = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
        if physical:
            return physical
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No se detecto ninguna GPU -- setea AVAILABLE_GPUS a mano o corre sin --parallel.")
    return [str(i) for i in range(n)]


def chunk_list(items, n_chunks):
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def spawn_workers(tasks, n_workers, gpu_list, workers_per_gpu, tmp_dir, worker_cmd_base):
    """Reparte `tasks` (lista de dicts serializables a JSON) en chunks, uno
    por worker, y lanza un subproceso REAL por chunk con
    CUDA_VISIBLE_DEVICES fijado a una GPU fisica. worker_cmd_base: lista
    de argv SIN --chunk (se le agrega ac, uno distinto por proceso).
    Devuelve la lista de exit codes."""
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
"""
INFERENCE.PY -- Targeted Object Color Steering on FLUX.1-dev.

Applies the paper's downstream application method:
1. Takes a text prompt describing a scene/object and a target color (Hex, RGB, or CIELAB).
2. Uses SAM3 to segment the target object at the gate step during denoising.
3. Measures the object's initial color in CIELAB space.
4. Uses the trained MLPShiftPCA to predict the optimal latent shift (m1, m2, m3).
5. Injects the spatial latent perturbation via the calibrated temporal schedule.
6. Decodes the final image where the targeted object achieves the exact desired color.

Supports batch processing of multiple prompts, objects, and colors efficiently
(models are loaded once and shared across all inferences).
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add current Flux directory to path
FLUX_DIR = Path(__file__).resolve().parent
if str(FLUX_DIR) not in sys.path:
    sys.path.insert(0, str(FLUX_DIR))

from flux_core import (
    build_envelope_bands,
    envelope_weight,
    latent_hw,
    unpack_to_4d,
    pack_from_4d,
    decode_latents_4d,
    build_mask_latent,
    PERFIL_GENERATORS,
    setup_flux,
)
import utils
from model_pca import load_mlp_pca

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5
DEFAULT_PROMPT = "a photo of a dog at color #A52A2A doing skateboarding in the park"
DEFAULT_GATE_FRAC = 0.5

PCA_AXES_PATH = FLUX_DIR / "fase_a_pca_out" / "pca_axes.json"
WINNING_SCHEDULE_PATH = FLUX_DIR / "fase_b_pca_out" / "fase_b_winning_schedule.json"
DEFAULT_CKPT_PATH = FLUX_DIR / "mlp_training_out" / "mlp_shift_pca_best.pt"

extract_hex_from_text = utils.extract_hex_from_text
extract_color_spec_from_text = utils.extract_color_spec_from_text
extract_object_from_text = utils.extract_object_from_text

# Global cache for pre-loaded models to avoid reloading when called multiple times
_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}


def load_pca_basis(axes_path=PCA_AXES_PATH):
    if os.path.exists(axes_path):
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return u1, u2, u3


def load_winning_schedule(sched_path=WINNING_SCHEDULE_PATH):
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            return json.load(f)
    return {"gate_frac": DEFAULT_GATE_FRAC, "n_partes": 1, "perfil_name": "ascendente"}


def shift_pca_4d(
    latents_4d: torch.Tensor,
    m1: float,
    m2: float,
    m3: float,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    out = latents_4d.clone()
    m_mask = mask[:, 0].to(out.dtype) if mask is not None else None
    num_ch = min(out.shape[1], len(u1))
    for c in range(num_ch):
        delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c])
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


def compute_color_metrics(
    target_lab: Tuple[float, float, float],
    measured_lab: Tuple[float, float, float]
) -> Dict[str, Any]:
    L_t, a_t, b_t = target_lab
    L_m, a_m, b_m = measured_lab

    # Delta E CIEDE2000 & CIE76
    de00 = utils.ciede2000(target_lab, measured_lab)
    de76 = math.sqrt((L_m - L_t) ** 2 + (a_m - a_t) ** 2 + (b_m - b_t) ** 2)

    # Chroma
    c_t = math.sqrt(a_t ** 2 + b_t ** 2)
    c_m = math.sqrt(a_m ** 2 + b_m ** 2)
    d_chroma = c_m - c_t
    d_chroma_abs = abs(d_chroma)

    # Hue (degrees [0, 360))
    h_t = math.degrees(math.atan2(b_t, a_t)) % 360.0
    h_m = math.degrees(math.atan2(b_m, a_m)) % 360.0

    # Angular Delta Hue (degrees [-180, 180])
    d_h_deg = ((h_m - h_t + 180.0) % 360.0) - 180.0
    d_h_abs_deg = abs(d_h_deg)

    # Metric Delta Hue (Delta H*ab)
    d_H_ab = 2.0 * math.sqrt(max(c_t * c_m, 0.0)) * math.sin(math.radians(d_h_deg) / 2.0)
    d_H_ab_abs = abs(d_H_ab)

    return {
        "delta_e00": de00,
        "delta_e76": de76,
        "delta_chroma": d_chroma,
        "delta_chroma_abs": d_chroma_abs,
        "delta_hue_deg": d_h_deg,
        "delta_hue_abs_deg": d_h_abs_deg,
        "delta_H_ab": d_H_ab,
        "delta_H_ab_abs": d_H_ab_abs,
    }


def extract_base_object(prompt: str) -> Optional[str]:
    """
    Extracts the base object description from a prompt using flexible patterns.
    """
    clean_text = prompt.strip()

    # Pattern 1: GenColorBench NCU: "A photo of a <OBJ> in the color / at color..."
    match = re.search(
        r"(?:photo|rendering|picture|image|shot|drawing|illustration)?\s*(?:of)?\s*\b(?:an|a|the)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:in\s+(?:the\s+)?color|with\s+(?:the\s+)?color|colored\s+(?:with|in|as)?|at\s+color|having\s+(?:the\s+)?color|painted\s+in)\b",
        clean_text,
        re.IGNORECASE,
    )
    if match:
        cand = match.group(1).strip()
        cand = re.sub(r"^(?:photo|picture|image|rendering)\s+of\s+(?:an|a|the\s+)?", "", cand, flags=re.IGNORECASE).strip()
        if cand:
            return cand

    # Pattern 2: "photo of a/an <OBJ> on/in/at/with/by..."
    match = re.search(
        r"(?:photo|picture|image|rendering)\s+of\s+\b(?:an|a|the)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:on|in|at|with|by|under|near|for|doing|standing|sitting|laying|resting|walking|running|\.|\,|$))",
        clean_text,
        re.IGNORECASE,
    )
    if match:
        cand = match.group(1).strip()
        if cand:
            return cand

    # Pattern 3: direct "a/an/the <OBJ> on/in/at..." (when no 'photo of')
    match = re.search(
        r"\b(?:an|a|the)\s+([a-zA-Z0-9_\-\s]+?)(?:\s+(?:on|in|at|with|by|under|near|for|doing|standing|sitting|laying|resting|walking|running|\.|\,|$))",
        clean_text,
        re.IGNORECASE,
    )
    if match:
        cand = match.group(1).strip()
        if cand and cand.lower() not in {"photo", "picture", "image", "rendering"}:
            return cand

    return utils.extract_object_from_text(prompt)


def replace_object_in_prompt(prompt: str, new_obj: str, old_obj: Optional[str] = None) -> str:
    """
    Substitutes an object inside a text prompt while preserving sentence structure and grammar.
    Handles 'a' vs 'an' article adjustments based on the new word.
    """
    if not new_obj or not prompt:
        return prompt
    new_obj = new_obj.strip()

    candidate_old_objs: List[str] = []
    if old_obj and old_obj.strip():
        candidate_old_objs.append(old_obj.strip())

    extracted = extract_base_object(prompt)
    if extracted and extracted not in candidate_old_objs:
        candidate_old_objs.append(extracted)

    # Also consider noun-only parts (e.g. 'ceramic mug' -> try 'ceramic mug' then 'mug')
    for cand in list(candidate_old_objs):
        parts = cand.split()
        if len(parts) > 1 and parts[-1] not in candidate_old_objs:
            candidate_old_objs.append(parts[-1])

    replaced = False
    new_prompt = prompt
    for cand in candidate_old_objs:
        pattern = re.compile(rf"\b{re.escape(cand)}\b", flags=re.IGNORECASE)
        if pattern.search(new_prompt):
            new_prompt = pattern.sub(new_obj, new_prompt, count=1)
            replaced = True
            break

    if not replaced:
        pattern_photo = re.compile(
            r"(\b(?:photo|picture|image|rendering)\s+of\s+(?:an|a|the)\s+)([a-zA-Z0-9_\-]+)",
            flags=re.IGNORECASE,
        )
        if pattern_photo.search(new_prompt):
            new_prompt = pattern_photo.sub(rf"\g<1>{new_obj}", new_prompt, count=1)
            replaced = True
        else:
            new_prompt = f"{prompt} with a {new_obj}"

    # Fix English indefinite articles a/an before vowel/consonant sounds
    new_prompt = re.sub(r"\b(a)\s+([aeiouAEIOU]\w*)", r"an \2", new_prompt)
    new_prompt = re.sub(r"\b(A)\s+([aeiouAEIOU]\w*)", r"An \2", new_prompt)
    new_prompt = re.sub(r"\b(an)\s+([^aeiouAEIOU\s]\w*)", r"a \2", new_prompt)
    new_prompt = re.sub(r"\b(An)\s+([^aeiouAEIOU\s]\w*)", r"A \2", new_prompt)

    return new_prompt


def replace_color_in_prompt(prompt: str, new_color_spec: str) -> str:
    """
    Substitutes an existing color specification or color name in a prompt with the new color.
    If the prompt has no color, leaves it untouched.
    """
    if not new_color_spec or not prompt:
        return prompt
    new_color_spec = str(new_color_spec).strip()

    # 1. Numeric color spec (hex, rgb, lab)
    old_spec = utils.extract_color_spec_from_text(prompt)
    if old_spec:
        return prompt.replace(old_spec, new_color_spec, 1)

    # 2. Check for hex pattern
    match_hex = re.search(r"#[0-9a-fA-F]{3,8}\b", prompt)
    if match_hex:
        return prompt[:match_hex.start()] + new_color_spec + prompt[match_hex.end():]

    # 3. Known color names
    try:
        lab = utils.parse_target_color(new_color_spec)
        new_name, _ = utils.nearest_color_name(lab)
    except Exception:
        new_name = new_color_spec

    for name in utils.CURATED_COLOR_NAMES:
        pat = re.compile(rf"\b{re.escape(name)}\b", flags=re.IGNORECASE)
        if pat.search(prompt):
            return pat.sub(new_name, prompt, count=1)

    return prompt


def normalize_color_spec(spec: Any) -> Any:
    if spec is None:
        return None
    s = str(spec).strip()
    if re.fullmatch(r"[0-9a-fA-F]{6}", s) or re.fullmatch(r"[0-9a-fA-F]{3}", s):
        return f"#{s.upper()}"
    return s


def flatten_list(items: Optional[List[Any]]) -> List[Any]:
    if items is None:
        return []
    res = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, str) and "," in it:
            sub = [s.strip() for s in it.split(",") if s.strip()]
            res.extend(sub)
        else:
            res.append(it)
    return res


def build_inference_tasks(
    prompts: List[str],
    colors: List[Any],
    objects: List[Optional[str]],
    base_object: Optional[str] = None,
    cartesian: bool = False,
) -> List[Dict[str, Any]]:
    """
    Constructs the list of inference tasks based on inputs:
    - Multiple objects, 1 prompt, 1 color: repeats same prompt & color with object swapped in prompt.
    - Multiple colors, 1 prompt, 1 object: repeats same prompt & object with color updated.
    - Everything different: pairs prompts, colors, and objects 1:1.
    - Cartesian mode (--cartesian): generates full Cartesian product (prompts x objects x colors).
    """
    flat_p = flatten_list([p for p in prompts if p]) if prompts else []
    prompts = flat_p if flat_p else [DEFAULT_PROMPT]
    flat_c = flatten_list(list(colors)) if colors else []
    colors = [normalize_color_spec(c) for c in flat_c] if flat_c else [None]
    flat_o = flatten_list(list(objects)) if objects else []
    objects = flat_o if flat_o else [None]

    P, C, O = len(prompts), len(colors), len(objects)

    if cartesian:
        tasks = []
        for p in prompts:
            for o in objects:
                p_curr = replace_object_in_prompt(p, o, old_obj=base_object) if (p and o) else p
                for c in colors:
                    p_final = replace_color_in_prompt(p_curr, c) if (p_curr and c) else p_curr
                    tasks.append({"prompt": p_final, "color": c, "object": o})
        return tasks

    N = max(P, C, O)
    if N == 1:
        return [{"prompt": prompts[0], "color": colors[0], "object": objects[0]}]

    # Case 1: 1 prompt, 1 color, multiple objects -> swap object in prompt
    if P == 1 and C == 1 and O > 1:
        base_p = prompts[0]
        base_c = colors[0]
        detected_base = base_object
        if not detected_base:
            for o in objects:
                if o and re.search(rf"\b{re.escape(o)}\b", base_p, re.IGNORECASE):
                    detected_base = o
                    break
            if not detected_base:
                detected_base = extract_base_object(base_p) or utils.extract_object_from_text(base_p)
        tasks = []
        for o in objects:
            p_obj = replace_object_in_prompt(base_p, o, old_obj=detected_base)
            if base_c:
                p_obj = replace_color_in_prompt(p_obj, base_c)
            tasks.append({"prompt": p_obj, "color": base_c, "object": o})
        return tasks

    # Case 2: 1 prompt, multiple colors, 1 object -> same prompt & object, multiple colors
    if P == 1 and C > 1 and O == 1:
        base_p = prompts[0]
        base_o = objects[0]
        tasks = []
        for c in colors:
            p_col = replace_color_in_prompt(base_p, c)
            tasks.append({"prompt": p_col, "color": c, "object": base_o})
        return tasks

    # Case 3: 1 prompt, multiple colors and objects (paired)
    if P == 1 and C == O and C > 1:
        base_p = prompts[0]
        detected_base = base_object
        if not detected_base:
            for o in objects:
                if o and re.search(rf"\b{re.escape(o)}\b", base_p, re.IGNORECASE):
                    detected_base = o
                    break
            if not detected_base:
                detected_base = extract_base_object(base_p) or utils.extract_object_from_text(base_p)
        tasks = []
        for i in range(C):
            p_curr = replace_object_in_prompt(base_p, objects[i], old_obj=detected_base)
            p_curr = replace_color_in_prompt(p_curr, colors[i])
            tasks.append({"prompt": p_curr, "color": colors[i], "object": objects[i]})
        return tasks

    # Case 4: Multiple prompts, 1 color, 1 object
    if P > 1 and C == 1 and O == 1:
        base_c = colors[0]
        tasks = []
        for p in prompts:
            p_final = replace_color_in_prompt(p, base_c) if base_c else p
            tasks.append({"prompt": p_final, "color": base_c, "object": objects[0]})
        return tasks

    # Case 5: Multiple prompts, multiple colors, 1 object (P == C)
    if P == C and O == 1:
        tasks = []
        for i in range(P):
            p_final = replace_color_in_prompt(prompts[i], colors[i]) if colors[i] else prompts[i]
            tasks.append({"prompt": p_final, "color": colors[i], "object": objects[0]})
        return tasks

    # Case 6: Multiple prompts, 1 color, multiple objects (P == O)
    if P == O and C == 1:
        base_c = colors[0]
        tasks = []
        for i in range(P):
            p_final = replace_color_in_prompt(prompts[i], base_c) if base_c else prompts[i]
            tasks.append({"prompt": p_final, "color": base_c, "object": objects[i]})
        return tasks

    # Case 7: All different (P == C == O)
    if P == C == O:
        tasks = []
        for i in range(P):
            p_final = replace_color_in_prompt(prompts[i], colors[i]) if colors[i] else prompts[i]
            tasks.append({"prompt": p_final, "color": colors[i], "object": objects[i]})
        return tasks

    # Fallback to Cartesian product if dimensions do not directly pair
    print(f"[Warning] Provided list sizes (prompts={P}, colors={C}, objects={O}) do not match 1:1 broadcast. Falling back to Cartesian product.")
    tasks = []
    for p in prompts:
        for o in objects:
            p_curr = replace_object_in_prompt(p, o, old_obj=base_object) if (p and o) else p
            for c in colors:
                p_final = replace_color_in_prompt(p_curr, c) if (p_curr and c) else p_curr
                tasks.append({"prompt": p_final, "color": c, "object": o})
    return tasks


def load_color_steering_models(
    device: str = "cuda:0",
    mlp_ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads FLUX.1 pipeline, VAE, SAM3 segmenter, and MLP PCA model once.
    Returns a dictionary of shared models for reuse across inferences.
    """
    print("=" * 80)
    print("LOADING FLUX.1 MODELS & DEPENDENCIES (ONE-TIME SETUP)")
    print(f"Device: {device} | Dtype: {DTYPE} | Channels: {NUM_LATENT_CHANNELS}")
    print("=" * 80)
    pipe, vae = setup_flux(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)

    ckpt_file = mlp_ckpt_path or str(DEFAULT_CKPT_PATH)
    mlp_shift = load_mlp_pca(ckpt_file, device)

    u1, u2, u3 = load_pca_basis()
    sched_cfg = load_winning_schedule()

    bands = build_envelope_bands(
        sched_cfg.get("gate_frac", DEFAULT_GATE_FRAC),
        PERFIL_GENERATORS[sched_cfg.get("perfil_name", "ascendente")](sched_cfg.get("n_partes", 1)),
        "ramp_down"
    )

    return {
        "pipe": pipe,
        "vae": vae,
        "seg_models": seg_models,
        "mlp_shift": mlp_shift,
        "u_basis": (u1, u2, u3),
        "sched_cfg": sched_cfg,
        "bands": bands,
    }


def get_or_load_models(
    device: str = "cuda:0",
    mlp_ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    key = (device, mlp_ckpt_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_color_steering_models(device=device, mlp_ckpt_path=mlp_ckpt_path)
    return _MODEL_CACHE[key]


@torch.inference_mode()
def run_color_steering_inference(
    prompt: str,
    target_color_spec: Any = None,
    object_word: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "./inference_outputs",
    mlp_ckpt_path: Optional[str] = None,
    metrics: bool = False,
    models: Optional[Dict[str, Any]] = None,
    task_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a single color-steered generation. If 'models' dictionary is provided,
    it reuses the preloaded pipeline, avoiding expensive model re-initializations.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Parse Target Color & Object using utils
    if target_color_spec is None:
        color_in_prompt = utils.extract_color_spec_from_text(prompt)
        target_color_spec = color_in_prompt or "#A52A2A"

    if object_word is None or str(object_word).strip() == "":
        object_word = extract_base_object(prompt) or utils.extract_object_from_text(prompt)

    target_lab = utils.parse_target_color(target_color_spec)
    target_rgb = utils.lab_to_rgb_single_np(target_lab)
    target_hex = f"#{target_rgb[0]:02X}{target_rgb[1]:02X}{target_rgb[2]:02X}"
    nearest_name, _ = utils.nearest_color_name(target_lab)

    clean_prompt = utils.clean_prompt_for_diffusion(prompt, nearest_name)

    print("=" * 80)
    print("RUNNING FLUX.1 DOWNSTREAM OBJECT COLOR STEERING")
    print(f"Raw Prompt:       \"{prompt}\"")
    print(f"Target Object:    \"{object_word}\"")
    print(f"Target Color:     {target_hex} (RGB: {target_rgb}) -> CIELAB: L*={target_lab[0]:.1f}, a*={target_lab[1]:.1f}, b*={target_lab[2]:.1f} (nearest ISCC-NBS L2: '{nearest_name}')")
    print(f"Semantic Prompt:  \"{clean_prompt}\"")
    print(f"Seed: {seed} | Steps: {steps} | Guidance: {guidance} | Res: {resolution}x{resolution} | Device: {device}")
    print("=" * 80)

    # 2. Get or reuse pre-loaded models (NO RELOADING)
    if models is None:
        models = get_or_load_models(device=device, mlp_ckpt_path=mlp_ckpt_path)

    pipe = models["pipe"]
    vae = models["vae"]
    seg_models = models["seg_models"]
    mlp_shift = models["mlp_shift"]
    u1, u2, u3 = models["u_basis"]
    sched_cfg = models["sched_cfg"]
    bands = models["bands"]

    latent_h, latent_w = latent_hw(pipe, resolution, resolution)

    print(f"\nGenerating color-steered image with target {target_hex}...")
    state = {
        "locked": False, "mask_latent": None, "mask_pixel": None,
        "init_lab": None, "m_pred": None, "img_x0": None, "failed": False, "reason": None
    }
    captured = {}
    orig_step = pipe.scheduler.step

    def patched_step(model_output, timestep, sample, *a, **k):
        if not state["locked"]:
            try:
                sched = pipe.scheduler
                idx = sched.index_for_timestep(timestep) if hasattr(sched, "index_for_timestep") else getattr(sched, "_step_index", None)
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

            pred_x0 = captured.get("pending_x0", lat)
            img_x0 = decode_latents_4d(vae, unpack_to_4d(pipe_, pred_x0, resolution, resolution))
            state["img_x0"] = img_x0

            mask_pixel = utils.get_object_mask(Image.fromarray(img_x0), object_word, seg_models)

            max_gate_step = int(steps * max(sched_cfg.get("gate_frac", DEFAULT_GATE_FRAC) * 0.65, 0.40))
            if mask_pixel is None or mask_pixel.sum() < 50:
                if step_index < max_gate_step:
                    return callback_kwargs
                print(f"  [Warning] Mask detection for '{object_word}' yielded low pixel count after retries. Using spatial center prior fallback.")
                h_img, w_img = img_x0.shape[:2]
                yy, xx = np.ogrid[:h_img, :w_img]
                mask_pixel = ((xx - w_img / 2) ** 2 + (yy - h_img / 2) ** 2) <= (min(h_img, w_img) * 0.35) ** 2

            state["locked"] = True

            init_lab = utils.measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Initial color measurement failed"
                return callback_kwargs

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            m_pred = mlp_shift.predict_m(init_lab, target_lab)
            state["m_pred"] = m_pred
            print(f"  [Gate Triggered @ Step {step_index}] Initial Lab: L*={init_lab[0]:.1f}, a*={init_lab[1]:.1f}, b*={init_lab[2]:.1f}")
            print(f"  [MLP Predicted Shift]: m1={m_pred[0]:+.4f}, m2={m_pred[1]:+.4f}, m3={m_pred[2]:+.4f}")

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return callback_kwargs

        unpacked = unpack_to_4d(pipe_, lat, resolution, resolution)
        m = state["mask_latent"]
        if m.shape[-2:] != unpacked.shape[-2:]:
            m = F.interpolate(m, size=unpacked.shape[-2:], mode="nearest")

        m1, m2, m3 = state["m_pred"]
        shifted = shift_pca_4d(unpacked, m1 * w, m2 * w, m3 * w, u1, u2, u3, mask=m)
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents_steered = pipe(
            prompt=clean_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="latent",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
    finally:
        pipe.scheduler.step = orig_step

    steered_np = decode_latents_4d(vae, unpack_to_4d(pipe, latents_steered, resolution, resolution))
    steered_img = Image.fromarray(steered_np)

    hex_clean = target_hex.replace("#", "")
    safe_obj = re.sub(r"[^\w\-]", "_", str(object_word).strip().lower())
    base_name = f"steered_{safe_obj}_{hex_clean}_seed_{seed}"
    steered_path = out_path / f"{base_name}.png"
    if steered_path.exists():
        suffix = f"_{task_idx:03d}" if task_idx is not None else "_1"
        steered_path = out_path / f"{base_name}{suffix}.png"
        counter = 1
        while steered_path.exists():
            counter += 1
            steered_path = out_path / f"{base_name}_{counter}.png"

    steered_img.save(steered_path)
    print(f"\n  Saved steered image to: {steered_path}")

    # 5. Measure achieved color & Delta E / metrics
    achieved_lab = None
    delta_e = None
    color_metrics_res = None
    if state["mask_pixel"] is not None or metrics:
        final_mask = utils.get_object_mask(steered_img, object_word, seg_models)
        eval_mask = final_mask if (final_mask is not None and final_mask.sum() >= 50) else state["mask_pixel"]
        if eval_mask is not None:
            achieved_lab = utils.measure_color_gt(steered_np, eval_mask)
            if achieved_lab is not None:
                delta_e = utils.ciede2000(target_lab, achieved_lab)
                print(f"  Achieved Color:  L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}")
                print(f"  CIEDE2000 ΔE00:  {delta_e:.2f}")

    if metrics:
        if achieved_lab is not None:
            color_metrics_res = compute_color_metrics(target_lab, achieved_lab)
            print("\n" + "=" * 65)
            print("COLOR EVALUATION METRICS (Output Image vs. Target Color)")
            print("=" * 65)
            print(f"  Target Color (CIELAB):     L*={target_lab[0]:.2f}, a*={target_lab[1]:.2f}, b*={target_lab[2]:.2f}")
            print(f"  Achieved Color (CIELAB):   L*={achieved_lab[0]:.2f}, a*={achieved_lab[1]:.2f}, b*={achieved_lab[2]:.2f}")
            print(f"  Delta E (CIEDE2000 ΔE00):  {color_metrics_res['delta_e00']:.2f}")
            print(f"  Delta E (CIE76 ΔE76):      {color_metrics_res['delta_e76']:.2f}")
            print(f"  Delta Chroma (ΔC*):        {color_metrics_res['delta_chroma']:+.2f}  (|ΔC*| = {color_metrics_res['delta_chroma_abs']:.2f})")
            print(f"  Delta Hue (Δh°):           {color_metrics_res['delta_hue_deg']:+.2f}° (|Δh°| = {color_metrics_res['delta_hue_abs_deg']:.2f}°)")
            print(f"  Delta Hue (metric ΔH*):    {color_metrics_res['delta_H_ab']:+.2f}  (|ΔH*| = {color_metrics_res['delta_H_ab_abs']:.2f})")
            print("=" * 65 + "\n")
        else:
            print("\n[Warning] Could not measure achieved object color for --metrics (mask empty).\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("FLUX.1 Color Steering Generation Completed Successfully!")
    return {
        "steered_img": steered_img,
        "steered_path": str(steered_path),
        "target_lab": target_lab,
        "target_hex": target_hex,
        "object_word": object_word,
        "clean_prompt": clean_prompt,
        "prompt": prompt,
        "achieved_lab": achieved_lab,
        "delta_e00": delta_e,
        "metrics": color_metrics_res,
    }


def run_batch_color_steering_inference(
    prompts: List[str],
    colors: Optional[List[Any]] = None,
    objects: Optional[List[Optional[str]]] = None,
    base_object: Optional[str] = None,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "./inference_outputs",
    mlp_ckpt_path: Optional[str] = None,
    metrics: bool = False,
    cartesian: bool = False,
    models: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Executes a batch of color steering inferences.
    Loads models once upfront and processes all tasks with maximum efficiency.
    """
    # 1. Build tasks
    tasks = build_inference_tasks(
        prompts=prompts,
        colors=colors or [None],
        objects=objects or [None],
        base_object=base_object,
        cartesian=cartesian,
    )

    if not tasks:
        print("[Warning] No inference tasks generated.")
        return []

    # 2. Load models ONCE if not provided
    if models is None:
        models = get_or_load_models(device=device, mlp_ckpt_path=mlp_ckpt_path)

    print("\n" + "=" * 80)
    print(f"STARTING FLUX.1 BATCH COLOR STEERING: {len(tasks)} GENERATION(S)")
    print(f"Device: {device} | Resolution: {resolution}x{resolution} | Steps: {steps} | Guidance: {guidance}")
    print(f"Mode: {'Cartesian product' if cartesian else 'Paired / smart-broadcast'}")
    print("=" * 80)
    for idx, t in enumerate(tasks, 1):
        print(f"  [{idx:2d}/{len(tasks)}] Object: {t['object']} | Color: {t['color']} | Prompt: \"{t['prompt']}\"")
    print("=" * 80 + "\n")

    results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, 1):
        task_seed = seeds[idx - 1] if (seeds and len(seeds) >= idx) else (seeds[0] if (seeds and len(seeds) == 1) else seed)
        print(f"\n>>> Running Task [{idx}/{len(tasks)}]: object='{task['object']}', color='{task['color']}'")
        res = run_color_steering_inference(
            prompt=task["prompt"],
            target_color_spec=task["color"],
            object_word=task["object"],
            seed=task_seed,
            device=device,
            steps=steps,
            guidance=guidance,
            resolution=resolution,
            out_dir=out_dir,
            mlp_ckpt_path=mlp_ckpt_path,
            metrics=metrics,
            models=models,
            task_idx=idx,
        )
        results.append(res)

    # 3. Print Batch Metrics Summary if requested
    if metrics and len(results) > 1:
        valid_metrics = [r["metrics"] for r in results if r.get("metrics") is not None]
        if valid_metrics:
            print("\n" + "=" * 90)
            print(f"FLUX.1 BATCH EVALUATION SUMMARY ({len(results)} generations, {len(valid_metrics)} evaluated)")
            print("=" * 90)
            print(f"{'#':<4} | {'Object':<15} | {'Target Hex':<10} | {'Achieved Lab':<20} | {'ΔE00':<6} | {'ΔE76':<6} | {'ΔC*':<7} | {'Δh°':<7}")
            print("-" * 90)
            for i, (task, res) in enumerate(zip(tasks, results), 1):
                m = res.get("metrics")
                ach_lab = res.get("achieved_lab")
                ach_str = f"({ach_lab[0]:.1f}, {ach_lab[1]:.1f}, {ach_lab[2]:.1f})" if ach_lab else "N/A"
                target_hex = res.get("target_hex", task["color"] or "N/A")
                if m:
                    print(f"{i:<4} | {str(task['object']):<15} | {target_hex:<10} | {ach_str:<20} | {m['delta_e00']:<6.2f} | {m['delta_e76']:<6.2f} | {m['delta_chroma']:<+7.2f} | {m['delta_hue_deg']:<+7.1f}°")
                else:
                    print(f"{i:<4} | {str(task['object']):<15} | {target_hex:<10} | {ach_str:<20} | {'N/A':<6} | {'N/A':<6} | {'N/A':<7} | {'N/A':<7}")
            print("-" * 90)
            mean_de00 = float(np.mean([m["delta_e00"] for m in valid_metrics]))
            mean_de76 = float(np.mean([m["delta_e76"] for m in valid_metrics]))
            mean_dc = float(np.mean([m["delta_chroma_abs"] for m in valid_metrics]))
            mean_dh = float(np.mean([m["delta_hue_abs_deg"] for m in valid_metrics]))
            print(f"Mean CIEDE2000 ΔE00: {mean_de00:.2f} | Mean CIE76 ΔE76: {mean_de76:.2f} | Mean |ΔC*|: {mean_dc:.2f} | Mean |Δh°|: {mean_dh:.2f}°")
            print("=" * 90 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="FLUX.1 Downstream Object Color Steering Application")
    parser.add_argument("--prompt", type=str,
                        default=DEFAULT_PROMPT,
                        help="Text prompt (or base prompt when varying objects/colors)")
    parser.add_argument("--list_prompts", "--prompts", nargs='+', default=None,
                        help="List of prompts to process (overrides --prompt)")
    parser.add_argument("--target-color", "--hex", type=str, default=None,
                        help="Target color specification (Hex, RGB, CIELAB, or name). Auto-detected if omitted.")
    parser.add_argument("--list_target_colors", "--colors", "--target_colors", nargs='+', default=None,
                        help="List of target colors to process")
    parser.add_argument("--object", type=str, default=None,
                        help="Target object to segment (e.g. dog). Auto-detected if omitted.")
    parser.add_argument("--list_objects", "--objects", nargs='+', default=None,
                        help="List of target objects to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list_seeds", "--seeds", nargs='+', type=int, default=None,
                        help="List of random seeds (one per task or broadcast)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Execution device")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Image resolution")
    parser.add_argument("--out-dir", type=str, default="./inference_outputs", help="Output directory")
    parser.add_argument("--mlp-ckpt-path", type=str, default=None, help="Custom MLP checkpoint path")
    parser.add_argument("--metrics", action="store_true",
                        help="Print Delta E, Delta Chroma, and Delta Hue between output image and target color in terminal")
    parser.add_argument("--cartesian", "--grid", action="store_true",
                        help="Generate the full Cartesian product of prompts x objects x colors")

    args = parser.parse_args()

    # Determine prompts
    if args.list_prompts is not None:
        prompts = args.list_prompts
    else:
        prompts = [args.prompt]

    # Determine colors
    if args.list_target_colors is not None:
        colors = args.list_target_colors
    elif args.target_color is not None:
        colors = [args.target_color]
    else:
        colors = [None]

    # Determine objects
    if args.list_objects is not None:
        objects = args.list_objects
    elif args.object is not None:
        objects = [args.object]
    else:
        objects = [None]

    run_batch_color_steering_inference(
        prompts=prompts,
        colors=colors,
        objects=objects,
        base_object=args.object,
        seed=args.seed,
        seeds=args.list_seeds,
        device=args.device,
        steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        out_dir=args.out_dir,
        mlp_ckpt_path=args.mlp_ckpt_path,
        metrics=args.metrics,
        cartesian=args.cartesian,
    )


if __name__ == "__main__":
    main()

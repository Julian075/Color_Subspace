"""
Multi-Zone Semantic Color Transfer for FLUX.1-dev.
Applies latent-space color steering based on extracted reference colors (palettes or images)
and semantic object segmentation (SAM3 / Grounding DINO).
"""

from __future__ import annotations

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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
    PERFIL_GENERATORS,
    setup_flux,
)
from model_pca import load_mlp_pca
import utils as flux_utils
import iscc_nbs

# ---------------------------------------------------------------------------
# Pretrained artifact paths
# ---------------------------------------------------------------------------
PCA_AXES_PATH = FLUX_DIR / "fase_a_pca_out" / "pca_axes.json"
WINNING_SCHEDULE_PATH = FLUX_DIR / "fase_b_pca_out" / "fase_b_winning_schedule.json"
DEFAULT_CKPT_PATH = FLUX_DIR / "mlp_training_out" / "mlp_shift_pca_best.pt"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5


# ===========================================================================
# 1. Principal Color Extraction (Images or Palettes)
# ===========================================================================
def extract_n_principal_colors(
    ref_img_path: str,
    n_colors: int = 6,
    min_delta_e: float = 7.5,
    k_clusters: int = 12,
) -> List[Dict[str, Any]]:
    """Extracts n principal CIELAB colors from a reference image using k-Means clustering
    and CIEDE2000 perceptual deduplication.

    Roles assigned:
      - Dominant / Background: Highest spatial coverage.
      - Focal / Subject: Highest chromatic salience.
      - Accent / Secondary: Distinct accent or complementary tones.
    """
    img = Image.open(ref_img_path).convert("RGB")
    img_small = img.resize((128, 128), Image.Resampling.LANCZOS)
    pixels = np.array(img_small).reshape(-1, 3)
    lab_pixels = flux_utils.rgb_to_lab_batch_np(pixels)

    kmeans = KMeans(n_clusters=max(k_clusters, n_colors * 2), random_state=42, n_init=5).fit(lab_pixels)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    counts = np.bincount(labels, minlength=len(centroids))
    total_pix = len(labels)
    cluster_order = np.argsort(-counts)

    # Selección iterativa garantizando distancia perceptual mínima
    selected_indices: List[int] = []
    curr_min_de = min_delta_e
    for _ in range(5):
        selected_indices = []
        for idx in cluster_order:
            cand_lab = tuple(float(x) for x in centroids[idx])
            is_distinct = True
            for sel_idx in selected_indices:
                sel_lab = tuple(float(x) for x in centroids[sel_idx])
                if flux_utils.ciede2000(cand_lab, sel_lab) < curr_min_de:
                    is_distinct = False
                    break
            if is_distinct:
                selected_indices.append(idx)
            if len(selected_indices) == n_colors:
                break
        if len(selected_indices) == n_colors:
            break
        curr_min_de = max(3.0, curr_min_de - 1.2)

    # If clusters are very close, relax threshold to complete n_colors
    if len(selected_indices) < n_colors:
        for idx in cluster_order:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == n_colors:
                break

    # Role hierarchy:
    # 1. Largest coverage -> dominant / background
    c_dom_idx = selected_indices[0]
    rem = [idx for idx in selected_indices if idx != c_dom_idx]

    # 2. Of the remainder, highest chroma sqrt(a^2 + b^2) -> focal
    chromas = [float(np.sqrt(centroids[i][1] ** 2 + centroids[i][2] ** 2)) for i in rem]
    max_c_pos = int(np.argmax(chromas)) if chromas else 0
    c_focal_idx = rem[max_c_pos] if rem else c_dom_idx
    rem_accents = [idx for idx in rem if idx != c_focal_idx]

    roles_spec = [
        ("dominant", "Dominant / Background", c_dom_idx),
        ("focal", "Focal / Subject", c_focal_idx),
    ]
    for i, a_idx in enumerate(rem_accents):
        roles_spec.append((f"accent_{i+1}", f"Accent / Secondary {i+1}", a_idx))

    results = []
    for key, role_desc, idx in roles_spec:
        lab_val = tuple(float(x) for x in centroids[idx])
        coverage = float(counts[idx] / total_pix * 100.0)
        l2_name, _ = iscc_nbs.find_nearest_iscc_l2(lab_val)
        rgb_approx = _lab_to_rgb_approx(lab_val)
        chroma = float(np.sqrt(lab_val[1] ** 2 + lab_val[2] ** 2))

        results.append({
            "key": key,
            "role": role_desc,
            "lab": lab_val,
            "rgb": rgb_approx,
            "coverage": coverage,
            "iscc_name": l2_name,
            "chroma": chroma,
        })

    return results


def extract_3_principal_colors(
    ref_img_path: str,
    min_delta_e: float = 12.0,
    k_clusters: int = 6,
) -> List[Dict[str, Any]]:
    """Compatibility helper: extracts 3 principal colors."""
    return extract_n_principal_colors(ref_img_path, n_colors=3, min_delta_e=min_delta_e, k_clusters=k_clusters)


def extract_reference_colors(
    ref_img_path: str,
    min_delta_e: float = 8.0,
    n_colors: int = 6,
) -> List[Dict[str, Any]]:
    """Extracts colors from a design palette card (via metadata.json)
    or extracts n principal colors if the reference is an image or artwork.
    """
    p = Path(ref_img_path)
    meta_path = p.parent / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if p.stem in meta and "hex_colors" in meta[p.stem]:
                hex_list = meta[p.stem]["hex_colors"]
                results = []
                for idx, h_col in enumerate(hex_list):
                    h_clean = h_col.lstrip("#")
                    rgb = tuple(int(h_clean[i:i+2], 16) for i in (0, 2, 4))
                    lab = flux_utils.rgb_to_lab_batch_np(np.array([rgb]))[0]
                    lab_tuple = (float(lab[0]), float(lab[1]), float(lab[2]))
                    l2_name, _ = iscc_nbs.find_nearest_iscc_l2(lab_tuple)
                    chroma = float(np.sqrt(lab_tuple[1] ** 2 + lab_tuple[2] ** 2))
                    results.append({
                        "key": f"color_{idx+1}",
                        "role": f"Color {idx+1}",
                        "lab": lab_tuple,
                        "rgb": rgb,
                        "coverage": 100.0 / len(hex_list),
                        "iscc_name": l2_name,
                        "chroma": chroma,
                    })
                # Assign functional roles by chroma and luminance
                sorted_by_chroma = sorted(results, key=lambda x: x["chroma"], reverse=True)
                sorted_by_chroma[0]["role"] = "Focal / Subject"
                sorted_by_chroma[0]["key"] = "focal"

                # Lowest chroma or extreme luminance assigned to background
                sorted_by_chroma[-1]["role"] = "Dominant / Background"
                sorted_by_chroma[-1]["key"] = "dominant"

                for i, r in enumerate(sorted_by_chroma[1:-1]):
                    r["role"] = f"Accent / Secondary {i+1}"
                    r["key"] = f"accent_{i+1}"
                return results
        except Exception:
            pass

    # Extract principal colors for photographs or artworks
    return extract_n_principal_colors(ref_img_path, n_colors=n_colors, min_delta_e=min_delta_e)


def _lab_to_rgb_approx(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Converts CIELAB coordinates to approximate sRGB uint8 for swatch visualization."""
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(t):
        return t ** 3 if t > (6.0 / 29.0) else (3.0 * (6.0 / 29.0) ** 2) * (t - 4.0 / 29.0)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * finv(fx)
    Y = Yn * finv(fy)
    Z = Zn * finv(fz)

    r_lin = X * 3.2404542 - Y * 1.5371385 - Z * 0.4985314
    g_lin = -X * 0.9692660 + Y * 1.8760108 + Z * 0.0415560
    b_lin = X * 0.0556434 - Y * 0.2040259 + Z * 1.0572252

    def to_srgb(c):
        c = max(0.0, min(1.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055

    r = int(round(to_srgb(r_lin) * 255.0))
    g = int(round(to_srgb(g_lin) * 255.0))
    b = int(round(to_srgb(b_lin) * 255.0))
    return (r, g, b)


# ===========================================================================
# 2. PROXY COLOR INJECTION INTO PROMPT
# ===========================================================================
def build_semantic_prompt(
    original_prompt: str,
    color_name: str,
    main_obj: Optional[str] = None,
) -> str:
    """Injects the main proxy color name into the prompt to initialize
    the attention trajectory into the correct chromatic basin.

    Examples:
      original: "a photo of a ceramic mug on a table", color: "purplish red", obj: "mug"
      -> "a photo of a purplish red ceramic mug on a table"
    """
    if not color_name:
        return original_prompt

    # If main object is specified, insert color before it
    if main_obj and main_obj.lower() in original_prompt.lower():
        pattern = re.compile(rf"\b(?:(a|an|the)\s+)?({re.escape(main_obj)})\b", re.IGNORECASE)
        match = pattern.search(original_prompt)
        if match:
            vowels = ("a", "e", "i", "o", "u")
            proper_article = "an" if color_name.lower().startswith(vowels) else "a"
            if match.group(1):
                replacement = f"{proper_article} {color_name} {match.group(2)}"
            else:
                replacement = f"{color_name} {match.group(2)}"
            return original_prompt[:match.start()] + replacement + original_prompt[match.end():]

    # If not found or not specified, insert at start of subject
    pattern_photo = re.compile(r"^(a photo of\s+(?:a|an|the)?\s*)(.*)$", re.IGNORECASE)
    match_photo = pattern_photo.match(original_prompt.strip())
    if match_photo:
        prefix = match_photo.group(1).strip()
        rest = match_photo.group(2).strip()
        article = "an" if color_name.lower().startswith(("a", "e", "i", "o", "u")) else "a"
        return f"a photo of {article} {color_name} {rest}"

    return f"{original_prompt}, in {color_name} tones"


# ===========================================================================
# 3. MULTI-ZONE SEMANTIC SEGMENTATION AT GATE STEP
# ===========================================================================
def segment_scene_zones(
    img_pil: Image.Image,
    main_obj_word: str,
    secondary_obj_words: Optional[List[str]],
    seg_models: Dict[str, Any],
    min_pixels: int = 200,
) -> Dict[str, np.ndarray]:
    """Segments decoded clean latent at gate step into semantic zones:
      - 'focal': primary object mask (M_focal)
      - 'sec_{name}': individual secondary object masks
      - 'bg_light' / 'bg_dark' or 'background': tonal background splits

    Returns:
      Dictionary mapping zone names to boolean masks of shape (H, W).
    """
    w, h = img_pil.size
    zones: Dict[str, np.ndarray] = {}

    # 1. Segment focal object with SAM3
    focal_mask = None
    if main_obj_word:
        focal_mask = flux_utils.get_object_mask(
            img_pil, main_obj_word, seg_models, min_score=0.30, min_pixels=min_pixels
        )

    if focal_mask is not None and focal_mask.sum() >= min_pixels:
        zones["focal"] = focal_mask
    else:
        # Fallback with relaxed score threshold if strict threshold failed
        focal_mask = flux_utils.get_object_mask(
            img_pil, main_obj_word, seg_models, min_score=0.20, min_pixels=min_pixels
        )
        if focal_mask is not None and focal_mask.sum() >= min_pixels:
            zones["focal"] = focal_mask

    # 2. Segment secondary objects individually
    occupied = np.zeros((h, w), dtype=bool)
    if "focal" in zones:
        occupied = occupied | zones["focal"]

    if secondary_obj_words:
        for sec_word in secondary_obj_words:
            sec_word = sec_word.strip()
            if not sec_word:
                continue
            m = flux_utils.get_object_mask(
                img_pil, sec_word, seg_models, min_score=0.25, min_pixels=min_pixels
            )
            if m is not None:
                m_clean = m & (~occupied)
                if m_clean.sum() >= min_pixels:
                    clean_tag = re.sub(r'[^a-zA-Z0-9_]', '_', sec_word.lower())
                    zones[f"sec_{clean_tag}"] = m_clean
                    occupied = occupied | m_clean

    # 3. Residual background / atmosphere
    bg_mask = ~occupied
    if bg_mask.sum() < min_pixels:
        bg_mask = np.ones((h, w), dtype=bool)
        if "focal" in zones:
            bg_mask = bg_mask & (~zones["focal"])

    # If background has sufficient lightness contrast, split into light and dark
    try:
        lab_img = flux_utils.rgb_to_lab_batch_np(np.array(img_pil).reshape(-1, 3)).reshape(h, w, 3)
        bg_l = lab_img[bg_mask, 0]
        if len(bg_l) > 1000 and (np.percentile(bg_l, 80) - np.percentile(bg_l, 20)) > 14.0:
            l_med = float(np.median(bg_l))
            bg_light = bg_mask & (lab_img[:, :, 0] > l_med)
            bg_dark = bg_mask & (lab_img[:, :, 0] <= l_med)
            if bg_light.sum() >= min_pixels and bg_dark.sum() >= min_pixels:
                zones["bg_light"] = bg_light
                zones["bg_dark"] = bg_dark
            else:
                zones["background"] = bg_mask
        else:
            zones["background"] = bg_mask
    except Exception:
        zones["background"] = bg_mask

    return zones


# ===========================================================================
# 4. COMPOSICIÓN ESPACIAL EN LATENTE 4D
# ===========================================================================
def compute_multizone_latent_shift(
    pipe,
    resolution: int,
    latent_h: int,
    latent_w: int,
    zone_masks: Dict[str, np.ndarray],
    zone_shifts: Dict[str, Tuple[float, float, float]],
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    steering_scale: float = 1.0,
) -> torch.Tensor:
    """Computes the spatial 4D perturbation:
        Delta_latent(b, c, y, x) = sum_k M_k^lat(y, x) * sum_i m_{i,k} u_i[c]

    Masks are downscaled to latent resolution using area interpolation,
    ensuring a smooth continuous transition without seam artifacts.
    """
    delta_latent = torch.zeros((1, NUM_LATENT_CHANNELS, latent_h, latent_w), device=device, dtype=torch.float32)

    # Prepare mask tensors in latent dimensions
    lat_masks = {}
    sum_weights = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=torch.float32)

    for zone_name, mask_np in zone_masks.items():
        if zone_name not in zone_shifts:
            continue
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        # Area interpolation for anti-aliasing smoothing
        mask_lat = F.interpolate(mask_t, size=(latent_h, latent_w), mode="area")
        lat_masks[zone_name] = mask_lat
        sum_weights += mask_lat

    # Normalize weights so that sum across zones equals 1.0 per pixel
    sum_weights = torch.clamp(sum_weights, min=1e-5)

    u1_t = torch.tensor(u1, device=device, dtype=torch.float32).view(1, NUM_LATENT_CHANNELS, 1, 1)
    u2_t = torch.tensor(u2, device=device, dtype=torch.float32).view(1, NUM_LATENT_CHANNELS, 1, 1)
    u3_t = torch.tensor(u3, device=device, dtype=torch.float32).view(1, NUM_LATENT_CHANNELS, 1, 1)

    for zone_name, mask_lat in lat_masks.items():
        norm_mask = mask_lat / sum_weights
        m1, m2, m3 = zone_shifts[zone_name]
        # Scaled color perturbation vector for this semantic zone
        color_vec = steering_scale * (m1 * u1_t + m2 * u2_t + m3 * u3_t)  # (1, 16, 1, 1)
        delta_latent += norm_mask * color_vec

    return delta_latent.to(dtype)


# ===========================================================================
# 5. MAIN MULTI-ZONE TRANSFER PIPELINE
# ===========================================================================
def run_multizone_color_transfer(
    prompt: str,
    ref_img_path: str,
    main_obj: str = "car",
    secondary_objs: Optional[List[str]] = None,
    inject_proxy: bool = True,
    seed: int = 42,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "multizone_transfer_out",
    save_baseline: bool = True,
    save_comparison: bool = True,
    mlp_ckpt_path: Optional[str] = None,
    preloaded_models: Optional[Dict[str, Any]] = None,
    target_focal_idx: Optional[int] = None,
    target_bg_idx: Optional[int] = None,
    steering_scale: float = 1.0,
    n_reference_colors: int = 6,
) -> Dict[str, Any]:
    """Executes multi-zone semantic color transfer:
      1. Extracts reference colors from palette or image.
      2. Maps target colors to semantic regions.
      3. Injects natural color proxy into prompt for trajectory initialization.
      4. Decodes x0 at gate step (t=14), segments with SAM3, predicts shifts with ResMLP,
         and applies spatial latent perturbation.
      5. Computes Delta E per zone and saves outputs.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ref_name = Path(ref_img_path).stem

    # 1. Reference color extraction
    principal_colors = extract_reference_colors(ref_img_path, n_colors=n_reference_colors)

    if target_focal_idx is not None and 0 <= target_focal_idx < len(principal_colors):
        c_focal = principal_colors[target_focal_idx]
    else:
        focal_candidates = [c for c in principal_colors if c.get("key") == "focal"]
        c_focal = focal_candidates[0] if focal_candidates else (principal_colors[1] if len(principal_colors) > 1 else principal_colors[0])

    if target_bg_idx is not None and 0 <= target_bg_idx < len(principal_colors):
        c_dominant = principal_colors[target_bg_idx]
    else:
        dom_candidates = [c for c in principal_colors if c.get("key") == "dominant"]
        c_dominant = dom_candidates[0] if dom_candidates else principal_colors[0]

    # Base semantic targets
    zone_targets: Dict[str, Dict[str, Any]] = {
        "focal": c_focal,
        "background": c_dominant,
    }

    # 2. Color proxy injection into prompt
    if inject_proxy:
        effective_prompt = build_semantic_prompt(prompt, c_focal["iscc_name"], main_obj)
    else:
        effective_prompt = prompt

    print("=" * 85)
    print("FLUX MULTI-ZONE / MULTI-OBJECT SEMANTIC COLOR TRANSFER")
    print(f"Original Prompt:    \"{prompt}\"")
    print(f"Effective Prompt:   \"{effective_prompt}\" (proxy: {c_focal['iscc_name']} -> {main_obj})")
    print(f"Reference Image:    {ref_img_path} ({ref_name})")
    print(f"Extracted {len(principal_colors)} Colors from Reference / Palette:")
    for i, c in enumerate(principal_colors):
        print(f"  * Color {i+1} ({c['role']:<22}): ISCC=\"{c['iscc_name']:<16}\" | "
              f"Lab=({c['lab'][0]:5.1f}, {c['lab'][1]:5.1f}, {c['lab'][2]:5.1f}) | Chroma={c.get('chroma', 0.0):4.1f}")
    print(f"Seed: {seed} | Steps: {steps} | Guidance: {guidance} | Res: {resolution}x{resolution} | Device: {device}")
    print("=" * 85)

    # 3. Model setup (or use preloaded)
    if preloaded_models is not None:
        pipe       = preloaded_models["pipe"]
        vae        = preloaded_models["vae"]
        mlp_shift  = preloaded_models["mlp_shift"]
        seg_models = preloaded_models["seg_models"]
        u1         = preloaded_models["u1"]
        u2         = preloaded_models["u2"]
        u3         = preloaded_models["u3"]
        bands      = preloaded_models["bands"]
        latent_h   = preloaded_models["latent_h"]
        latent_w   = preloaded_models["latent_w"]
    else:
        print("\n[Initialization] Loading FLUX, VAE, ResMLP, and SAM3...")
        pipe, vae = setup_flux(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
        ckpt_file = mlp_ckpt_path or str(DEFAULT_CKPT_PATH)
        mlp_shift = load_mlp_pca(ckpt_file, device)
        seg_models = flux_utils.setup_seg_models(device)

        with open(PCA_AXES_PATH) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)

        if WINNING_SCHEDULE_PATH.exists():
            with open(WINNING_SCHEDULE_PATH) as f:
                sched_cfg = json.load(f)
        else:
            sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

        bands = build_envelope_bands(
            sched_cfg["gate_frac"],
            PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
            "ramp_down",
        )
        latent_h, latent_w = latent_hw(pipe, resolution, resolution)

    # 4. Baseline generation (optional for visual comparison)
    baseline_img = None
    if save_baseline or save_comparison:
        print("\n[Step 1/2] Generating baseline image (original prompt, unsteered)...")
        latents_base = pipe(
            prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
        ).images
        base_np = decode_latents_4d(vae, unpack_to_4d(pipe, latents_base, resolution, resolution))
        baseline_img = Image.fromarray(base_np)
        base_path = out_path / f"baseline_seed{seed}.png"
        baseline_img.save(base_path)
        print(f"  -> Saved baseline: {base_path.name}")

    # 5. Multi-Zone Color Transfer generation
    print(f"\n[Step 2/2] Generating with multi-zone transfer from '{ref_name}'...")
    state: Dict[str, Any] = {
        "locked": False,
        "delta_latent": None,
        "zone_masks": {},
        "zone_sources": {},
        "zone_shifts": {},
        "img_x0_np": None,
        "failed": False,
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
            state["img_x0_np"] = img_x0_np
            img_x0_pil = Image.fromarray(img_x0_np)

            # Segment zones on predicted x0
            zone_masks = segment_scene_zones(
                img_x0_pil,
                main_obj_word=main_obj,
                secondary_obj_words=secondary_objs,
                seg_models=seg_models,
            )
            state["zone_masks"] = zone_masks

            print(f"\n  [GATE STEP @ Step {step_index} | frac={frac:.2f}]")
            print(f"  Segmentation of x0 detected {len(zone_masks)} zones:")

            # Measure source CIELAB for each detected zone
            zone_sources: Dict[str, Tuple[float, float, float]] = {}
            for zone_name, mask_np in zone_masks.items():
                s_lab = flux_utils.measure_color_gt(img_x0_np, mask_np)
                if s_lab is None:
                    pix = img_x0_np[mask_np]
                    if len(pix) > 0:
                        lab_batch = flux_utils.rgb_to_lab_batch_np(pix)
                        s_lab = (float(lab_batch[:, 0].mean()),
                                 float(lab_batch[:, 1].mean()),
                                 float(lab_batch[:, 2].mean()))
                    else:
                        s_lab = (50.0, 0.0, 0.0)
                zone_sources[zone_name] = s_lab
                state["zone_sources"][zone_name] = s_lab

            # Assign reference colors to semantic zones
            available_colors = list(principal_colors)

            # Assign focal zone
            if "focal" in zone_masks:
                zone_targets["focal"] = c_focal
                if c_focal in available_colors:
                    available_colors.remove(c_focal)

            # Assign background zones
            if "background" in zone_masks:
                if c_dominant in available_colors:
                    zone_targets["background"] = c_dominant
                    available_colors.remove(c_dominant)
                elif available_colors:
                    best_c = min(available_colors, key=lambda c: flux_utils.ciede2000(zone_sources["background"], c["lab"]))
                    zone_targets["background"] = best_c
                    available_colors.remove(best_c)
                else:
                    zone_targets["background"] = c_dominant

            if "bg_light" in zone_masks:
                candidates = [c for c in available_colors if c["lab"][0] >= 45.0] or available_colors or [c_dominant]
                best_c = min(candidates, key=lambda c: flux_utils.ciede2000(zone_sources["bg_light"], c["lab"]))
                zone_targets["bg_light"] = best_c
                if best_c in available_colors:
                    available_colors.remove(best_c)

            if "bg_dark" in zone_masks:
                candidates = [c for c in available_colors if c["lab"][0] < 45.0] or available_colors or [c_dominant]
                best_c = min(candidates, key=lambda c: flux_utils.ciede2000(zone_sources["bg_dark"], c["lab"]))
                zone_targets["bg_dark"] = best_c
                if best_c in available_colors:
                    available_colors.remove(best_c)

            # Assign remaining secondary zones with distinct colors
            other_zones = [z for z in zone_masks if z not in zone_targets]
            for z_name in other_zones:
                s_lab = zone_sources[z_name]
                if available_colors:
                    best_c = min(available_colors, key=lambda c: flux_utils.ciede2000(s_lab, c["lab"]))
                    zone_targets[z_name] = best_c
                    available_colors.remove(best_c)
                else:
                    best_c = min(principal_colors, key=lambda c: flux_utils.ciede2000(s_lab, c["lab"]))
                    zone_targets[z_name] = best_c

            # ResMLP shift prediction per zone
            zone_shifts: Dict[str, Tuple[float, float, float]] = {}
            for zone_name, mask_np in zone_masks.items():
                cov_pct = mask_np.mean() * 100.0
                source_lab = zone_sources[zone_name]
                target_info = zone_targets.get(zone_name, c_dominant)
                target_lab = target_info["lab"]

                m_pred = mlp_shift.predict_m(source_lab, target_lab)
                zone_shifts[zone_name] = m_pred

                delta_lab = (target_lab[0] - source_lab[0], target_lab[1] - source_lab[1], target_lab[2] - source_lab[2])
                print(f"    - Zone '{zone_name}' ({cov_pct:4.1f}% coverage):")
                print(f"        Source Lab: L*={source_lab[0]:.1f}, a*={source_lab[1]:.1f}, b*={source_lab[2]:.1f}")
                print(f"        Target Lab: L*={target_lab[0]:.1f}, a*={target_lab[1]:.1f}, b*={target_lab[2]:.1f} ({target_info['iscc_name']})")
                print(f"        Delta Lab:  ΔL*={delta_lab[0]:+.1f}, Δa*={delta_lab[1]:+.1f}, Δb*={delta_lab[2]:+.1f}")
                print(f"        ResMLP ->   m1={m_pred[0]:+.4f}, m2={m_pred[1]:+.4f}, m3={m_pred[2]:+.4f}")

            state["zone_shifts"] = zone_shifts

            # Compute composite spatial perturbation in 4D latent space
            delta_lat = compute_multizone_latent_shift(
                pipe_, resolution, latent_h, latent_w,
                zone_masks, zone_shifts, u1, u2, u3, device=device, dtype=DTYPE,
                steering_scale=steering_scale,
            )
            state["delta_latent"] = delta_lat
            state["locked"] = True

        if state["failed"] or w == 0.0 or state["delta_latent"] is None:
            return callback_kwargs

        # Apply envelope-modulated spatial latent shift
        unpacked = unpack_to_4d(pipe_, lat, resolution, resolution)
        shifted = unpacked + w * state["delta_latent"]
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents_steered = pipe(
            effective_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
            callback_on_step_end=cb,
        ).images
    finally:
        pipe.scheduler.step = orig_step

    # 6. Decode steered output image
    steered_np = decode_latents_4d(vae, unpack_to_4d(pipe, latents_steered, resolution, resolution))
    steered_img = Image.fromarray(steered_np)

    case_tag = f"{ref_name}_{main_obj}_seed{seed}"
    steered_path = out_path / f"transfer_{case_tag}.png"
    steered_img.save(steered_path)
    print(f"\n  -> Saved transfer output: {steered_path.name}")

    # 7. Quantitative evaluation (Delta E per zone)
    zone_results = {}
    print("\n" + "-" * 70)
    print("ZONE COLOR TRANSFER METRICS:")
    for zone_name, mask_np in state["zone_masks"].items():
        target_info = zone_targets.get(zone_name, c_dominant)
        target_lab = target_info["lab"]
        source_lab = state["zone_sources"].get(zone_name)

        achieved_lab = flux_utils.measure_color_gt(steered_np, mask_np)
        if achieved_lab is None:
            pix = steered_np[mask_np]
            if len(pix) > 0:
                lab_b = flux_utils.rgb_to_lab_batch_np(pix)
                achieved_lab = (float(lab_b[:, 0].mean()), float(lab_b[:, 1].mean()), float(lab_b[:, 2].mean()))
            else:
                achieved_lab = (50.0, 0.0, 0.0)

        de_source = flux_utils.ciede2000(target_lab, source_lab) if source_lab else 0.0
        de_achieved = flux_utils.ciede2000(target_lab, achieved_lab)
        impr = ((de_source - de_achieved) / de_source * 100.0) if de_source > 0 else 0.0

        zone_results[zone_name] = {
            "source_lab": source_lab,
            "target_lab": target_lab,
            "achieved_lab": achieved_lab,
            "target_color_name": target_info["iscc_name"],
            "delta_e_source": de_source,
            "delta_e_achieved": de_achieved,
            "improvement_pct": impr,
            "coverage_pct": float(mask_np.mean() * 100.0),
        }

        print(f"  * Zone '{zone_name}':")
        print(f"      Target:    L*={target_lab[0]:.1f}, a*={target_lab[1]:.1f}, b*={target_lab[2]:.1f} ({target_info['iscc_name']})")
        print(f"      Achieved:  L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}")
        print(f"      ΔE00:      {de_source:.2f} -> {de_achieved:.2f} ({impr:+.1f}%)")

    # Global full-image color evaluation
    ref_np = np.array(Image.open(ref_img_path).convert("RGB"))
    ref_global_lab = flux_utils.rgb_to_lab_batch_np(ref_np.reshape(-1, 3)).mean(axis=0)
    achieved_global_lab = flux_utils.rgb_to_lab_batch_np(steered_np.reshape(-1, 3)).mean(axis=0)
    global_de = flux_utils.ciede2000(tuple(ref_global_lab), tuple(achieved_global_lab))
    print(f"  * Global ΔE00: {global_de:.2f}")
    print("-" * 70)

    # 8. Save 4-view comparison panel
    if save_comparison:
        comp_path = out_path / f"comparison_{case_tag}.png"
        render_multizone_comparison_panel(
            ref_img_path=ref_img_path,
            principal_colors=principal_colors,
            baseline_img=baseline_img,
            img_x0_np=state["img_x0_np"],
            zone_masks=state["zone_masks"],
            steered_img=steered_img,
            zone_results=zone_results,
            prompt=prompt,
            effective_prompt=effective_prompt,
            main_obj=main_obj,
            save_path=comp_path,
        )
        print(f"  -> Saved comparison panel: {comp_path.name}")

    return {
        "case_tag": case_tag,
        "effective_prompt": effective_prompt,
        "principal_colors": principal_colors,
        "zone_results": zone_results,
        "global_delta_e": global_de,
        "steered_path": str(steered_path),
    }


# ===========================================================================
# 6. Comparison panel rendering (4 views)
# ===========================================================================
def render_multizone_comparison_panel(
    ref_img_path: str,
    principal_colors: List[Dict[str, Any]],
    baseline_img: Optional[Image.Image],
    img_x0_np: Optional[np.ndarray],
    zone_masks: Dict[str, np.ndarray],
    steered_img: Image.Image,
    zone_results: Dict[str, Any],
    prompt: str,
    effective_prompt: str,
    main_obj: str,
    save_path: Path,
):
    """Renders a 4-column high-resolution comparison figure:
      Panel 1: Reference image (or palette) with dominant color swatches.
      Panel 2: Baseline generation without color transfer.
      Panel 3: Semantic zone segmentation with color overlay.
      Panel 4: Final generation with multi-zone color transfer.
    """
    panel_size = 512
    margin = 20
    header_h = 160
    footer_h = 170
    width = panel_size * 4 + margin * 5
    height = panel_size + header_h + footer_h + margin * 2

    canvas = Image.new("RGB", (width, height), color=(24, 26, 32))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        font_panel = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_code = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
    except Exception:
        font_title = font_sub = font_panel = font_text = font_code = ImageFont.load_default()

    # --- Header ---
    draw.text((margin, 18), "FLUX.1-dev Multi-Zone Semantic Color Transfer", fill=(255, 255, 255), font=font_title)
    draw.text((margin, 50), f"Original Prompt:  \"{prompt}\"", fill=(200, 210, 230), font=font_sub)
    draw.text((margin, 76), f"Effective Prompt: \"{effective_prompt}\"", fill=(110, 230, 180), font=font_sub)
    draw.text((margin, 104), f"Focal Object: {main_obj} | Reference: {Path(ref_img_path).name}", fill=(180, 185, 200), font=font_text)

    y_panels = header_h

    # --- Panel 1: Reference + Swatches ---
    ref_img = Image.open(ref_img_path).convert("RGB").resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    x1 = margin
    canvas.paste(ref_img, (x1, y_panels))
    draw.rectangle([x1, y_panels, x1 + panel_size, y_panels + 34], fill=(0, 0, 0, 200))
    draw.text((x1 + 10, y_panels + 8), "1. Reference / Palette", fill=(255, 255, 255), font=font_panel)

    # --- Panel 2: Baseline ---
    x2 = x1 + panel_size + margin
    if baseline_img is not None:
        base_resized = baseline_img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        canvas.paste(base_resized, (x2, y_panels))
    else:
        draw.rectangle([x2, y_panels, x2 + panel_size, y_panels + panel_size], fill=(40, 42, 50))
        draw.text((x2 + 100, y_panels + 240), "No Baseline Generated", fill=(160, 160, 160), font=font_sub)
    draw.rectangle([x2, y_panels, x2 + panel_size, y_panels + 34], fill=(0, 0, 0, 200))
    draw.text((x2 + 10, y_panels + 8), "2. Baseline (No Steering)", fill=(255, 255, 255), font=font_panel)

    # --- Panel 3: Gate Step x0 with Overlays ---
    x3 = x2 + panel_size + margin
    if img_x0_np is not None:
        x0_base = Image.fromarray(img_x0_np).resize((panel_size, panel_size), Image.Resampling.LANCZOS).convert("RGBA")
        overlay = Image.new("RGBA", (panel_size, panel_size), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        # Overlay colors for segmented zones
        zone_overlay_palette = [
            (255, 50, 80, 130),   # Crimson / Magenta
            (255, 170, 0, 130),   # Amber
            (0, 180, 240, 110),   # Sky blue
            (160, 60, 240, 130),  # Purple
            (30, 200, 120, 130),  # Mint emerald
            (240, 210, 40, 120),  # Gold
            (80, 120, 240, 110),  # Indigo
        ]

        for z_i, (z_name, m_np) in enumerate(zone_masks.items()):
            col = zone_overlay_palette[z_i % len(zone_overlay_palette)]
            m_res = Image.fromarray((m_np * 255).astype(np.uint8)).resize((panel_size, panel_size), Image.Resampling.NEAREST)
            m_res_np = np.array(m_res) > 128
            color_layer = np.zeros((panel_size, panel_size, 4), dtype=np.uint8)
            color_layer[m_res_np] = col
            overlay = Image.alpha_composite(overlay, Image.fromarray(color_layer, mode="RGBA"))

        x0_composite = Image.alpha_composite(x0_base, overlay).convert("RGB")
        canvas.paste(x0_composite, (x3, y_panels))
    draw.rectangle([x3, y_panels, x3 + panel_size, y_panels + 34], fill=(0, 0, 0, 200))
    draw.text((x3 + 10, y_panels + 8), "3. Gate Step (x0 + SAM3 Masks)", fill=(255, 255, 255), font=font_panel)

    # --- Panel 4: Steered Multi-Zone Transfer Output ---
    x4 = x3 + panel_size + margin
    steered_res = steered_img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    canvas.paste(steered_res, (x4, y_panels))
    draw.rectangle([x4, y_panels, x4 + panel_size, y_panels + 34], fill=(0, 0, 0, 200))
    draw.text((x4 + 10, y_panels + 8), "4. Multi-Zone Transfer (Output)", fill=(255, 255, 255), font=font_panel)

    # --- Footer: Swatches and Quantitative Metrics ---
    y_footer = y_panels + panel_size + 15

    # Reference color swatches
    draw.text((x1, y_footer), f"Extracted Reference Colors ({len(principal_colors)} Total):", fill=(230, 230, 230), font=font_panel)
    sw_y = y_footer + 28
    sw_w_total = (x3 - x1) - 30
    sw_spacing = max(110, int(sw_w_total / max(len(principal_colors), 1)))
    for i, c in enumerate(principal_colors):
        sw_x = x1 + i * sw_spacing
        draw.rectangle([sw_x, sw_y, sw_x + 28, sw_y + 28], fill=c["rgb"], outline=(255, 255, 255), width=2)
        short_role = c["role"].split("/")[0].strip()
        draw.text((sw_x + 34, sw_y), f"C{i+1}: {short_role[:10]}", fill=(255, 255, 255), font=font_text)
        draw.text((sw_x + 34, sw_y + 14), f"\"{c['iscc_name'][:12]}\"", fill=(180, 210, 255), font=font_text)
        draw.text((sw_x + 34, sw_y + 28), f"Cov:{c['coverage']:.1f}%", fill=(150, 150, 150), font=font_text)

    # Quantitative metrics per zone
    draw.text((x3, y_footer), "Quantitative CIEDE2000 (ΔE00) Evaluation:", fill=(230, 230, 230), font=font_panel)
    met_y = y_footer + 26
    for z_name, z_data in zone_results.items():
        de_init = z_data["delta_e_source"]
        de_final = z_data["delta_e_achieved"]
        impr = z_data["improvement_pct"]
        target_name = z_data["target_color_name"]
        color_tag = (110, 230, 180) if impr > 0 else (255, 120, 120)

        clean_z_name = z_name.replace("sec_", "s_").capitalize()
        line = f"• {clean_z_name:<12} ({target_name[:14]}): ΔE00 {de_init:5.2f} -> {de_final:5.2f}  ({impr:+5.1f}%)"
        draw.text((x3, met_y), line, fill=color_tag, font=font_code)
        met_y += 18

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, quality=95)


# ===========================================================================
# 7. CLI Entrypoint
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FLUX Multi-Zone Semantic Color Transfer using ResMLP + PCA color subspace and SAM3."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt (e.g. 'a photo of a ceramic mug on a table')")
    parser.add_argument("--ref-img", type=str, required=True, help="Path to reference image (palette card or photograph/artwork)")
    parser.add_argument("--main-obj", type=str, default="car", help="Focal object keyword for segmentation and proxy injection")
    parser.add_argument("--secondary-objs", type=str, default=None, help="Comma-separated secondary object keywords (e.g. 'wheel,street')")
    parser.add_argument("--no-proxy", action="store_true", help="Disable color proxy injection into the prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Classifier-free guidance scale")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--out-dir", type=str, default="multizone_transfer_out", help="Output directory")
    parser.add_argument("--no-baseline", action="store_true", help="Do not generate baseline image without steering")
    parser.add_argument("--no-comparison", action="store_true", help="Do not generate 4-view comparison panel")
    parser.add_argument("--mlp-ckpt", type=str, default=None, help="Optional path to ResMLP checkpoint")

    args = parser.parse_args()

    sec_list = [x.strip() for x in args.secondary_objs.split(",")] if args.secondary_objs else None

    run_multizone_color_transfer(
        prompt=args.prompt,
        ref_img_path=args.ref_img,
        main_obj=args.main_obj,
        secondary_objs=sec_list,
        inject_proxy=not args.no_proxy,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        out_dir=args.out_dir,
        save_baseline=not args.no_baseline,
        save_comparison=not args.no_comparison,
        mlp_ckpt_path=args.mlp_ckpt,
    )


if __name__ == "__main__":
    main()

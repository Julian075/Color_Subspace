"""
UTILS.PY -- Colorimetry, Perceptual Metrics, and Instance Segmentation for SD3.

Provides:
  1. Exact sRGB <-> CIELAB Conversions (Batch and Single NumPy routines)
  2. Robust Dominant Object Color Extraction (PCA Chromaticity + Perpendicular MAD Trimming)
  3. CIEDE2000 (dE00) and CIE76 (dE76) Color Difference Formulas
  4. SAM-3 Instance Segmentation Loader & Text-Prompt Mask Generator
  5. Comprehensive CSS/HTML Color Names to CIELAB Mapping
"""

import os
import math
import re
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
from PIL import Image

# =========================== COLOR SPACE CONVERSIONS ===========================

def rgb_to_lab_batch_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Converts (N, 3) or (H, W, 3) uint8 sRGB array to CIELAB (D65, 2 deg observer).
    """
    orig_shape = rgb_uint8.shape
    rgb = rgb_uint8.reshape(-1, 3).astype(np.float64) / 255.0

    # sRGB to Linear RGB
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # Linear RGB to XYZ (D65 matrix)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float64)
    xyz = rgb_lin @ M.T

    # Normalize by D65 reference white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xyz[:, 0] /= Xn
    xyz[:, 1] /= Yn
    xyz[:, 2] /= Zn

    # Non-linear transformation function f(t)
    delta = 6.0 / 29.0
    mask_xyz = xyz > (delta ** 3)
    f_xyz = np.where(mask_xyz, np.cbrt(xyz), (xyz / (3.0 * delta ** 2)) + (4.0 / 29.0))

    # Calculate L*, a*, b*
    L = (116.0 * f_xyz[:, 1]) - 16.0
    a = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])

    lab = np.column_stack([L, a, b])
    return lab.reshape(orig_shape)


def rgb_to_lab_single_np(rgb_triplet: Tuple[int, int, int]) -> Tuple[float, float, float]:
    arr = np.array([rgb_triplet], dtype=np.uint8)
    lab = rgb_to_lab_batch_np(arr)[0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def lab_to_rgb_single_np(lab_triplet: Tuple[float, float, float]) -> Tuple[int, int, int]:
    L, a, b = lab_triplet
    delta = 6.0 / 29.0

    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    x = fx ** 3 if fx > delta else (fx - 16.0 / 116.0) * 3.0 * (delta ** 2)
    y = fy ** 3 if fy > delta else (fy - 16.0 / 116.0) * 3.0 * (delta ** 2)
    z = fz ** 3 if fz > delta else (fz - 16.0 / 116.0) * 3.0 * (delta ** 2)

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = x * Xn, y * Yn, z * Zn

    # Inverse XYZ to linear sRGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float64)

    rgb_lin = np.array([X, Y, Z]) @ M_inv.T
    rgb = np.where(rgb_lin > 0.0031308, 1.055 * (np.maximum(rgb_lin, 0.0) ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_lin)
    rgb = np.clip(rgb * 255.0, 0, 255).round().astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    elif len(hex_clean) == 5:
        hex_clean = hex_clean.ljust(6, "0")
    elif len(hex_clean) != 6:
        raise ValueError(f"Invalid Hex code: {hex_str!r}")
    return int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)


def parse_target_color(spec: Any) -> Tuple[float, float, float]:
    """
    Parses color specifications in multiple formats into a CIELAB (L, a, b) tuple:
      - Hex: "#FF5733" or "FF5733" or "#F00"
      - RGB: (255, 87, 51) or [255, 87, 51] or "rgb(255, 87, 51)"
      - CIELAB: {"lab": (60, 45, 50)} or "lab(60.0, 45.0, 50.0)"
      - Color name: "crimson", "maroon", etc.
    """
    if isinstance(spec, dict):
        if "lab" in spec:
            return tuple(float(x) for x in spec["lab"])
        if all(k in spec for k in ("r", "g", "b")):
            return tuple(rgb_to_lab_single_np((int(spec["r"]), int(spec["g"]), int(spec["b"]))))
    if isinstance(spec, (tuple, list, np.ndarray)) and len(spec) == 3:
        return tuple(rgb_to_lab_single_np(spec))
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower().startswith("lab(") and s.endswith(")"):
            vals = [float(x.strip()) for x in s[4:-1].split(",")]
            return (vals[0], vals[1], vals[2])
        if s.lower().startswith("rgb(") and s.endswith(")"):
            vals = [float(x.strip()) for x in s[4:-1].split(",")]
            return tuple(rgb_to_lab_single_np(vals))
        # Hex attempt
        try:
            return tuple(rgb_to_lab_single_np(hex_to_rgb(s)))
        except Exception:
            pass
        # Named color attempt via PIL
        try:
            from PIL import ImageColor
            return tuple(rgb_to_lab_single_np(ImageColor.getrgb(s)))
        except Exception:
            pass
        # Named color attempt via ISCC-NBS
        try:
            from iscc_nbs import get_iscc_l2_centroid
            return get_iscc_l2_centroid(s)
        except Exception:
            pass
    raise ValueError(f"Unable to parse target color: {spec!r}")


def extract_hex_from_text(text: str) -> Optional[str]:
    """Matches any 3, 6, or 8-digit hex code in text."""
    match = re.search(r"#[0-9a-fA-F]{3,8}", text)
    return match.group(0) if match else None


def extract_color_spec_from_text(text: str) -> Optional[str]:
    """Discovers color specification (hex, rgb, lab) from prompt text."""
    m_rgb = re.search(r"rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", text, re.IGNORECASE)
    if m_rgb:
        return m_rgb.group(0)
    m_lab = re.search(r"lab\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)", text, re.IGNORECASE)
    if m_lab:
        return m_lab.group(0)
    m_hex = re.search(r"#[0-9a-fA-F]{3,8}", text)
    if m_hex:
        return m_hex.group(0)
    m_hex_kw = re.search(r"\bhex\s+([0-9a-fA-F]{6})\b", text, re.IGNORECASE)
    if m_hex_kw:
        return f"#{m_hex_kw.group(1)}"
    return None


def extract_object_from_text(text: str) -> str:
    """
    Dynamically discovers the target object from prompt text without hardcoded object whitelists.
    Fully compatible with GenColorBench NCU prompt templates and open natural language prompts.
    """
    cleaned = text.strip().rstrip(".")

    # 1. '<HEX/RGB/LAB>-colored <OBJ>' pattern
    m = re.search(r"(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\))-colored\s+([^,.;]+)", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Color clause pattern matching tail
    color_clause_pattern = (
        r"(?:\s+(?:in\s+(?:the\s+|hex\s+|rgb\s+)?color"
        r"|in\s+hex"
        r"|in\s+rgb"
        r"|in"
        r"|colored"
        r"|with\s+(?:the\s+)?color"
        r"|with\s+hex\s+color"
        r"|rendered\s+(?:entirely\s+)?in(?:\s+(?:rgb|hex))?(?:\s+color)?"
        r"|designed\s+in)"
        r"\s+(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\)|[a-zA-Z]+)(?:\s+color)?)"
    )

    # Priority A: prompt starting with photo/image/close-up/picture of ...
    m = re.search(r"^(?:an?\s+)?(?:photo|image|close-up|picture)\s+of\s+(?:an?\s+)?(.*?)(?=" + color_clause_pattern + r"|$)", cleaned, re.IGNORECASE)
    if m and m.group(1).strip():
        candidate = m.group(1).strip()
        candidate = re.sub(r"^(?:highly\s+detailed|realistic|single)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = re.split(r"\s+(?:on|at|in|near|by|with|under|over)\s+", candidate, flags=re.IGNORECASE)[0]
        if candidate.strip():
            return candidate.strip()

    # Priority B: prompt starting with adjectives like highly detailed / realistic
    m = re.search(r"^(?:an?\s+)?(?:highly\s+detailed|realistic|single)\s+(.*?)(?=" + color_clause_pattern + r"|$)", cleaned, re.IGNORECASE)
    if m and m.group(1).strip():
        candidate = m.group(1).strip()
        candidate = re.split(r"\s+(?:on|at|in|near|by|with|under|over)\s+", candidate, flags=re.IGNORECASE)[0]
        if candidate.strip():
            return candidate.strip()

    # Priority C: general prefix before color clause
    m = re.search(r"^(?:an?\s+)?(.*?)(?=" + color_clause_pattern + r")", cleaned, re.IGNORECASE)
    if m and m.group(1).strip():
        candidate = m.group(1).strip()
        candidate = re.split(r"\s+(?:on|at|in|near|by|with|under|over)\s+", candidate, flags=re.IGNORECASE)[0]
        if candidate.strip():
            return candidate.strip()

    # Fallback
    clean = re.sub(r"^(?:an?\s+)?(?:photo|image|close-up|picture)\s+of\s+(?:an?\s+)?", "", cleaned, flags=re.IGNORECASE)
    clean = re.split(r"\s+(?:on|at|in|near|by|with|under|over)\s+", clean, flags=re.IGNORECASE)[0]
    return clean.strip() if clean.strip() else "object"


def clean_prompt_for_diffusion(prompt: str, color_name: str) -> str:
    """
    Substitutes numerical color codes in the prompt with natural language color names,
    preserving natural semantics for text-to-image backbones.
    """
    p = prompt
    p = re.sub(r"#[0-9a-fA-F]{3,8}-colored", f"{color_name}-colored", p)
    p = re.sub(r"rgb\([^)]+\)-colored", f"{color_name}-colored", p, flags=re.IGNORECASE)
    p = re.sub(r"lab\([^)]+\)-colored", f"{color_name}-colored", p, flags=re.IGNORECASE)

    p = re.sub(r"\b(?:in\s+(?:the\s+|hex\s+|rgb\s+)?color|with\s+(?:the\s+|hex\s+)?color|in\s+hex|in\s+rgb)\s+(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\))", f"colored {color_name}", p, flags=re.IGNORECASE)
    p = re.sub(r"\bdesigned\s+in\s+(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\))(?:\s+color)?", f"colored {color_name}", p, flags=re.IGNORECASE)
    p = re.sub(r"\brendered\s+in\s+(?:RGB\s+color\s+|hex\s+color\s+)?(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\))(?:\s+color)?", f"colored {color_name}", p, flags=re.IGNORECASE)
    p = re.sub(r"\bcolored\s+(?:#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|lab\([^)]+\))", f"colored {color_name}", p, flags=re.IGNORECASE)

    p = re.sub(r"#[0-9a-fA-F]{3,8}", color_name, p)
    p = re.sub(r"rgb\([^)]+\)", color_name, p, flags=re.IGNORECASE)
    p = re.sub(r"lab\([^)]+\)", color_name, p, flags=re.IGNORECASE)
    p = re.sub(r"\bat\s+color\b", "colored", p, flags=re.IGNORECASE)
    return p.strip()


# =========================== COLOR DIFFERENCE METRICS ===========================

def ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float],
              kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """
    Standard CIEDE2000 total color difference formulation (ISO/CIE 11664-6:2014).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))
    a1_p = (1.0 + G) * a1
    a2_p = (1.0 + G) * a2

    C1_p = math.sqrt(a1_p**2 + b1**2)
    C2_p = math.sqrt(a2_p**2 + b2**2)

    h1_p = math.degrees(math.atan2(b1, a1_p)) % 360.0
    h2_p = math.degrees(math.atan2(b2, a2_p)) % 360.0

    dL_p = L2 - L1
    dC_p = C2_p - C1_p

    if C1_p * C2_p == 0.0:
        dh_p = 0.0
    else:
        diff_h = h2_p - h1_p
        if abs(diff_h) <= 180.0:
            dh_p = diff_h
        elif diff_h > 180.0:
            dh_p = diff_h - 360.0
        else:
            dh_p = diff_h + 360.0

    dH_p = 2.0 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(dh_p / 2.0))

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1_p + C2_p) / 2.0

    if C1_p * C2_p == 0.0:
        h_bar_p = h1_p + h2_p
    else:
        sum_h = h1_p + h2_p
        diff_h = abs(h1_p - h2_p)
        if diff_h <= 180.0:
            h_bar_p = sum_h / 2.0
        elif sum_h < 360.0:
            h_bar_p = (sum_h + 360.0) / 2.0
        else:
            h_bar_p = (sum_h - 360.0) / 2.0

    T = (1.0 - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
             + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
             + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
             - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0)))

    delta_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * math.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7))
    S_L = 1.0 + ((0.015 * ((L_bar_p - 50.0)**2)) / math.sqrt(20.0 + (L_bar_p - 50.0)**2))
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * delta_theta)) * R_C

    dE = math.sqrt(
        (dL_p / (kL * S_L))**2 +
        (dC_p / (kC * S_C))**2 +
        (dH_p / (kH * S_H))**2 +
        R_T * (dC_p / (kC * S_C)) * (dH_p / (kH * S_H))
    )
    return float(dE)


def cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    return float(math.sqrt((lab1[0] - lab2[0])**2 + (lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2))


# =========================== DOMINANT COLOR EXTRACTION ===========================

def measure_color_gt(img_uint8: np.ndarray, mask_np: Optional[np.ndarray], z_thresh: float = 2.5) -> Optional[Tuple[float, float, float]]:
    """
    Robust dominant color measurement:
    1. Extracts object pixels in CIELAB space.
    2. Performs PCA on the (a*, b*) chromaticity plane.
    3. Trims outliers via median absolute deviation (MAD) z-scores projected
       perpendicular to the principal chromatic axis.
    Returns (L*, a*, b*) tuple or None if mask is invalid.
    """
    if mask_np is None:
        return None
    pixels = img_uint8[mask_np.astype(bool)]
    if len(pixels) < 10:
        return None
    lab = rgb_to_lab_batch_np(pixels)
    L, ab = lab[:, 0], lab[:, 1:3]

    mean_ab = ab.mean(0)
    centered = ab - mean_ab
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    # Outlier rejection along the perpendicular axis
    perp = np.array([-axis[1], axis[0]])
    perp_proj = centered @ perp
    med = np.median(perp_proj)
    mad = np.median(np.abs(perp_proj - med)) * 1.4826 + 1e-6
    z = np.abs(perp_proj - med) / mad
    inliers = z < z_thresh

    if inliers.sum() < 5:
        return float(L.mean()), float(mean_ab[0]), float(mean_ab[1])

    L_in = L[inliers]
    ab_in = ab[inliers]
    return float(L_in.mean()), float(ab_in[:, 0].mean()), float(ab_in[:, 1].mean())


# =========================== SAM-3 SEGMENTATION LOADER ===========================

def setup_seg_models(device: str = "cuda"):
    """
    Initializes SAM-3 (facebook/sam3) text-prompted instance segmentation model.
    """
    print("Loading SAM-3 segmentation model (facebook/sam3)...")
    from transformers import Sam3Processor, Sam3Model
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    return {"processor": processor, "model": model, "device": device}


@torch.no_grad()
def get_object_mask(img_pil: Image.Image, obj_word: str, seg_models: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Generates a boolean 2D mask of the target object using SAM-3.
    Returns (H, W) boolean array, or None if no valid mask is found.
    """
    processor, model, device = seg_models["processor"], seg_models["model"], seg_models["device"]

    inputs = processor(images=img_pil, text=obj_word.strip(), return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(img_pil.size[1], img_pil.size[0])]
    )[0]

    mask_np = None
    # Attempt 1: Standard transformers pattern (segmentation + segments_info)
    if isinstance(results, dict) and "segmentation" in results and results.get("segments_info"):
        seg = results["segmentation"]
        seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
        best_seg = max(results["segments_info"], key=lambda s: s.get("score", 0.0))
        mask_np = (seg_np == best_seg["id"])

    # Attempt 2: Direct raw mask tensor fallback
    if mask_np is None:
        if isinstance(results, dict) and "masks" in results:
            masks_tensor = results["masks"]
            if masks_tensor.ndim >= 2 and masks_tensor.shape[0] > 0:
                scores = results.get("scores", None)
                best_idx = int(torch.argmax(scores).item()) if (scores is not None and len(scores) > 0) else 0
                mask_np = masks_tensor[best_idx].cpu().numpy() > 0.5
        elif isinstance(results, dict) and "segmentation" in results:
            seg = results["segmentation"]
            seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
            mask_np = seg_np > 0

    if mask_np is not None and mask_np.ndim > 2:
        mask_np = mask_np[0]
    if mask_np is None or not mask_np.any() or mask_np.sum() < 20:
        return None
    return mask_np.astype(bool)


# =========================== CSS COLOR DICTIONARY ===========================

CSS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (53.2, 80.1, 67.2),
    "green": (46.2, -51.7, 49.9),
    "blue": (32.3, 79.2, -107.9),
    "yellow": (97.1, -21.6, 94.5),
    "orange": (74.9, 23.9, 78.9),
    "purple": (29.8, 58.8, -36.5),
    "pink": (81.2, 33.7, 7.8),
    "cyan": (91.1, -48.1, -14.1),
    "magenta": (60.3, 98.2, -60.8),
    "lime": (87.7, -86.2, 83.2),
    "teal": (51.9, -28.6, -8.5),
    "navy": (12.7, 47.3, -64.3),
    "maroon": (25.2, 47.9, 37.9),
    "olive": (51.9, -12.9, 56.7),
    "silver": (77.1, 0.0, 0.0),
    "gray": (53.6, 0.0, 0.0),
    "white": (100.0, 0.0, 0.0),
    "black": (0.0, 0.0, 0.0),
    "coral": (68.2, 42.4, 42.6),
    "turquoise": (80.1, -38.6, -7.0),
    "gold": (85.2, 7.0, 84.7),
    "lavender": (89.5, 9.8, -13.0),
    "beige": (94.7, -0.6, 12.0),
}


def get_css_lab(color_name: str) -> Optional[Tuple[float, float, float]]:
    clean = color_name.strip().lower()
    return CSS_COLORS.get(clean, None)


def nearest_color_name(target_lab: Tuple[float, float, float] | Any) -> Tuple[str, float]:
    """
    Finds the closest ISCC-NBS Level 2 color name to target_lab (L, a, b)
    using CIEDE2000 color difference.
    """
    from iscc_nbs import find_nearest_iscc_l2
    return find_nearest_iscc_l2(target_lab, ciede2000_fn=ciede2000)


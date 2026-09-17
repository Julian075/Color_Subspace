"""
UTILS.PY -- Segmentation (SAM3) + Color Science (CIELAB, Robust Dominant Color within Mask, CIEDE2000)
for Stable Diffusion 3.5 Medium (SD3.5-M).
"""

import math
import re
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from PIL import Image


# =========================== SEGMENTATION: SAM3 ===========================
def setup_seg_models(device):
    """
    SAM3: Takes direct natural language text (e.g. 'sphere', 'car') and outputs
    instance segmentation masks without requiring bounding boxes.
    """
    from transformers import Sam3Processor, Sam3Model
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    return {"processor": processor, "model": model, "device": device}


@torch.no_grad()
def get_object_mask(img_pil, obj_word, seg_models):
    """
    Generates a boolean 2D mask of the target object using SAM-3.
    Returns (H, W) boolean array, or None if no valid mask is found.
    """
    processor, model, device = seg_models["processor"], seg_models["model"], seg_models["device"]

    inputs = processor(images=img_pil, text=obj_word, return_tensors="pt").to(device)
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
                mask_np = masks_tensor[0].cpu().numpy() > 0.5
        elif isinstance(results, dict) and "segmentation" in results:
            seg = results["segmentation"]
            seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
            mask_np = seg_np > 0

    if mask_np is not None and mask_np.ndim > 2:
        mask_np = mask_np[0]
    if mask_np is None or not mask_np.any():
        return None
    return mask_np.astype(bool)


# =========================== COLOR: sRGB -> CIELAB ===========================
def srgb_to_linear_np(c01):
    return np.where(c01 <= 0.04045, c01 / 12.92, ((c01 + 0.055) / 1.055) ** 2.4)


def rgb_to_lab_batch_np(rgb255):
    """Converts (N, 3) sRGB [0..255] to (N, 3) CIELAB (D65 standard illuminant)."""
    rgb01 = np.asarray(rgb255, dtype=np.float64) / 255.0
    rlin, glin, blin = (
        srgb_to_linear_np(rgb01[:, 0]),
        srgb_to_linear_np(rgb01[:, 1]),
        srgb_to_linear_np(rgb01[:, 2]),
    )
    X = rlin * 0.4124564 + glin * 0.3575761 + blin * 0.1804375
    Y = rlin * 0.2126729 + glin * 0.7151522 + blin * 0.0721750
    Z = rlin * 0.0193339 + glin * 0.1191920 + blin * 0.9503041
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    d = 6.0 / 29.0

    def f(t):
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def rgb_to_lab_single_np(rgb255):
    return rgb_to_lab_batch_np(np.asarray(rgb255).reshape(1, 3))[0]


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_clean = hex_code.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    elif len(hex_clean) == 5:
        hex_clean = hex_clean.ljust(6, "0")
    elif len(hex_clean) != 6:
        raise ValueError(f"Invalid Hex code: {hex_code!r}")
    return tuple(int(hex_clean[i:i + 2], 16) for i in (0, 2, 4))


def lab_to_rgb_single_np(lab_triplet: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Converts a single (L*, a*, b*) CIELAB tuple to sRGB uint8 (r, g, b) in [0..255]."""
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


# =========================== COLOR: ROBUST DOMINANT COLOR WITHIN MASK ===========================
def measure_color_gt(img_uint8, mask_np, z_thresh=2.5):
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
    v1 = eigvecs[:, np.argmax(eigvals)]
    v_perp = np.array([-v1[1], v1[0]])

    proj_along = centered @ v1
    proj_perp = centered @ v_perp
    med = np.median(proj_perp)
    mad = np.median(np.abs(proj_perp - med)) * 1.4826 + 1e-8
    z = np.abs(proj_perp - med) / mad
    keep = z <= z_thresh
    if not keep.any():
        keep = np.ones_like(keep)

    a_mean = mean_ab[0] + proj_along[keep].mean() * v1[0]
    b_mean = mean_ab[1] + proj_along[keep].mean() * v1[1]
    L_mean = L[keep].mean()
    return float(L_mean), float(a_mean), float(b_mean)


# =========================== COLOR: CIEDE2000 & CIE76 ===========================
def delta_e_cie76(lab1, lab2):
    """Standard Euclidean distance in CIELAB space."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


def ciede2000(lab1, lab2):
    """
    Exact CIEDE2000 color difference formula (Sharma et al., 2005).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    kL = kC = kH = 1.0
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    def hue(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = np.degrees(np.arctan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p, h2p = hue(a1p, b1), hue(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        diff = h2p - h1p
        dhp = diff if abs(diff) <= 180 else (diff - 360 if diff > 180 else diff + 360)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        Hbarp = h1p + h2p
    else:
        diff = abs(h1p - h2p)
        if diff <= 180:
            Hbarp = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360:
            Hbarp = (h1p + h2p + 360) / 2.0
        else:
            Hbarp = (h1p + h2p - 360) / 2.0

    T = (1 - 0.17 * np.cos(np.radians(Hbarp - 30)) + 0.24 * np.cos(np.radians(2 * Hbarp))
        + 0.32 * np.cos(np.radians(3 * Hbarp + 6)) - 0.20 * np.cos(np.radians(4 * Hbarp - 63)))
    dTheta = 30 * np.exp(-(((Hbarp - 275) / 25.0) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    dE = np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))
    return float(dE)


# =========================== COLOR: NEAREST CSS COLOR NAME ===========================
CHROMA_THRESHOLD = 15.0
USE_FULL_COLOR_PALETTE = True

CURATED_COLOR_NAMES = [
    "white", "maroon", "beige", "gray", "black", "red", "orange", "yellow", "purple", "blue",
    "green", "pink", "brown", "cyan", "turquoise", "magenta", "olive", "teal",
    "lightblue", "navy", "lightgreen", "darkgreen", "lime", "coral", "gold", "silver"
]


def _build_color_names_table(use_full=USE_FULL_COLOR_PALETTE):
    from PIL import ImageColor
    names_source = ImageColor.colormap.keys() if use_full else CURATED_COLOR_NAMES
    seen_rgb = {}
    for name in names_source:
        rgb = ImageColor.getrgb(name)
        if rgb not in seen_rgb:
            seen_rgb[rgb] = name
    return {name: rgb_to_lab_single_np(rgb) for rgb, name in seen_rgb.items()}


COLOR_NAMES_LAB = _build_color_names_table()
COLOR_NAMES = list(COLOR_NAMES_LAB.keys())
ACHROMATIC_NAMES = {name for name, (L, a, b) in COLOR_NAMES_LAB.items()
                    if math.hypot(a, b) < CHROMA_THRESHOLD}
CHROMATIC_NAMES = [n for n in COLOR_NAMES if n not in ACHROMATIC_NAMES]


def nearest_color_name(target_lab):
    L, a, b = target_lab
    chroma = math.sqrt(a ** 2 + b ** 2)
    candidates = ACHROMATIC_NAMES if chroma < CHROMA_THRESHOLD else CHROMATIC_NAMES

    best_name, best_dist = None, float("inf")
    for name in candidates:
        d = ciede2000(target_lab, COLOR_NAMES_LAB[name])
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist

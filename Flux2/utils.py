"""
UTILS.PY -- segmentacion (SAM3) + color (Lab, medicion robusta dentro de
mascara) para el proyecto FLUX. Un solo archivo: las dos cosas se usan
siempre juntas ac (segmentar el objeto, despues medir su color), no hay
razon real para tenerlas separadas.

Copiado (no importado) de las funciones equivalentes del proyecto SDXL --
proyectos separados a proposito (ver discusion anterior): un cambio futuro
en el pipeline de SDXL no rompe nada ac, y viceversa.
"""

import math
import os
import re
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
from PIL import Image

def _resolve_hf_snapshot(repo_id):
    hf_home = os.environ.get("HF_HOME", "/leonardo_work/AIFAC_S07_004/jsantamaria/.cache/huggingface")
    hub_dir = os.path.join(hf_home, "hub")
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    repo_path = os.path.join(hub_dir, repo_folder)
    if os.path.exists(repo_path):
        snapshots_dir = os.path.join(repo_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snaps = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir) if not s.startswith(".")]
            if snaps:
                return snaps[0]
    return repo_id


# =========================== segmentacion: SAM3 ===========================
def setup_seg_models(device):
    """SAM3: toma texto directo (ej. 'sphere'), no necesita que otro modelo
    le pase una caja primero (~200 propuestas candidatas internas, tipo
    DETR, filtradas por el propio texto)."""
    from transformers import Sam3Processor, Sam3Model
    sam3_path = _resolve_hf_snapshot("facebook/sam3")
    try:
        processor = Sam3Processor.from_pretrained(sam3_path, local_files_only=True)
        model = Sam3Model.from_pretrained(sam3_path, local_files_only=True).to(device).eval()
    except Exception:
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    return {"processor": processor, "model": model, "device": device}


@torch.no_grad()
def get_object_mask(img_pil, obj_word, seg_models, min_score=0.35, min_pixels=50):
    """Segmenta todas las instancias del objeto que superen el umbral de confianza (min_score)
    y combina sus mascaras en una mascara binaria unica (H, W)."""
    processor, model, device = seg_models["processor"], seg_models["model"], seg_models["device"]

    inputs = processor(images=img_pil, text=obj_word, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(img_pil.size[1], img_pil.size[0])])[0]

    mask_np = None
    # Intento 1: patron estandar de transformers (segmentation + segments_info)
    if isinstance(results, dict) and "segmentation" in results and results.get("segments_info"):
        seg = results["segmentation"]
        seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
        # Seleccionar todas las instancias con score >= min_score
        valid_ids = [s["id"] for s in results["segments_info"] if s.get("score", 0.0) >= min_score]
        if valid_ids:
            mask_np = np.isin(seg_np, valid_ids)
        elif results["segments_info"]:
            # Fallback al segmento con mayor score si ninguno supero min_score
            best_seg = max(results["segments_info"], key=lambda s: s.get("score", 0.0))
            mask_np = (seg_np == best_seg["id"])

    # Intento 2 (fallback para formato de tensores de mascaras directas)
    if mask_np is None:
        if isinstance(results, dict) and "masks" in results:
            masks_tensor = results["masks"]
            scores = results.get("scores", None)
            if masks_tensor.ndim >= 2 and masks_tensor.shape[0] > 0:
                if masks_tensor.ndim == 2:
                    masks_tensor = masks_tensor.unsqueeze(0)
                if scores is not None and len(scores) == masks_tensor.shape[0]:
                    keep = (scores >= min_score).cpu()
                    if keep.any():
                        masks_tensor = masks_tensor[keep]
                    else:
                        masks_tensor = masks_tensor[:1]
                mask_np = (masks_tensor.sum(dim=0) > 0.5).cpu().numpy()
        elif isinstance(results, dict) and "segmentation" in results:
            seg = results["segmentation"]
            seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
            mask_np = (seg_np > 0)

    if mask_np is not None and mask_np.ndim > 2:
        mask_np = mask_np[0]
    if mask_np is None or not mask_np.any() or mask_np.sum() < min_pixels:
        return None
    return mask_np.astype(bool)


@torch.no_grad()
def get_individual_object_masks(img_pil, obj_word, seg_models, min_score=0.35, min_pixels=50):
    """Devuelve una lista de mascaras individuales [mask_1, mask_2, ...] ordenadas por score."""
    processor, model, device = seg_models["processor"], seg_models["model"], seg_models["device"]

    inputs = processor(images=img_pil, text=obj_word, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(img_pil.size[1], img_pil.size[0])])[0]

    individual_masks = []
    if isinstance(results, dict) and "segmentation" in results and results.get("segments_info"):
        seg = results["segmentation"]
        seg_np = seg.cpu().numpy() if hasattr(seg, "cpu") else np.array(seg)
        sorted_segs = sorted(results["segments_info"], key=lambda s: s.get("score", 0.0), reverse=True)
        for s in sorted_segs:
            if s.get("score", 0.0) >= min_score:
                inst = (seg_np == s["id"])
                if inst.sum() >= min_pixels:
                    individual_masks.append(inst.astype(bool))

    if not individual_masks and isinstance(results, dict) and "masks" in results:
        masks_tensor = results["masks"]
        scores = results.get("scores", None)
        if masks_tensor.ndim >= 3 and masks_tensor.shape[0] > 0:
            for idx in range(masks_tensor.shape[0]):
                score = scores[idx].item() if scores is not None else 1.0
                if score >= min_score:
                    inst = (masks_tensor[idx].cpu().numpy() > 0.5)
                    if inst.sum() >= min_pixels:
                        individual_masks.append(inst.astype(bool))

    return individual_masks


# =========================== color: RGB->Lab ===========================
def srgb_to_linear_np(c01):
    return np.where(c01 <= 0.04045, c01 / 12.92, ((c01 + 0.055) / 1.055) ** 2.4)


def rgb_to_lab_batch_np(rgb255):
    rgb01 = np.asarray(rgb255, dtype=np.float64) / 255.0
    rlin, glin, blin = (srgb_to_linear_np(rgb01[:, 0]), srgb_to_linear_np(rgb01[:, 1]),
                       srgb_to_linear_np(rgb01[:, 2]))
    X = rlin * 0.4124564 + glin * 0.3575761 + blin * 0.1804375
    Y = rlin * 0.2126729 + glin * 0.7151522 + blin * 0.0721750
    Z = rlin * 0.0193339 + glin * 0.1191920 + blin * 0.9503041
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    d = 6.0 / 29.0

    def f(t):
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


# =========================== color: medicion robusta dentro de mascara ===========================
def measure_color_gt(img_uint8, mask_np, z_thresh=2.5):
    """dominant_color completo -- PCA en el plano a*b* + recorte de
    outliers por z-score sobre la proyeccion perpendicular al eje
    principal. Devuelve (L,a,b) o None si la mascara viene vacia/
    degenerada (ej. el objeto se rompio del todo con un shift extremo)."""
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


# =========================== color: CIEDE2000 (deltaE perceptual real, para Fase C) ===========================
def ciede2000(lab1, lab2):
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


# =========================== color: hex/rgb/lab -> Lab (parseo de targets) ===========================
def rgb_to_lab_single_np(rgb255):
    return rgb_to_lab_batch_np(np.asarray(rgb255, dtype=np.float32).reshape(1, 3))[0]


def lab_to_rgb_single_np(lab):
    """Convierte un punto Lab (L, a, b) escalar a sRGB [0..255] uint8."""
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    # Lab -> XYZ (D65, 2 deg)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def f_inv(t):
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xr, yr, zr = 0.95047, 1.00000, 1.08883
    X = xr * f_inv(fx)
    Y = yr * f_inv(fy)
    Z = zr * f_inv(fz)

    # XYZ -> linear RGB (sRGB D65)
    r_lin = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    def gamma(u):
        u_clamped = max(0.0, min(1.0, u))
        if u_clamped <= 0.0031308:
            return 12.92 * u_clamped
        return 1.055 * (u_clamped ** (1.0 / 2.4)) - 0.055

    r = int(round(gamma(r_lin) * 255.0))
    g = int(round(gamma(g_lin) * 255.0))
    b_val = int(round(gamma(b_lin) * 255.0))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b_val)))


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Parsea codigo hexadecimal a RGB (r, g, b).
    Tolera 6 digitos (#RRGGBB o RRGGBB), 3 digitos (#RGB),
    o errores comunes de tipeo (ej. 5 o 7 digitos completados o recortados a 6)."""
    s = hex_code.strip().lstrip("#")
    if len(s) == 3:
        return (int(s[0] * 2, 16), int(s[1] * 2, 16), int(s[2] * 2, 16))
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    if len(s) == 5:
        # Typo comun (ej #FFFF0 -> #FFFF00)
        s = s + "0"
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    if len(s) > 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    raise ValueError(f"hex invalido: {hex_code!r}")


def parse_target_color(spec: Any) -> Tuple[float, float, float]:
    """Acepta el color target en cualquier formato y devuelve SIEMPRE Lab (L, a, b):
      - hex: '#0000FF', '0000FF', '#FFF', etc.
      - rgb: (r, g, b) o [r, g, b] o 'rgb(0, 0, 255)' o '(0, 0, 255)' con valores 0..255
      - lab: dict {'lab': (L, a, b)} o 'lab(53.2, 79.2, -107.9)'
      - nombre comun de color: 'red', 'light blue', etc.
    """
    if isinstance(spec, dict) and "lab" in spec:
        return tuple(float(x) for x in spec["lab"])
    if isinstance(spec, (tuple, list, np.ndarray)) and len(spec) == 3:
        # Si los valores son float <= 1.0, verificar si es float rgb [0, 1]
        vals = [float(x) for x in spec]
        if all(0.0 <= v <= 1.0 for v in vals) and any(v > 0 and v < 1.0 for v in vals):
            vals = [v * 255.0 for v in vals]
        return tuple(rgb_to_lab_single_np(vals))
    if isinstance(spec, str):
        s = spec.strip()
        # Formato lab(L, a, b)
        if s.lower().startswith("lab(") and s.endswith(")"):
            inner = s[4:-1]
            parts = [float(x.strip()) for x in inner.split(",") if x.strip()]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        # Formato rgb(r, g, b)
        if s.lower().startswith("rgb(") and s.endswith(")"):
            inner = s[4:-1]
            parts = [float(x.strip()) for x in inner.split(",") if x.strip()]
            if len(parts) == 3:
                return tuple(rgb_to_lab_single_np(parts))
        # Formato "(r, g, b)" o "r, g, b"
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            inner = s[1:-1]
            try:
                parts = [float(x.strip()) for x in inner.split(",") if x.strip()]
                if len(parts) == 3:
                    return tuple(rgb_to_lab_single_np(parts))
            except ValueError:
                pass
        # Formato hex (con o sin #)
        cleaned = s.lstrip("#")
        if re.fullmatch(r"[0-9a-fA-F]{3,8}", cleaned):
            return tuple(rgb_to_lab_single_np(hex_to_rgb(s)))
        # Nombre de color conocido (CSS / PIL)
        try:
            from PIL import ImageColor
            rgb = ImageColor.getrgb(s)
            return tuple(rgb_to_lab_single_np(rgb))
        except Exception:
            pass

    raise ValueError(f"No se pudo interpretar el color target: {spec!r}")


# =========================== Prompt & Object Extraction Agnosticos ===========================
def extract_hex_from_text(text: str) -> Optional[str]:
    """Busca codigos hexadecimales en el texto."""
    match = re.search(r"#[0-9a-fA-F]{3,8}\b", text)
    if match:
        return match.group(0)
    match = re.search(r"\b[0-9a-fA-F]{6}\b", text)
    if match:
        return f"#{match.group(0)}"
    return None


def extract_color_spec_from_text(text: str) -> Optional[str]:
    """Busca cualquier especificacion numerica de color en el texto (hex, rgb(...), lab(...))."""
    # 1. Hex
    hex_match = extract_hex_from_text(text)
    if hex_match:
        return hex_match
    # 2. rgb(...)
    rgb_match = re.search(r"rgb\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", text, re.IGNORECASE)
    if rgb_match:
        return rgb_match.group(0)
    # 3. lab(...)
    lab_match = re.search(r"lab\s*\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)", text, re.IGNORECASE)
    if lab_match:
        return lab_match.group(0)
    return None


def extract_object_from_text(text: str) -> str:
    """Descubre dinamicamente el objeto objetivo a partir del prompt sin lista fija (hardcodeada).
    Soporta los patrones de GenColorBench (NCU) y lenguaje natural abierto.
    Ejemplos:
      - 'A photo of a [obj] in the color #HEX' -> '[obj]'
      - 'A [obj] colored with rgb(r, g, b)' -> '[obj]'
      - 'a photo of a ceramic mug on a wooden desk' -> 'ceramic mug' / 'mug'
    """
    clean_text = text.strip()

    # Patron 1: GenColorBench NCU "A photo/rendering/etc of a/an <OBJ> in the color / with color / at color ..."
    match = re.search(
        r"(?:photo|rendering|picture|image|shot|drawing|illustration)?\s*(?:of)?\s*(?:a|an|the)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:in\s+(?:the\s+)?color|with\s+(?:the\s+)?color|colored\s+(?:with|in|as)?|at\s+color|having\s+(?:the\s+)?color|painted\s+in)\b",
        clean_text,
        re.IGNORECASE
    )
    if match:
        obj = match.group(1).strip()
        obj = re.sub(r"^(?:photo|picture|image|rendering)\s+of\s+(?:a|an|the\s+)?", "", obj, flags=re.IGNORECASE).strip()
        if obj:
            return obj

    # Patron 2: "A/An <OBJ> in/with/at <COLOR_SPEC>"
    match = re.search(
        r"(?:a|an|the)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:in|with|at)\s+(?:#|rgb|lab)",
        clean_text,
        re.IGNORECASE
    )
    if match:
        obj = match.group(1).strip()
        if obj:
            return obj

    # Patron 3: GenColorBench color al principio: "[Color] <OBJ> ..." o "A/An [Color] <OBJ> ..."
    match = re.search(
        r"(?:a|an|the)?\s*(?:#[0-9a-fA-F]{3,8}|rgb\([^\)]+\)|lab\([^\)]+\))\s+([a-zA-Z0-9_\-]+)",
        clean_text,
        re.IGNORECASE
    )
    if match:
        obj = match.group(1).strip()
        if obj:
            return obj

    # Patron 4: "photo of a/an <OBJ> [context...]"
    match = re.search(
        r"(?:photo|rendering|picture|image|shot|drawing|illustration)\s+of\s+(?:a|an|the)\s+([a-zA-Z0-9_\-]+)",
        clean_text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # Fallback: primera palabra sustantiva tras articulo
    match = re.search(r"\b(?:a|an|the)\s+([a-zA-Z0-9_\-]+)", clean_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower()
        if candidate not in {"photo", "picture", "image", "rendering", "shot", "illustration"}:
            return candidate

    return "object"


def clean_prompt_for_diffusion(prompt: str, color_name: str) -> str:
    """Reemplaza codigos numericos de color en el prompt por un nombre semantico comprensible
    por el text encoder del modelo de difusion."""
    clean = prompt
    # Reemplazar patrones de color numerico
    clean = re.sub(r"#[0-9a-fA-F]{3,8}\b", color_name, clean)
    clean = re.sub(r"rgb\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", color_name, clean, flags=re.IGNORECASE)
    clean = re.sub(r"lab\s*\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)", color_name, clean, flags=re.IGNORECASE)

    # Suavizar frases sintacticas como "in the color blue" -> "colored blue" o "blue"
    clean = re.sub(rf"\bin the color {re.escape(color_name)}\b", f"colored {color_name}", clean, flags=re.IGNORECASE)
    clean = re.sub(rf"\bat color {re.escape(color_name)}\b", f"colored {color_name}", clean, flags=re.IGNORECASE)
    clean = re.sub(rf"\bwith the color {re.escape(color_name)}\b", f"colored {color_name}", clean, flags=re.IGNORECASE)
    return clean


# =========================== color: mapeo Lab -> nombre de color mas cercano ===========================
# Mismo diseño que utils.py de SDXL: el modelo de texto a imagen no entiende
# hex, asi que el color target preciso se traduce al NOMBRE CSS mas cercano
# para armar el prompt, y la precision real la aporta despues el shift del
# latente (que si usa el Lab exacto).
CHROMA_THRESHOLD = 15.0   # por debajo se considera "casi sin tono" (achromatico)
USE_FULL_COLOR_PALETTE = True   # True = todo ImageColor.colormap (~139 unicos tras dedup)

CURATED_COLOR_NAMES = [
    "white", "maroon","beige","gray", "black", "red", "orange", "yellow", "purple", "blue",
    "green", "pink", "brown", "cyan", "turquoise", "magenta", "olive", "teal",
    "lightblue", "navy", "lightgreen", "darkgreen", "lime", "coral", "gold", "silver"]


def _build_color_names_table(use_full=USE_FULL_COLOR_PALETTE):
    """Dedup por RGB (varios nombres CSS mapean al mismo color exacto, ej.
    'cyan'/'aqua') -- se queda con el primero en el orden de la fuente."""
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
    """Hibrido: el CHROMA decide a que GRUPO comparar (evita que un pastel
    claro con tono real termine en 'white' solo por luminosidad alta).
    DENTRO de cada grupo se usa CIEDE2000 completo. Devuelve (nombre,
    distancia_ciede2000)."""
    L, a, b = target_lab
    chroma = math.sqrt(a ** 2 + b ** 2)
    candidates = ACHROMATIC_NAMES if chroma < CHROMA_THRESHOLD else CHROMATIC_NAMES

    best_name, best_dist = None, float("inf")
    for name in candidates:
        d = ciede2000(target_lab, COLOR_NAMES_LAB[name])
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist
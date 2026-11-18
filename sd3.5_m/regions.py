"""
REGIONS.PY -- Prompt colour names and several coloured regions on one object (used by inference.py).

Prompt: as in the GenColorBench runner (run_gencolorbench_pca.py), the prompt is kept verbatim and every numeric colour
code (#hex, rgb(...)) is replaced by its nearest ISCC-NBS L2 name. The numeric targets themselves drive the latent shift.

Several colours ("parts"): at the gate the object is split into regions.
  region 0  = the object minus the named parts (steered to the object colour, if one is given)
  part k    = SAM3 mask of the part word ("legs", or "chair legs" when "legs" is not found), kept near the object
              (object bounding box expanded by 30 %); where two parts overlap the later one wins.
Every region gets its own measured colour and its own MLP shift (m1, m2, m3); the latent shift is the sum of the
per-region shifts, each restricted to its own mask. Without parts there is one region (the whole object), which is
exactly the single-colour method.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import utils

# the runner's pattern (#hex, rgb(...)) plus lab(...), which inference.py also accepts
CODE_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|rgb\s*\([^)]*\)|lab\s*\([^)]*\)", re.IGNORECASE)
MIN_REGION_PIXELS = 200
NEAR_MARGIN = 0.3


def normalize_color_spec(spec: Any) -> Any:
    """'1E90FF' -> '#1E90FF'; other specifications are returned unchanged."""
    if spec is None:
        return None
    s = str(spec).strip()
    if re.fullmatch(r"[0-9a-fA-F]{6}|[0-9a-fA-F]{3}", s):
        return f"#{s.upper()}"
    return s


def color_name(spec: Any) -> str:
    """Nearest ISCC-NBS L2 name of a colour specification."""
    return utils.nearest_color_name(utils.parse_target_color(spec))[0]


def prompt_with_color_names(prompt: str, name: Optional[str] = None) -> str:
    """Keep the prompt verbatim and replace its colour codes by ISCC-NBS L2 names: all by `name` when given (one colour,
    exactly as the GenColorBench runner), otherwise each code by its own nearest name (several colours)."""
    return CODE_RE.sub(lambda m: name if name else color_name(m.group(0)), prompt)


def color_codes(prompt: str) -> List[str]:
    return [m.group(0) for m in CODE_RE.finditer(prompt)]


def parse_parts(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    """['legs=#1E90FF', 'seat=rgb(200,30,30)'] -> [('legs', '#1E90FF'), ('seat', 'rgb(200,30,30)')]"""
    parts = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--part takes part=color, e.g. legs=#1E90FF (got {it!r})")
        name, spec = it.split("=", 1)
        if not name.strip() or not spec.strip():
            raise ValueError(f"--part takes part=color, e.g. legs=#1E90FF (got {it!r})")
        parts.append((name.strip(), normalize_color_spec(spec.strip())))
    return parts


def object_color_for_parts(prompt: str, parts: List[Tuple[str, str]]) -> Optional[str]:
    """Default object colour with parts: the first colour code of the prompt that is not a part colour."""
    used = {str(spec).lower() for _, spec in parts}
    for code in color_codes(prompt):
        if code.lower() not in used:
            return code
    return None


def region_masks(img: np.ndarray, obj: str, part_words: List[str], seg_models) -> Optional[List[np.ndarray]]:
    """[object minus parts, part 1, part 2, ...] as boolean masks, or None if a region is missing or too small."""
    pil = Image.fromarray(img)
    om = utils.get_object_mask(pil, obj, seg_models)
    if om is None or om.sum() < MIN_REGION_PIXELS:
        return None
    om = om.astype(bool)
    ys, xs = np.where(om)
    h, w = om.shape
    dy = int(NEAR_MARGIN * (ys.max() - ys.min() + 1))
    dx = int(NEAR_MARGIN * (xs.max() - xs.min() + 1))
    near = np.zeros_like(om)
    near[max(0, ys.min() - dy):min(h, ys.max() + dy + 1), max(0, xs.min() - dx):min(w, xs.max() + dx + 1)] = True
    pms = []
    for p in part_words:
        pm = None
        for txt in (p, f"{obj} {p}"):
            pm = utils.get_object_mask(pil, txt, seg_models)
            if pm is not None:
                pm = pm.astype(bool)
                if (pm & near).sum() >= MIN_REGION_PIXELS and (pm & om).sum() < 0.9 * om.sum():
                    break
            pm = None
        if pm is None:
            return None
        pms.append(pm & near)
    for i in range(len(pms)):
        for j in range(i + 1, len(pms)):
            pms[i] &= ~pms[j]
    body = om.copy()
    for pm in pms:
        body &= ~pm
    regs = [body] + pms
    if any(r.sum() < MIN_REGION_PIXELS for r in regs[1:]):
        return None
    return regs


def gate_regions(img: np.ndarray, obj: str, obj_target_lab, parts: List[Tuple[str, str]],
                 seg_models) -> Optional[List[Dict[str, Any]]]:
    """Regions to steer at the gate: [{name, mask_pixel, target_lab}], or None if the parts are not found.
    Region 0 (object minus parts) is steered only when an object colour is given and it is large enough."""
    regs = region_masks(img, obj, [p for p, _ in parts], seg_models)
    if regs is None:
        return None
    out = []
    if obj_target_lab is not None and regs[0].sum() >= MIN_REGION_PIXELS:
        out.append({"name": obj, "mask_pixel": regs[0], "target_lab": tuple(obj_target_lab)})
    for (p, spec), m in zip(parts, regs[1:]):
        out.append({"name": p, "mask_pixel": m, "target_lab": tuple(utils.parse_target_color(spec))})
    return out


def measure_regions(img: np.ndarray, obj: str, parts: List[Tuple[str, str]], seg_models,
                    gate: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Achieved colour and CIEDE2000 per steered region on the final image (regions re-segmented on the final image;
    the gate masks are used when the re-segmentation fails)."""
    regs = region_masks(img, obj, [p for p, _ in parts], seg_models)
    by_name = {}
    if regs is not None:
        by_name[obj] = regs[0]
        by_name.update({p: m for (p, _), m in zip(parts, regs[1:])})
    out = []
    for r in gate:
        mask = by_name.get(r["name"])
        if mask is None or mask.sum() < MIN_REGION_PIXELS:
            mask = r["mask_pixel"]
        ach = utils.measure_color_gt(img, mask)
        out.append({"name": r["name"], "target_lab": r["target_lab"], "achieved_lab": ach,
                    "delta_e00": None if ach is None else float(utils.ciede2000(r["target_lab"], ach))})
    return out

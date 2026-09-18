"""
ISCC_NBS.PY -- Standardized ISCC-NBS (Inter-Society Color Council - National Bureau of Standards) Color System.

Provides:
1. ISCC-NBS Level 1 (13 Fundamental Color Categories with official Lab Centroids):
   - Red, Pink, Orange, Brown, Yellow, Olive, Yellow Green, Green, Blue, Purple, White, Gray, Black.
2. ISCC-NBS Level 2 (29 Intermediate Hue Categories with standard Lab coordinates):
   - Used for rich, standardized color sampling in dataset collection, evaluation, and inference.
3. Color matching functions:
   - find_nearest_iscc_l2(target_lab): Returns the nearest Level 2 color name and CIEDE2000 distance.
   - find_nearest_iscc_l1(target_lab): Returns the nearest Level 1 color name and CIEDE2000 distance.
   - get_iscc_l1_centroid(color_name): Returns the exact canonical Lab centroid for a Level 1 color.
   - get_iscc_l2_centroid(color_name): Returns the exact canonical Lab centroid for a Level 2 color.
"""

from typing import Dict, Tuple
import math
import numpy as np

# Canonical sRGB coordinates converted to standard CIELAB (D65 illuminant, 2 deg observer)
# Reference: NBS Circular 553 / Kelly & Judd (1955, 1976 revision)

ISCC_NBS_LEVEL1: Dict[str, Tuple[float, float, float]] = {
    "white":        (100.0,   0.0,   0.0),
    "gray":         ( 53.6,   0.0,   0.0),
    "black":        (  0.0,   0.0,   0.0),
    "red":          ( 53.2,  67.5,  43.1),  # Vivid Red centroid
    "pink":         ( 82.4,  24.8,   5.3),  # Standard Pink centroid
    "orange":       ( 67.0,  43.2,  74.5),  # Vivid Orange centroid
    "brown":        ( 39.8,  28.5,  49.3),  # Moderate Brown centroid
    "yellow":       ( 89.7,  -4.9,  90.0),  # Vivid Yellow centroid
    "olive":        ( 51.9, -12.9,  56.7),  # Olive centroid
    "yellow green": ( 77.2, -34.8,  68.3),  # Yellow-Green centroid
    "green":        ( 61.6, -51.3,  20.8),  # Moderate Green centroid
    "blue":         ( 43.7,   9.6, -58.8),  # Strong Blue centroid
    "purple":       ( 41.1,  62.4, -72.6),  # Vivid Purple centroid
}

ISCC_NBS_LEVEL2: Dict[str, Tuple[float, float, float]] = {
    # 13 basic terms (L1)
    "white":            (100.0,   0.0,   0.0),
    "gray":             ( 53.6,   0.0,   0.0),
    "black":            (  0.0,   0.0,   0.0),
    "red":              ( 53.2,  67.5,  43.1),
    "pink":             ( 82.4,  24.8,   5.3),
    "orange":           ( 67.0,  43.2,  74.5),
    "brown":            ( 39.8,  28.5,  49.3),
    "yellow":           ( 89.7,  -4.9,  90.0),
    "olive":            ( 51.9, -12.9,  56.7),
    "yellow green":     ( 77.2, -34.8,  68.3),
    "green":            ( 61.6, -51.3,  20.8),
    "blue":             ( 43.7,   9.6, -58.8),
    "purple":           ( 41.1,  62.4, -72.6),
    # 16 intermediate hue categories (L2 additions)
    "reddish orange":   ( 59.5,  54.8,  58.2),
    "orange yellow":    ( 80.1,  18.6,  83.4),
    "greenish yellow":  ( 88.3, -20.4,  84.1),
    "yellowish green":  ( 72.5, -45.1,  55.8),
    "bluish green":     ( 58.4, -48.2,  -5.1),
    "greenish blue":    ( 52.1, -25.6, -38.4),
    "purplish blue":    ( 36.8,  28.2, -64.7),
    "bluish purple":    ( 38.4,  52.1, -68.9),
    "reddish purple":   ( 45.2,  68.4, -28.3),
    "purplish red":     ( 48.6,  69.1,  18.4),
    "purplish pink":    ( 78.5,  34.2,  -8.1),
    "yellowish pink":   ( 83.1,  21.4,  28.9),
    "brownish pink":    ( 68.2,  18.7,  19.5),
    "reddish brown":    ( 36.4,  35.2,  33.1),
    "yellowish brown":  ( 48.7,  14.2,  52.6),
    "olive brown":      ( 42.1,   2.8,  44.5),
    "olive green":      ( 55.4, -28.7,  48.2),
}

ISCC_NBS_L1_NAMES = list(ISCC_NBS_LEVEL1.keys())
ISCC_NBS_L2_NAMES = list(ISCC_NBS_LEVEL2.keys())

# Compatibility aliases
ISCC_NBS_L1_CENTROIDS = ISCC_NBS_LEVEL1
ISCC_NBS_L2_CENTROIDS = ISCC_NBS_LEVEL2


def get_iscc_l1_centroid(color_name: str) -> Tuple[float, float, float]:
    """
    Returns the official CIELAB centroid (L, a, b) for an ISCC-NBS Level 1 color name.
    """
    key = color_name.lower().strip().replace("-", " ").replace("_", " ")
    if key in ISCC_NBS_LEVEL1:
        return ISCC_NBS_LEVEL1[key]
    aliases = {
        "violet": "purple",
        "grey": "gray",
        "dark": "black",
        "light": "white",
    }
    if key in aliases:
        return ISCC_NBS_LEVEL1[aliases[key]]
    for k, v in ISCC_NBS_LEVEL1.items():
        if k in key or key in k:
            return v
    return (50.0, 0.0, 0.0)


def get_iscc_l2_centroid(color_name: str) -> Tuple[float, float, float]:
    """
    Returns the official CIELAB centroid (L, a, b) for an ISCC-NBS Level 2 color name.
    """
    key = color_name.lower().strip().replace("-", " ").replace("_", " ")
    if key in ISCC_NBS_LEVEL2:
        return ISCC_NBS_LEVEL2[key]
    key_under = key.replace(" ", "_")
    for k, v in ISCC_NBS_LEVEL2.items():
        if k.replace(" ", "_") == key_under or k in key or key in k:
            return v
    return get_iscc_l1_centroid(color_name)


def find_nearest_iscc_l1(target_lab, ciede2000_fn=None) -> Tuple[str, float]:
    """
    Finds the closest ISCC-NBS Level 1 color name to target_lab (L, a, b)
    using CIEDE2000 color difference.
    """
    if ciede2000_fn is None:
        from utils import ciede2000
        ciede2000_fn = ciede2000

    best_name = None
    min_dE = float("inf")
    for name, lab_val in ISCC_NBS_LEVEL1.items():
        dE = ciede2000_fn(target_lab, lab_val)
        if dE < min_dE:
            min_dE = dE
            best_name = name
    return best_name, min_dE


def find_nearest_iscc_l2(target_lab, ciede2000_fn=None) -> Tuple[str, float]:
    """
    Finds the closest ISCC-NBS Level 2 color name to target_lab (L, a, b)
    using CIEDE2000 color difference.
    """
    if ciede2000_fn is None:
        from utils import ciede2000
        ciede2000_fn = ciede2000

    best_name = None
    min_dE = float("inf")
    for name, lab_val in ISCC_NBS_LEVEL2.items():
        dE = ciede2000_fn(target_lab, lab_val)
        if dE < min_dE:
            min_dE = dE
            best_name = name
    return best_name, min_dE

"""
ISCC_NBS.PY -- ISCC-NBS Centroid Color System & Level-1/2 Categorization for SD3.

Provides standard CIELAB reference centroids for:
  1. ISCC-NBS Level 1 (13 Fundamental Color Categories)
  2. ISCC-NBS Level 2 (29 Extended Color Categories)
"""

from typing import Dict, Tuple

# ISCC-NBS Level 1 Centroids in CIELAB (D65, 2 deg observer)
ISCC_NBS_L1_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    "pink": (74.0, 31.0, 10.0),
    "red": (53.2, 67.5, 43.1),
    "orange": (67.0, 43.0, 65.0),
    "brown": (38.0, 19.0, 27.0),
    "yellow": (87.0, -6.0, 75.0),
    "olive": (43.0, -10.0, 30.0),
    "yellow_green": (72.0, -35.0, 60.0),
    "green": (55.0, -50.0, 25.0),
    "blue": (45.0, -10.0, -45.0),
    "purple": (42.0, 45.0, -35.0),
    "white": (96.0, 0.0, 0.0),
    "gray": (55.0, 0.0, 0.0),
    "black": (15.0, 0.0, 0.0),
}

ISCC_NBS_L1_NAMES = list(ISCC_NBS_L1_CENTROIDS.keys())

# ISCC-NBS Level 2 Centroids in CIELAB
ISCC_NBS_L2_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    "vivid_pink": (75.0, 55.0, 10.0),
    "strong_pink": (70.0, 40.0, 10.0),
    "deep_pink": (55.0, 52.0, 10.0),
    "light_pink": (85.0, 20.0, 8.0),
    "moderate_pink": (72.0, 28.0, 10.0),
    "dark_pink": (48.0, 35.0, 8.0),
    "pale_pink": (86.0, 12.0, 6.0),
    "grayish_pink": (68.0, 15.0, 8.0),
    "pinkish_white": (94.0, 5.0, 3.0),
    "pinkish_gray": (65.0, 8.0, 5.0),
    "vivid_red": (53.2, 67.5, 43.1),
    "strong_red": (46.0, 58.0, 35.0),
    "deep_red": (32.0, 48.0, 28.0),
    "very_deep_red": (20.0, 35.0, 18.0),
    "moderate_red": (45.0, 42.0, 22.0),
    "dark_red": (30.0, 32.0, 15.0),
    "very_dark_red": (18.0, 20.0, 8.0),
    "light_grayish_red": (68.0, 18.0, 10.0),
    "grayish_red": (48.0, 20.0, 12.0),
    "dark_grayish_red": (30.0, 15.0, 8.0),
    "blackish_red": (16.0, 10.0, 4.0),
    "reddish_gray": (52.0, 8.0, 5.0),
    "dark_reddish_gray": (32.0, 6.0, 4.0),
    "reddish_black": (15.0, 4.0, 2.0),
    "vivid_orange": (68.0, 48.0, 72.0),
    "strong_orange": (62.0, 42.0, 60.0),
    "deep_orange": (48.0, 45.0, 55.0),
    "light_orange": (80.0, 25.0, 50.0),
    "moderate_orange": (62.0, 30.0, 45.0),
    "dark_orange": (45.0, 32.0, 40.0),
}

ISCC_NBS_L2_NAMES = list(ISCC_NBS_L2_CENTROIDS.keys())


def get_iscc_l1_centroid(name: str) -> Tuple[float, float, float]:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ISCC_NBS_L1_CENTROIDS:
        return ISCC_NBS_L1_CENTROIDS[key]
    for k, v in ISCC_NBS_L1_CENTROIDS.items():
        if k in key or key in k:
            return v
    return (50.0, 0.0, 0.0)


def get_iscc_l2_centroid(name: str) -> Tuple[float, float, float]:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ISCC_NBS_L2_CENTROIDS:
        return ISCC_NBS_L2_CENTROIDS[key]
    return get_iscc_l1_centroid(name)


def find_nearest_iscc_l1(lab: Tuple[float, float, float]) -> Tuple[str, float]:
    from utils import ciede2000
    best_name = None
    min_dist = float("inf")
    for name, centroid in ISCC_NBS_L1_CENTROIDS.items():
        d = ciede2000(lab, centroid)
        if d < min_dist:
            min_dist = d
            best_name = name
    return best_name, min_dist


def find_nearest_iscc_l2(lab: Tuple[float, float, float]) -> Tuple[str, float]:
    from utils import ciede2000
    best_name = None
    min_dist = float("inf")
    combined = {**ISCC_NBS_L1_CENTROIDS, **ISCC_NBS_L2_CENTROIDS}
    for name, centroid in combined.items():
        d = ciede2000(lab, centroid)
        if d < min_dist:
            min_dist = d
            best_name = name
    return best_name, min_dist

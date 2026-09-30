"""
COLECCION_DATOS_MLP_PCA.PY -- Large-Scale Dataset Collection for Flow-Matching MLP in PCA Space.

Generates ~25,000 training samples across 85 diverse objects and ISCC-NBS Level 2 color categories.
Applies continuous multi-axial 3D spherical direction vectors projected onto the 16D PCA basis.
Calibrated magnitude sampling:
  - 80% Zone 1 (m in [0.05, 0.45]): High-fidelity linear color regime.
  - 20% Zone 2 (m in [0.45, 1.25]): Transition & saturation regime.
"""

import os
import sys
import csv
import json
import argparse
import random
import numpy as np
import torch
from PIL import Image

# Ensure parent directory is in sys.path for flux_core and utils
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from flux_core import (
    build_envelope_bands, setup_flux, latent_hw, unpack_to_4d, decode_latents_4d,
    build_mask_latent, run_generation, get_gpu_list, spawn_workers,
)
from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000
from iscc_nbs import ISCC_NBS_LEVEL1, ISCC_NBS_LEVEL2, ISCC_NBS_L1_NAMES, ISCC_NBS_L2_NAMES

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
except ImportError as _e:
    raise ImportError(f"Failed to import from skimage: {_e}")

# =========================== CONFIGURATION ===========================
MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5
REF_MODE = "none"

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = str(SCRIPT_DIR / "coleccion_datos_mlp_pca_out")
PCA_AXES_PATH = str(SCRIPT_DIR / "fase_a_pca_out" / "pca_axes.json")
WINNING_SCHEDULE_PATH = str(SCRIPT_DIR / "fase_b_pca_out" / "fase_b_winning_schedule.json")

# Dataset scale configuration
N_BASELINES_TOTAL = 3125      # 3125 baselines x 8 variants = 25,000 samples
K_VARIANTS_PER_BASELINE = 8
MASTER_SEED = 20260825

# Calibrated magnitude sampling probabilities
P_ZONE1_LINEAR = 0.80         # 80% in Zone 1 (m in [0.05, 0.45])
ZONE1_RANGE = (0.05, 0.45)
ZONE2_RANGE = (0.45, 1.25)    # 20% in Zone 2 (m in [0.45, 1.25])

# Load PCA vectors
with open(PCA_AXES_PATH) as f:
    pca_data = json.load(f)["axes"]

U1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
U2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
U3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
PCA_BASIS = np.stack([U1, U2, U3], axis=1)  # shape: (16, 3)

# Load winning schedule
if os.path.exists(WINNING_SCHEDULE_PATH):
    with open(WINNING_SCHEDULE_PATH) as f:
        sched_cfg = json.load(f)
else:
    sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

PERFIL_GENERATORS = {
    "plano":        lambda n: [1.0] * n,
    "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
    "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
    "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
    "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
}

BANDS = build_envelope_bands(
    sched_cfg["gate_frac"],
    PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
    "ramp_down"
)

# =========================== 100 DIVERSE OBJECTS POOL ===========================
OBJECT_SCENES = [
    # 1-16: Fruits, Vegetables & Organic Produce
    ("apple", "apple", ["on a wooden cutting board", "on a kitchen counter", "on a plain white background"]),
    ("banana", "banana", ["on a fruit bowl", "on a kitchen table", "on a marble counter"]),
    ("orange", "orange", ["on a wooden table", "on a cutting board", "in a fruit basket"]),
    ("lemon", "lemon", ["on a white plate", "on a wooden countertop", "on a cutting board"]),
    ("pear", "pear", ["on a ceramic plate", "on a rustic table", "on a kitchen counter"]),
    ("bell pepper", "pepper", ["on a wooden cutting board", "on a marble counter", "in a vegetable basket"]),
    ("tomato", "tomato", ["on a white plate", "on a wooden countertop", "on a cutting board"]),
    ("strawberry", "strawberry", ["on a white ceramic saucer", "on a marble countertop", "on a wooden table"]),
    ("carrot", "carrot", ["on a wooden cutting board", "on a kitchen counter", "on a table"]),
    ("avocado", "avocado", ["on a marble countertop", "on a cutting board", "on a white plate"]),
    ("eggplant", "eggplant", ["on a wooden table", "on a kitchen counter", "in a vegetable basket"]),
    ("pumpkin", "pumpkin", ["on a wooden porch", "on a rustic table", "on a clean floor"]),
    ("broccoli", "broccoli", ["on a white plate", "on a wooden cutting board", "on a kitchen counter"]),
    ("mushroom", "mushroom", ["on a wooden surface", "on a kitchen board", "on a clean plate"]),
    ("grapes", "grapes", ["on a fruit bowl", "on a ceramic platter", "on a dining table"]),
    ("watermelon", "watermelon", ["on a large cutting board", "on a kitchen counter", "on a wooden table"]),

    # 17-56: Canonical & Everyday Objects (Original 40 Categories)
    ("phone", "phone", ["on a wooden desk", "on a table", "on a nightstand"]),
    ("pencil", "pencil", ["on a desk", "on a notepad", "in an office"]),
    ("mug", "mug", ["on a kitchen table", "on a wooden desk", "on a marble counter"]),
    ("backpack", "backpack", ["on the floor", "on a wooden chair", "on a bench"]),
    ("bicycle", "bicycle", ["on a street", "in a park", "by a brick wall"]),
    ("umbrella", "umbrella", ["on a stand", "on a porch", "in an entryway"]),
    ("cap", "cap", ["on a wooden shelf", "on a table", "on a hat rack"]),
    ("balloon", "balloon", ["in a white room", "at a studio", "in a celebration hall"]),
    ("vase", "vase", ["on a marble table", "on a windowsill", "on a dining table"]),
    ("kettle", "kettle", ["on a kitchen counter", "on a stove", "on a dining tray"]),
    ("lamp", "lamp", ["on a nightstand", "on a desk", "in a living room"]),
    ("pillow", "pillow", ["on a modern sofa", "on a bed", "on an armchair"]),
    ("scarf", "scarf", ["on a chair", "on a wooden table", "on a coat rack"]),
    ("mailbox", "mailbox", ["on a green lawn", "by a fence", "on a suburban street"]),
    ("kite", "kite", ["on the grass", "in a park", "on a sandy beach"]),
    ("watering can", "can", ["in a garden", "on a patio", "by a potted plant"]),
    ("wallet", "wallet", ["on a wooden desk", "on a table", "on a nightstand"]),
    ("book", "book", ["on a wooden shelf", "on a coffee table", "on a study desk"]),
    ("bottle", "bottle", ["on a marble counter", "on a table", "in a gym"]),
    ("skateboard", "skateboard", ["on a concrete pavement", "in a room", "on a ramp"]),
    ("helmet", "helmet", ["on a wooden shelf", "on a desk", "on a motorbike seat"]),
    ("basket", "basket", ["on a wooden table", "on the floor", "in a pantry"]),
    ("teapot", "teapot", ["on a kitchen counter", "on a tray", "on a dining table"]),
    ("frisbee", "frisbee", ["on the lawn", "on a park bench", "on the grass"]),
    ("suitcase", "suitcase", ["on a clean floor", "by a chair", "in an airport lounge"]),
    ("candle", "candle", ["on a marble tray", "on a table", "on a fireplace mantel"]),
    ("guitar", "guitar", ["on a stand", "against a wall", "in a studio"]),
    ("chair", "chair", ["in a minimalist studio", "by a desk", "in a living room"]),
    ("clock", "clock", ["on a desk", "on a wooden shelf", "on a nightstand"]),
    ("headphones", "headphones", ["on a wooden desk", "on a stand", "by a laptop"]),
    ("sneaker", "sneaker", ["on a clean floor", "in a studio", "on a display stand"]),
    ("cup", "cup", ["on a wooden saucer", "on a table", "in a cafe"]),
    ("bowl", "bowl", ["on a marble countertop", "on a table", "in a kitchen"]),
    ("plate", "plate", ["on a dining table", "on a kitchen counter", "on a placemat"]),
    ("sunglasses", "sunglasses", ["on a wooden desk", "on a table", "by a pool"]),
    ("glove", "glove", ["on a table", "on a wooden surface", "on a bench"]),
    ("pot", "pot", ["on a stove", "on a kitchen counter", "on a dining mat"]),
    ("toaster", "toaster", ["on a kitchen counter", "on a marble island", "in a pantry"]),
    ("thermos", "thermos", ["on a desk", "on a wooden bench", "in a backpack"]),
    ("bucket", "bucket", ["on the floor", "on a patio", "in a garage"]),

    # 57-78: Consumer Electronics, Tools & Appliances
    ("laptop", "laptop", ["on a modern desk", "on a clean table", "in a minimalist workspace"]),
    ("computer mouse", "mouse", ["on a mousepad", "on a desk", "by a laptop"]),
    ("keyboard", "keyboard", ["on a clean desk", "on a workspace", "in an office"]),
    ("smartwatch", "smartwatch", ["on a wooden nightstand", "on a charger", "on a desk"]),
    ("camera", "camera", ["on a wooden table", "on a tripod", "on a camera strap"]),
    ("microphone", "microphone", ["on a studio stand", "on a desk", "in a sound booth"]),
    ("blender", "blender", ["on a marble countertop", "in a kitchen", "on an island counter"]),
    ("coffee maker", "maker", ["on a kitchen counter", "in a cafe", "in an office breakroom"]),
    ("iron", "iron", ["on an ironing board", "on a counter", "in a laundry room"]),
    ("hairdryer", "hairdryer", ["on a bathroom counter", "on a vanity", "in a salon"]),
    ("flashlight", "flashlight", ["on a workbench", "on a table", "on a shelf"]),
    ("binoculars", "binoculars", ["on a wooden table", "on a window ledge", "on a desk"]),
    ("padlock", "padlock", ["on a wooden table", "on a shelf", "on a workbench"]),
    ("pliers", "pliers", ["on a workshop table", "on a workbench", "in a tool kit"]),
    ("screwdriver", "screwdriver", ["on a wooden workbench", "on a tool tray", "on a desk"]),
    ("tape measure", "tape", ["on a wooden table", "on a workbench", "in a tool box"]),
    ("stapler", "stapler", ["on an office desk", "on a table", "on a workstation"]),
    ("calculator", "calculator", ["on a study desk", "on a table", "in a classroom"]),
    ("desk fan", "fan", ["on a wooden desk", "on a nightstand", "on a table"]),
    ("alarm clock", "clock", ["on a wooden nightstand", "on a table", "by a bed"]),
    ("radio", "radio", ["on a wooden shelf", "on a table", "on a counter"]),
    ("game controller", "controller", ["on a coffee table", "on a desk", "by a TV"]),

    # 79-100: Toys, Vehicles, Musical Instruments & Decor
    ("drone", "drone", ["on a concrete floor", "on a table", "in an open space"]),
    ("vr headset", "headset", ["on a clean desk", "on a stand", "on a coffee table"]),
    ("toy car", "car", ["on a plain white background", "on a table", "on a wooden floor"]),
    ("rubber duck", "duck", ["on a clean surface", "in a studio", "on a bathroom counter"]),
    ("teddy bear", "bear", ["on a wooden chair", "on a bed", "on a playroom rug"]),
    ("action figure", "figure", ["on a display shelf", "on a table", "on a desk"]),
    ("toy robot", "robot", ["on a play table", "on a wooden floor", "on a shelf"]),
    ("soccer ball", "ball", ["on the grass", "in a studio", "on a gym floor"]),
    ("basketball", "basketball", ["on a gym floor", "in a studio", "on an asphalt court"]),
    ("tennis racket", "racket", ["on a court bench", "on a table", "in a sports bag"]),
    ("baseball glove", "glove", ["on a wooden bench", "on the grass", "on a field table"]),
    ("roller skates", "skates", ["on a pavement", "on the floor", "in an entryway"]),
    ("scooter", "scooter", ["on a sidewalk", "in a park", "on a driveway"]),
    ("violin", "violin", ["on a velvet stand", "on a table", "in a music case"]),
    ("trumpet", "trumpet", ["on a music stand", "on a table", "on a cloth surface"]),
    ("drum", "drum", ["on a stage", "in a studio", "in a music room"]),
    ("flute", "flute", ["on a velvet cloth", "on a music desk", "in a case"]),
    ("harmonica", "harmonica", ["on a wooden table", "on a desk", "on a sheet of music"]),
    ("picture frame", "frame", ["on a wooden mantel", "on a desk", "on a bookshelf"]),
    ("hourglass", "hourglass", ["on a wooden desk", "on a shelf", "on a study table"]),
    ("globe", "globe", ["on a study desk", "on a library shelf", "on an office table"]),
    ("smooth matte sphere", "sphere", ["on a plain white background", "in a studio", "on a pedestal"]),
]

# Color Pool: ISCC-NBS Level 1 (Core 30% weight) + Level 2 (Rest 70% weight)
CORE_COLORS = ISCC_NBS_L1_NAMES
L2_COLORS = ISCC_NBS_L2_NAMES

CSV_FIELDS = [
    "sample_id", "baseline_id", "variant_idx", "obj_name", "obj_word", "base_color_name",
    "prompt", "seed", "m1", "m2", "m3", "magnitude_total",
    "base_L", "base_a", "base_b", "mod_L", "mod_a", "mod_b",
    "delta_L", "delta_a", "delta_b", "deltaE",
    "ssim_in", "psnr_in", "note"
]


# =========================== HELPER FUNCTIONS ===========================
def gen(pipe, prompt, seed, device, direction=None, magnitude=None, bands=None, mask_latent=None):
    return run_generation(
        pipe, prompt, int(seed), RESOLUTION, RESOLUTION, device, STEPS, GUIDANCE,
        channel_idxs=None, direction=direction, combo=None, magnitude=magnitude, bands=bands,
        mask_latent=mask_latent, ref_mode=REF_MODE, ref_channels=None
    )


def calculate_psnr_masked(img1_np, img2_np, mask_2d=None):
    if mask_2d is not None and mask_2d.sum() > 0:
        diff = (img1_np.astype(np.float32) - img2_np.astype(np.float32))[mask_2d]
        mse = float(np.mean(diff ** 2))
    else:
        mse = float(np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def sample_sphere_direction():
    """Generates a uniform random 3D unit vector d on S^2."""
    g = np.random.normal(0, 1, 3)
    norm = np.linalg.norm(g)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (g / norm).astype(np.float32)


def sample_calibrated_magnitude():
    """Samples magnitude following 80% Zone 1 and 20% Zone 2 calibrated distribution."""
    if random.random() < P_ZONE1_LINEAR:
        return float(random.uniform(ZONE1_RANGE[0], ZONE1_RANGE[1]))
    else:
        return float(random.uniform(ZONE2_RANGE[0], ZONE2_RANGE[1]))


def build_all_dataset_tasks(n_baselines, k_variants, master_seed):
    """Deterministically pre-generates the entire balanced baseline task list."""
    random.seed(master_seed)
    np.random.seed(master_seed)

    tasks = []
    for b_idx in range(n_baselines):
        # Pick object and scene
        obj_name, obj_word, scenes = random.choice(OBJECT_SCENES)
        scene = random.choice(scenes)

        # Pick color (30% core L1, 70% intermediate L2)
        if random.random() < 0.30:
            color_name = random.choice(CORE_COLORS)
        else:
            color_name = random.choice(L2_COLORS)

        prompt = f"a photo of a {color_name} {obj_name} {scene}, studio lighting, high quality, 8k"
        seed = int(random.randint(1000, 99999999))

        # Build K variants for this baseline
        variant_specs = []
        for v_idx in range(k_variants):
            d_unit = sample_sphere_direction()
            mag = sample_calibrated_magnitude()
            m_vec = d_unit * mag  # [m1, m2, m3] in PC space

            # Project into 16D latent direction: v_16d = U * m_vec
            v_16d = PCA_BASIS @ m_vec
            norm_16d = float(np.linalg.norm(v_16d))
            if norm_16d > 1e-8:
                direction_list = [(c, float(v_16d[c] / norm_16d)) for c in range(NUM_LATENT_CHANNELS)]
            else:
                direction_list = [(c, 0.0) for c in range(NUM_LATENT_CHANNELS)]

            variant_specs.append({
                "variant_idx": v_idx,
                "m1": float(m_vec[0]),
                "m2": float(m_vec[1]),
                "m3": float(m_vec[2]),
                "magnitude_total": float(mag),
                "direction_list": direction_list,
            })

        tasks.append({
            "baseline_id": b_idx,
            "obj_name": obj_name,
            "obj_word": obj_word,
            "base_color_name": color_name,
            "prompt": prompt,
            "seed": seed,
            "variants": variant_specs,
        })
    return tasks


# =========================== WORKER ROUTINE ===========================
def run_worker(chunk_path, out_dir, pipe, vae, seg_models):
    pid = os.getpid()
    with open(chunk_path) as f:
        baseline_tasks = json.load(f)
    print(f"[worker pid={pid}] Assigned {len(baseline_tasks)} baselines ({len(baseline_tasks) * K_VARIANTS_PER_BASELINE} variants)", flush=True)

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"dataset_part_{pid}.csv")

    # Resume support: read already finished baseline_ids in this worker's part
    finished_baselines = set()
    if os.path.exists(part_path):
        with open(part_path, newline="") as f_in:
            reader = csv.DictReader(f_in)
            for r in reader:
                finished_baselines.add(int(r["baseline_id"]))

    file_exists = os.path.exists(part_path)
    with open(part_path, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for b_task in baseline_tasks:
            b_id = b_task["baseline_id"]
            if b_id in finished_baselines:
                continue

            prompt = b_task["prompt"]
            seed = b_task["seed"]
            obj_name = b_task["obj_name"]
            obj_word = b_task["obj_word"]
            base_color = b_task["base_color_name"]

            # 1. Generate Baseline
            lat_base = gen(pipe, prompt, seed, DEVICE)
            if torch.isnan(lat_base).any():
                print(f"[worker pid={pid}] Baseline NaN for b_id={b_id}, skipping...", flush=True)
                continue

            img_base = decode_latents_4d(vae, unpack_to_4d(pipe, lat_base, RESOLUTION, RESOLUTION))
            mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
            if mask_pixel is None or mask_pixel.sum() < 50:
                # Mask detection failed, skip baseline to guarantee clean supervision
                continue

            base_lab = measure_color_gt(img_base, mask_pixel)
            if base_lab is None:
                continue

            lh, lw = latent_hw(pipe, RESOLUTION, RESOLUTION)
            mask_latent = build_mask_latent(mask_pixel, lh, lw, DEVICE)

            # 2. Generate all K variants for this baseline
            for v in b_task["variants"]:
                v_idx = v["variant_idx"]
                sample_id = f"b{b_id:06d}_v{v_idx:02d}"

                lat_mod = gen(
                    pipe, prompt, seed, DEVICE, direction=v["direction_list"],
                    magnitude=v["magnitude_total"], bands=BANDS, mask_latent=mask_latent
                )
                if torch.isnan(lat_mod).any():
                    continue

                img_mod = decode_latents_4d(vae, unpack_to_4d(pipe, lat_mod, RESOLUTION, RESOLUTION))
                mod_lab = measure_color_gt(img_mod, mask_pixel)
                if mod_lab is None:
                    continue

                deltaE = float(ciede2000(base_lab, mod_lab))
                ssim_in = float(structural_similarity(img_base, img_mod, channel_axis=2, data_range=255))
                psnr_in = calculate_psnr_masked(img_base, img_mod, mask_pixel)

                delta_L = float(mod_lab[0] - base_lab[0])
                delta_a = float(mod_lab[1] - base_lab[1])
                delta_b = float(mod_lab[2] - base_lab[2])

                row = {
                    "sample_id": sample_id,
                    "baseline_id": b_id,
                    "variant_idx": v_idx,
                    "obj_name": obj_name,
                    "obj_word": obj_word,
                    "base_color_name": base_color,
                    "prompt": prompt,
                    "seed": seed,
                    "m1": f"{v['m1']:.4f}",
                    "m2": f"{v['m2']:.4f}",
                    "m3": f"{v['m3']:.4f}",
                    "magnitude_total": f"{v['magnitude_total']:.4f}",
                    "base_L": f"{base_lab[0]:.2f}",
                    "base_a": f"{base_lab[1]:.2f}",
                    "base_b": f"{base_lab[2]:.2f}",
                    "mod_L": f"{mod_lab[0]:.2f}",
                    "mod_a": f"{mod_lab[1]:.2f}",
                    "mod_b": f"{mod_lab[2]:.2f}",
                    "delta_L": f"{delta_L:.2f}",
                    "delta_a": f"{delta_a:.2f}",
                    "delta_b": f"{delta_b:.2f}",
                    "deltaE": f"{deltaE:.3f}",
                    "ssim_in": f"{ssim_in:.4f}",
                    "psnr_in": f"{psnr_in:.2f}",
                    "note": "ok",
                }
                writer.writerow(row)
                f_out.flush()

            print(f"[worker pid={pid}] Finished Baseline b_id={b_id} ({obj_name}, {base_color}) -> 8 variants logged", flush=True)


# =========================== CONSOLIDATION & ANALYSIS ===========================
def consolidate_dataset(out_dir):
    """Merges all worker part files into a unified dataset CSV and generates summary statistics."""
    parts_dir = os.path.join(out_dir, "_csv_parts")
    master_csv = os.path.join(out_dir, "dataset_mlp_pca_25k.csv")

    rows = []
    if os.path.exists(parts_dir):
        for fn in os.listdir(parts_dir):
            if fn.endswith(".csv"):
                fp = os.path.join(parts_dir, fn)
                with open(fp) as f:
                    reader = csv.DictReader(f)
                    rows.extend(list(reader))

    if not rows:
        print("No records found to consolidate.")
        return

    # Deduplicate by sample_id
    seen = set()
    dedup_rows = []
def get_completed_baselines(out_dir, k_variants):
    """Finds all baseline_ids that already have >= k_variants completed in CSV parts or consolidated files."""
    completed = set()
    counts = {}

    csv_paths = []
    # 1. Master consolidated CSVs
    if os.path.exists(out_dir):
        for fn in os.listdir(out_dir):
            if fn.endswith(".csv"):
                csv_paths.append(os.path.join(out_dir, fn))

    # 2. Part CSVs
    parts_dir = os.path.join(out_dir, "_csv_parts")
    if os.path.exists(parts_dir):
        for fn in os.listdir(parts_dir):
            if fn.endswith(".csv"):
                csv_paths.append(os.path.join(parts_dir, fn))

    for p in csv_paths:
        try:
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "baseline_id" in r:
                        b_id = int(r["baseline_id"])
                        counts[b_id] = counts.get(b_id, 0) + 1
        except Exception as e:
            print(f"Warning reading {p}: {e}")

    for b_id, c in counts.items():
        if c >= k_variants:
            completed.add(b_id)

    return completed


# =========================== CONSOLIDATION & ANALYSIS ===========================
def consolidate_dataset(out_dir):
    """Merges all worker part files into a unified dataset CSV and generates summary statistics."""
    parts_dir = os.path.join(out_dir, "_csv_parts")
    master_csv = os.path.join(out_dir, "dataset_mlp_pca_25k.csv")

    rows = []
    if os.path.exists(parts_dir):
        for fn in sorted(os.listdir(parts_dir)):
            if fn.endswith(".csv"):
                fp = os.path.join(parts_dir, fn)
                with open(fp) as f:
                    reader = csv.DictReader(f)
                    rows.extend(list(reader))

    if not rows:
        print("No records found to consolidate.")
        return

    # Deduplicate by sample_id
    seen = set()
    dedup_rows = []
    for r in rows:
        if r["sample_id"] not in seen:
            seen.add(r["sample_id"])
            dedup_rows.append(r)

    with open(master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(dedup_rows)

    print("=" * 75)
    print(f"DATASET CONSOLIDATED: {len(dedup_rows)} total unique training samples saved to:")
    print(f"  {master_csv}")
    print("=" * 75)


# =========================== MAIN DRIVER ===========================
def main():
    parser = argparse.ArgumentParser(description="Large-Scale Dataset Collection in PCA Latent Space")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--n-baselines", type=int, default=N_BASELINES_TOTAL)
    parser.add_argument("--k-variants", type=int, default=K_VARIANTS_PER_BASELINE)
    parser.add_argument("--baseline-start", type=int, default=None)
    parser.add_argument("--baseline-end", type=int, default=None)
    parser.add_argument("--job-tag", type=str, default="job")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.consolidate_only:
        consolidate_dataset(args.out_dir)
        return

    if args.worker:
        seg_models = setup_seg_models(DEVICE)
        pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models)
        return

    # Master Controller
    gpu_list = get_gpu_list(None)
    n_workers = len(gpu_list) * args.workers_per_gpu

    # Generate full deterministic baseline list
    all_tasks = build_all_dataset_tasks(args.n_baselines, args.k_variants, MASTER_SEED)

    # 1. Apply baseline range slice if specified
    if args.baseline_start is not None or args.baseline_end is not None:
        b_start = args.baseline_start if args.baseline_start is not None else 0
        b_end = args.baseline_end if args.baseline_end is not None else args.n_baselines
        all_tasks = [t for t in all_tasks if b_start <= t["baseline_id"] < b_end]

    # 2. Automatically filter out already completed baselines
    completed_baselines = get_completed_baselines(args.out_dir, args.k_variants)
    remaining_tasks = [t for t in all_tasks if t["baseline_id"] not in completed_baselines]

    print("=" * 80)
    print(f">>> STARTING LARGE-SCALE PCA DATASET COLLECTION")
    print(f"    Total Target Baselines: {args.n_baselines} (x {args.k_variants} = {args.n_baselines * args.k_variants} samples)")
    print(f"    Already Completed:      {len(completed_baselines)} baselines ({len(completed_baselines) * args.k_variants} samples)")
    print(f"    Remaining for this Run: {len(remaining_tasks)} baselines ({len(remaining_tasks) * args.k_variants} samples)")
    print(f"    Objects Solid:          {len(OBJECT_SCENES)} diverse categories (including 16 fruits & vegetables)")
    print(f"    Workers:                {n_workers} across {len(gpu_list)} GPU(s)")
    print("=" * 80)

    if not remaining_tasks:
        print("All baselines are already completed! Consolidating dataset...")
        consolidate_dataset(args.out_dir)
        return

    cmd_base = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--out-dir", args.out_dir,
    ]
    tmp_chunks = os.path.join(args.out_dir, f"_tmp_chunks_{args.job_tag}")
    spawn_workers(remaining_tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

    consolidate_dataset(args.out_dir)


if __name__ == "__main__":
    main()


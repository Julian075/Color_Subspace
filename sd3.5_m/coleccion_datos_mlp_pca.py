"""
COLECCION_DATOS_MLP_PCA.PY -- Large-Scale Dataset Collection for SD3.5-M in PCA Latent Space.

Generates 25,000 paired training samples (3,125 baselines x 8 variants) across 100 diverse objects
and ISCC-NBS Level 2 color categories.
Applies continuous multi-axial 3D spherical direction vectors projected onto the 16D PCA basis (U1, U2, U3).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
from sd35_core import (
    build_envelope_bands, setup_sd35, latent_hw, decode_latents_4d,
    build_mask_latent, run_generation, get_gpu_list, spawn_workers,
)
from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000
from iscc_nbs import ISCC_NBS_LEVEL1, ISCC_NBS_LEVEL2, ISCC_NBS_L1_NAMES, ISCC_NBS_L2_NAMES

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
except ImportError as _e:
    raise ImportError(f"Failed to import from skimage: {_e}")

# =========================== CONFIGURATION ===========================
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16

RESOLUTION = 1024
STEPS = 28
GUIDANCE = 4.5
REF_MODE = "none"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "coleccion_datos_mlp_pca_out")
PCA_AXES_PATH = os.path.join(BASE_DIR, "fase_a_pca_out", "pca_axes.json")
WINNING_SCHEDULE_PATH = os.path.join(BASE_DIR, "fase_b_pca_out", "fase_b_winning_schedule.json")

N_BASELINES_TOTAL = 3125
K_VARIANTS_PER_BASELINE = 8
MASTER_SEED = 20260825

P_ZONE1_LINEAR = 0.80
ZONE1_RANGE = (0.05, 0.45)
ZONE2_RANGE = (0.45, 1.25)


def load_pca_basis(axes_path=PCA_AXES_PATH):
    if os.path.exists(axes_path):
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        print(f"[INFO] PCA axes not found at {axes_path}. Using placeholder canonical vectors.")
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return np.column_stack([u1, u2, u3])  # Shape (16, 3)


def load_winning_schedule(sched_path=WINNING_SCHEDULE_PATH):
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            return json.load(f)
    return {"gate_frac": 0.6, "n_partes": 1, "perfil_name": "ascendente"}


def load_bands(sched_path=WINNING_SCHEDULE_PATH):
    sched = load_winning_schedule(sched_path)
    perfil_generators = {
        "plano":        lambda n: [1.0] * n,
        "ascendente":   lambda n: list(np.linspace(1.0 / n, 1.0, n)),
        "suave":        lambda n: list(np.linspace(1.0 / n, 1.0, n) ** 0.5),
        "angosto_alto": lambda n: [1.0] if n == 1 else [0.2] * (n - 1) + [1.0],
        "descendente":  lambda n: list(np.linspace(1.0, 1.0 / n, n)),
    }
    return build_envelope_bands(
        sched["gate_frac"],
        perfil_generators[sched["perfil_name"]](sched["n_partes"]),
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

    # 17-56: Canonical & Everyday Objects
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
    ("teddy bear", "bear", ["on a wooden chair", "on a bed", "on a shelf"]),
    ("action figure", "figure", ["on a display shelf", "on a wooden desk", "on a table"]),
    ("dice", "dice", ["on a green felt table", "on a wooden board", "on a marble counter"]),
    ("yo-yo", "yo-yo", ["on a wooden table", "on a desk", "on a shelf"]),
    ("violin", "violin", ["on a velvet stand", "in a case", "on a polished table"]),
    ("trumpet", "trumpet", ["on a music stand", "on a velvet mat", "on a wooden table"]),
    ("flute", "flute", ["on a velvet cloth", "on a wooden desk", "in a music case"]),
    ("tambourine", "tambourine", ["on a wooden table", "on a stool", "on a studio bench"]),
    ("scooter", "scooter", ["on a pavement", "by a garage wall", "in a hallway"]),
    ("skateboard helmet", "helmet", ["on a ramp bench", "on a wooden shelf", "on the floor"]),
    ("canoe paddle", "paddle", ["against a wooden wall", "on a dock", "on a boat bench"]),
    ("cushion", "cushion", ["on a leather couch", "on an armchair", "on a bench"]),
    ("rug", "rug", ["on a hardwood floor", "in a studio room", "in a living room"]),
    ("curtain", "curtain", ["by a sunlit window", "in a studio setup", "in a modern room"]),
    ("table lamp", "lamp", ["on a wooden desk", "on a modern nightstand", "on a table"]),
    ("wall clock", "clock", ["on a white gallery wall", "on a clean brick wall", "in an office"]),
    ("flower pot", "pot", ["on a sunny windowsill", "on a wooden table", "on a balcony"]),
    ("picture frame", "frame", ["on a wooden mantle", "on a desk", "on a gallery wall"])
]

CSV_FIELDS = [
    "sample_id", "baseline_id", "variant_idx", "obj_name", "obj_word", "base_color_name",
    "prompt", "seed", "m1", "m2", "m3", "magnitude_total",
    "base_L", "base_a", "base_b", "mod_L", "mod_a", "mod_b",
    "delta_L", "delta_a", "delta_b", "deltaE", "ssim_in", "psnr_in", "note"
]


# =========================== SAMPLING FUNCTIONS ===========================
def sample_sphere_direction():
    """Uniform random unit vector on S^2 sphere (Marsaglia method)."""
    g = np.random.normal(0, 1, 3)
    norm = np.linalg.norm(g)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (g / norm).astype(np.float32)


def sample_calibrated_magnitude():
    """Calibrated 80/20 magnitude sampling across Zone 1 and Zone 2."""
    if random.random() < P_ZONE1_LINEAR:
        return float(random.uniform(ZONE1_RANGE[0], ZONE1_RANGE[1]))
    else:
        return float(random.uniform(ZONE2_RANGE[0], ZONE2_RANGE[1]))


def calculate_psnr_masked(img1_np, img2_np, mask_2d=None):
    if mask_2d is not None and mask_2d.sum() > 0:
        diff = (img1_np.astype(np.float32) - img2_np.astype(np.float32))[mask_2d]
        mse = float(np.mean(diff ** 2))
    else:
        mse = float(np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def build_all_tasks(n_baselines, k_variants, master_seed, pca_basis):
    random.seed(master_seed)
    np.random.seed(master_seed)

    tasks = []
    for b_idx in range(n_baselines):
        obj_name, obj_word, scenes = random.choice(OBJECT_SCENES)
        scene = random.choice(scenes)

        if random.random() < 0.30:
            color_name = random.choice(ISCC_NBS_L1_NAMES)
        else:
            color_name = random.choice(ISCC_NBS_L2_NAMES)

        prompt = f"a photo of a {color_name} {obj_name} {scene}, studio lighting, high quality, realistic"
        seed = int(random.randint(1000, 99999999))

        variant_specs = []
        for v_idx in range(k_variants):
            d_unit = sample_sphere_direction()
            mag = sample_calibrated_magnitude()
            m_vec = d_unit * mag  # [m1, m2, m3]

            # Project into 16D latent direction: v_16d = U * m_vec
            v_16d = pca_basis @ m_vec
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
def run_worker(chunk_path, out_dir, pipe, vae, seg_models, bands):
    pid = os.getpid()
    with open(chunk_path) as f:
        baseline_tasks = json.load(f)
    print(f"[worker pid={pid}] Assigned {len(baseline_tasks)} baselines ({len(baseline_tasks) * K_VARIANTS_PER_BASELINE} variants)", flush=True)

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"dataset_part_{pid}.csv")

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

            # 1. Baseline Generation
            lat_base = run_generation(pipe, prompt, seed, RESOLUTION, RESOLUTION, DEVICE, STEPS, GUIDANCE, ref_mode=REF_MODE)
            if torch.isnan(lat_base).any():
                continue

            img_base = decode_latents_4d(vae, lat_base)
            mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
            if mask_pixel is None or mask_pixel.sum() < 30:
                continue

            base_lab = measure_color_gt(img_base, mask_pixel)
            if base_lab is None:
                continue

            lh, lw = latent_hw(RESOLUTION, RESOLUTION)
            mask_latent = build_mask_latent(mask_pixel, lh, lw, DEVICE)

            # 2. Variants Generation
            for v in b_task["variants"]:
                v_idx = v["variant_idx"]
                sample_id = f"b{b_id:06d}_v{v_idx:02d}"

                lat_mod = run_generation(
                    pipe, prompt, seed, RESOLUTION, RESOLUTION, DEVICE, STEPS, GUIDANCE,
                    direction=v["direction_list"], magnitude=v["magnitude_total"], bands=bands, mask_latent=mask_latent, ref_mode=REF_MODE
                )
                if torch.isnan(lat_mod).any():
                    continue

                img_mod = decode_latents_4d(vae, lat_mod)
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

            print(f"[worker pid={pid}] Baseline b_id={b_id} ({obj_name}, {base_color}) -> 8 variants logged", flush=True)


# =========================== CONSOLIDATION ===========================
def consolidate_dataset(out_dir):
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
    dedup = []
    for r in rows:
        sid = r["sample_id"]
        if sid not in seen:
            seen.add(sid)
            dedup.append(r)

    with open(master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(dedup)

    print(f"\n[DATASET CONSOLIDATION] Successfully saved {len(dedup)} unique samples to {master_csv}")


# =========================== MAIN DRIVER ===========================
def main():
    parser = argparse.ArgumentParser(description="Phase C Dataset Collection for SD3.5-M")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-baselines", type=int, default=N_BASELINES_TOTAL)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bands = load_bands()

    if args.consolidate_only:
        consolidate_dataset(args.out_dir)
        return

    if args.worker:
        pipe, vae = setup_sd35(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        seg_models = setup_seg_models(DEVICE)
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models, bands)
        return

    # Master Controller
    pca_basis = load_pca_basis()
    tasks = build_all_tasks(args.n_baselines, K_VARIANTS_PER_BASELINE, MASTER_SEED, pca_basis)

    gpu_list = get_gpu_list(None)
    n_workers = len(gpu_list) * args.workers_per_gpu

    print(f"Generated {len(tasks)} baseline tasks ({len(tasks) * K_VARIANTS_PER_BASELINE} variants) -> distributing over {n_workers} workers across {len(gpu_list)} GPU(s)")

    cmd_base = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--out-dir", args.out_dir,
    ]
    tmp_chunks = os.path.join(args.out_dir, "_tmp_chunks_dataset")
    spawn_workers(tasks, n_workers, gpu_list, args.workers_per_gpu, tmp_chunks, cmd_base)

    consolidate_dataset(args.out_dir)


if __name__ == "__main__":
    main()

"""
COLECCION_DATOS_MLP_PCA.PY -- Large-Scale Dataset Collection for SDXL in PCA Latent Space.

Generates paired training samples (N baselines x 8 variants) across 100 diverse objects
and ISCC-NBS Level 2 color categories.
Applies continuous multi-axial 3D spherical direction vectors projected onto the 4D PCA basis (U1, U2, U3).
Calibrated magnitude sampling:
  - 80% Zone 1 (m in [0.05, 0.45]): High-fidelity linear color regime.
  - 20% Zone 2 (m in [0.45, 1.25]): Transition & saturation regime.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import csv
import json
import glob
import argparse
import random
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
from sdxl_core import (
    build_envelope_bands, setup_sdxl, latent_hw, decode_latents_4d,
    build_mask_latent, run_generation, get_gpu_list, spawn_workers,
)
from utils import setup_seg_models, get_object_mask, measure_color_gt, ciede2000
from iscc_nbs import ISCC_NBS_LEVEL1, ISCC_NBS_LEVEL2, ISCC_NBS_L1_NAMES, ISCC_NBS_L2_NAMES

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
except ImportError as _e:
    raise ImportError(f"Failed to import from skimage: {_e}")

# =========================== CONFIGURATION ===========================
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4

RESOLUTION = 1024
STEPS = 30
GUIDANCE = 5.0
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
    return np.column_stack([u1, u2, u3])  # Shape (4, 3)


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
    ("towel", "towel", ["hanging on a hook", "folded on a shelf", "on a marble counter"]),
    ("sunglasses", "sunglasses", ["on a table", "on a book", "on a wooden shelf"]),
    ("headphones", "headphones", ["on a wooden stand", "on a desk", "by a laptop"]),
    ("cushion", "cushion", ["on an armchair", "on a sofa", "on a wooden bench"]),
    ("plate", "plate", ["on a dining table", "on a kitchen counter", "on a placemat"]),
    ("bowl", "bowl", ["on a wooden tray", "on a dining table", "on a countertop"]),
    ("pot", "pot", ["on a stove", "on a kitchen island", "on a trivet"]),
    ("pan", "pan", ["on a cooktop", "hanging on a rack", "on a wooden counter"]),
    ("comb", "comb", ["on a bathroom vanity", "on a marble shelf", "on a dresser"]),
    ("mirror", "mirror", ["on a wooden dresser", "on a table", "mounted on a wall"]),
    ("blanket", "blanket", ["folded on a chair", "draped over a sofa", "at the foot of a bed"]),

    # 57-100: Vehicles, Animals, Apparel & Additional Everyday Objects
    ("car", "car", ["parked on an empty asphalt road", "in a modern showroom", "on a city street"]),
    ("truck", "truck", ["on an open highway", "in a parking lot", "on a dirt road"]),
    ("motorcycle", "motorcycle", ["parked by a scenic overlook", "in a garage", "on a street"]),
    ("bus", "bus", ["at a city transit stop", "on a wide boulevard", "on a road"]),
    ("scooter", "scooter", ["on a pedestrian walkway", "by a cafe patio", "in a park"]),
    ("van", "van", ["parked near a campsite", "on a suburban driveway", "on a road"]),
    ("tractor", "tractor", ["in a rural farm field", "by a barn", "on a dirt path"]),
    ("boat", "boat", ["docked at a calm harbor", "on a quiet lake", "by a wooden pier"]),
    ("train", "train", ["at a railway station platform", "on tracks in a valley", "on a rail line"]),
    ("airplane", "airplane", ["on an airport tarmac", "in an open hangar", "on a runway"]),
    ("helicopter", "helicopter", ["on a concrete helipad", "in an open airfield", "on a platform"]),
    ("jacket", "jacket", ["on a wooden mannequin", "hanging on a minimalist hanger", "on a chair"]),
    ("shirt", "shirt", ["neatly folded on a shelf", "hanging in a wardrobe", "on a table"]),
    ("pants", "pants", ["folded on a clean surface", "hanging on a rail", "on a display table"]),
    ("dress", "dress", ["on a boutique dressform", "hanging against a neutral wall", "on a hanger"]),
    ("shoes", "shoes", ["on a retail display riser", "on a polished floor", "on a shelf"]),
    ("boots", "boots", ["standing by a rustic doorway", "on a wooden floor", "on a mat"]),
    ("sneakers", "sneakers", ["on a clean studio plinth", "on a gym floor", "on a shelf"]),
    ("hat", "hat", ["resting on a hat stand", "on a wooden table", "on a shelf"]),
    ("gloves", "gloves", ["laid flat on a table", "on a wooden shelf", "by a coat"]),
    ("socks", "socks", ["folded in a neat pair", "on a wooden tray", "in a drawer"]),
    ("belt", "belt", ["coiled on a dresser top", "hanging on a hook", "on a shelf"]),
    ("tie", "tie", ["rolled on a presentation tray", "hanging with a suit", "on a dresser"]),
    ("sunglasses_case", "case", ["on a nightstand", "on an office desk", "in a bag"]),
    ("mug_cup", "cup", ["on a saucer on a patio table", "on a coaster", "on a counter"]),
    ("saucer", "saucer", ["under a delicate teacup", "on a tablecloth", "on a tray"]),
    ("fork", "fork", ["beside a napkin", "on a dark placemat", "on a dining table"]),
    ("spoon", "spoon", ["resting on a ceramic rest", "on a saucer", "on a table"]),
    ("knife", "knife", ["on a butcher block", "beside a cutting board", "on a table"]),
    ("cutting_board", "board", ["on a kitchen island", "standing against tile", "on a table"]),
    ("soap_dispenser", "dispenser", ["by a modern sink", "on a bathroom counter", "on a tray"]),
    ("toothbrush_holder", "holder", ["on a marble vanity", "on a shelf", "by a mirror"]),
    ("flower_pot", "pot", ["on a sunny windowsill", "on a garden patio", "on a balcony"]),
    ("watering_pot", "pot", ["beside potted ferns", "on a potting bench", "on a terrace"]),
    ("birdhouse", "birdhouse", ["mounted on a wooden post", "in a lush backyard", "in a garden"]),
    ("bench", "bench", ["along a cobblestone path", "in a botanical garden", "in a park"]),
    ("stool", "stool", ["beside a high counter", "in an artist studio", "in a kitchen"]),
    ("desk_organizer", "organizer", ["on a study desk", "holding pens and notes", "on a shelf"]),
    ("tissue_box", "box", ["on a contemporary end table", "on a coffee table", "on a desk"]),
    ("trash_bin", "bin", ["under a modern desk", "in a sleek kitchen corner", "in an office"]),
    ("laundry_basket", "basket", ["in a bright laundry room", "by a dresser", "in a bedroom"]),
    ("coasters", "coasters", ["stacked on a glass coffee table", "on a wooden tray", "on a table"]),
    ("mousepad", "pad", ["on a gaming desk", "beside a mechanical keyboard", "on a table"]),
    ("laptop_stand", "stand", ["on a minimalist workstation", "elevating a notebook", "on a desk"]),
]

CSV_FIELDS = [
    "sample_id", "baseline_id", "prompt", "obj_name", "seed",
    "m1", "m2", "m3", "magnitude", "zone",
    "base_L", "base_a", "base_b", "base_iscc_l1", "base_iscc_l2",
    "mod_L", "mod_a", "mod_b", "mod_iscc_l1", "mod_iscc_l2",
    "delta_L", "delta_a", "delta_b", "deltaE",
    "ssim_in", "psnr_in", "note"
]


# =========================== SAMPLING ENGINE ===========================
def sample_m_vector(rng: random.Random) -> tuple:
    """
    Samples continuous 3D spherical direction vector and calibrated magnitude.
    """
    theta = rng.uniform(0.0, np.pi)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    dir_v = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=np.float32)
    dir_v /= np.linalg.norm(dir_v)

    if rng.random() < P_ZONE1_LINEAR:
        mag = rng.uniform(ZONE1_RANGE[0], ZONE1_RANGE[1])
        zone = "Zone1_Linear"
    else:
        mag = rng.uniform(ZONE2_RANGE[0], ZONE2_RANGE[1])
        zone = "Zone2_Transition"

    m_vec = (dir_v * mag).astype(np.float32)
    return float(m_vec[0]), float(m_vec[1]), float(m_vec[2]), float(mag), zone


def calculate_psnr_masked(img1_np, img2_np, mask_2d=None):
    if mask_2d is not None and mask_2d.sum() > 0:
        diff = (img1_np.astype(np.float32) - img2_np.astype(np.float32))[mask_2d]
        mse = float(np.mean(diff ** 2))
    else:
        mse = float(np.mean((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2))
    if mse < 1e-10:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def build_all_dataset_tasks(n_baselines: int = N_BASELINES_TOTAL, k_variants: int = K_VARIANTS_PER_BASELINE, master_seed: int = MASTER_SEED):
    rng = random.Random(master_seed)
    tasks = []
    sample_id = 0

    for b_idx in range(n_baselines):
        obj_name, obj_word, scene_variants = rng.choice(OBJECT_SCENES)
        scene = rng.choice(scene_variants)
        seed = rng.randint(1000, 9999999)
        prompt = f"a photo of a {obj_name} {scene}, studio lighting, product photography, high quality"

        variants = []
        for v_idx in range(k_variants):
            m1, m2, m3, mag, zone = sample_m_vector(rng)
            variants.append({
                "sample_id": sample_id,
                "m1": m1, "m2": m2, "m3": m3,
                "magnitude": mag, "zone": zone
            })
            sample_id += 1

        tasks.append({
            "baseline_id": b_idx,
            "prompt": prompt,
            "obj_name": obj_name,
            "obj_word": obj_word,
            "seed": seed,
            "variants": variants,
        })
    return tasks


def run_worker(chunk_path, out_dir, pipe, vae, seg_models, pca_basis, bands):
    pid = os.getpid()
    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(chunk_path) as f:
        tasks = json.load(f)
    print(f"[worker pid={pid} GPU_phys={gpu_env}] {len(tasks)} baselines assigned")

    u1, u2, u3 = pca_basis[:, 0], pca_basis[:, 1], pca_basis[:, 2]

    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"dataset_part_{pid}.csv")

    with open(part_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for t in tasks:
            b_id, prompt, obj_word, seed = t["baseline_id"], t["prompt"], t["obj_word"], t["seed"]

            # 1. Baseline Generation
            latents_base = run_generation(
                pipe, prompt, seed, RESOLUTION, RESOLUTION, DEVICE, STEPS, GUIDANCE
            )
            if torch.isnan(latents_base).any():
                continue
            img_base = decode_latents_4d(vae, latents_base)

            mask_pixel = get_object_mask(Image.fromarray(img_base), obj_word, seg_models)
            if mask_pixel is None:
                h, w = img_base.shape[:2]
                yy, xx = np.ogrid[:h, :w]
                mask_pixel = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= (min(h, w) * 0.35) ** 2

            base_lab = measure_color_gt(img_base, mask_pixel)
            if base_lab is None:
                continue

            from iscc_nbs import find_nearest_iscc_l1, find_nearest_iscc_l2
            base_l1, _ = find_nearest_iscc_l1(base_lab)
            base_l2, _ = find_nearest_iscc_l2(base_lab)

            lh, lw = latent_hw(RESOLUTION, RESOLUTION)
            mask_latent = build_mask_latent(mask_pixel, lh, lw, DEVICE)

            # 2. Variants Generation
            for v in t["variants"]:
                m_vec = (v["m1"], v["m2"], v["m3"])
                latents_mod = run_generation(
                    pipe, prompt, seed, RESOLUTION, RESOLUTION, DEVICE, STEPS, GUIDANCE,
                    pca_basis=(u1, u2, u3), m_vector=m_vec, bands=bands, mask_latent=mask_latent
                )
                if torch.isnan(latents_mod).any():
                    continue
                img_mod = decode_latents_4d(vae, latents_mod)

                lab_mod = measure_color_gt(img_mod, mask_pixel)
                if lab_mod is None:
                    continue

                dE = ciede2000(base_lab, lab_mod)
                ssim_in = float(structural_similarity(img_base, img_mod, channel_axis=2, data_range=255))
                psnr_in = calculate_psnr_masked(img_base, img_mod, mask_pixel)

                mod_l1, _ = find_nearest_iscc_l1(lab_mod)
                mod_l2, _ = find_nearest_iscc_l2(lab_mod)

                writer.writerow({
                    "sample_id": v["sample_id"],
                    "baseline_id": b_id,
                    "prompt": prompt,
                    "obj_name": t["obj_name"],
                    "seed": seed,
                    "m1": round(v["m1"], 4),
                    "m2": round(v["m2"], 4),
                    "m3": round(v["m3"], 4),
                    "magnitude": round(v["magnitude"], 4),
                    "zone": v["zone"],
                    "base_L": round(base_lab[0], 2),
                    "base_a": round(base_lab[1], 2),
                    "base_b": round(base_lab[2], 2),
                    "base_iscc_l1": base_l1,
                    "base_iscc_l2": base_l2,
                    "mod_L": round(lab_mod[0], 2),
                    "mod_a": round(lab_mod[1], 2),
                    "mod_b": round(lab_mod[2], 2),
                    "mod_iscc_l1": mod_l1,
                    "mod_iscc_l2": mod_l2,
                    "delta_L": round(lab_mod[0] - base_lab[0], 2),
                    "delta_a": round(lab_mod[1] - base_lab[1], 2),
                    "delta_b": round(lab_mod[2] - base_lab[2], 2),
                    "deltaE": round(dE, 3),
                    "ssim_in": round(ssim_in, 4),
                    "psnr_in": round(psnr_in, 2),
                    "note": "ok"
                })
                f.flush()
    print(f"[worker pid={pid} GPU_phys={gpu_env}] Finished")


def collect_dataset(out_dir, n_baselines, parallel=False, available_gpus=None, workers_per_gpu=1):
    pca_basis = load_pca_basis()
    bands = load_bands()
    all_tasks = build_all_dataset_tasks(n_baselines=n_baselines)
    os.makedirs(out_dir, exist_ok=True)
    parts_dir = os.path.join(out_dir, "_csv_parts")
    os.makedirs(parts_dir, exist_ok=True)

    # Resume check: scan _csv_parts to find already completed baseline_ids
    completed_baselines = set()
    for pf in glob.glob(os.path.join(parts_dir, "*.csv")):
        try:
            with open(pf, newline="") as fin:
                reader = csv.DictReader(fin)
                counts = {}
                for row in reader:
                    if "baseline_id" in row and row["baseline_id"]:
                        b_id = int(row["baseline_id"])
                        counts[b_id] = counts.get(b_id, 0) + 1
                for b_id, cnt in counts.items():
                    if cnt >= K_VARIANTS_PER_BASELINE:
                        completed_baselines.add(b_id)
        except Exception:
            pass

    tasks = [t for t in all_tasks if t["baseline_id"] not in completed_baselines]
    n_done = len(completed_baselines)
    n_rem = len(tasks)
    print(f"\n>>> Phase C Dataset Collection: {n_baselines} baselines total ({n_baselines * K_VARIANTS_PER_BASELINE} samples)")
    print(f"    [RESUME] Found {n_done} completed baselines ({n_done * K_VARIANTS_PER_BASELINE} samples already saved).")
    print(f"    [RESUME] Remaining baselines to generate: {n_rem} ({n_rem * K_VARIANTS_PER_BASELINE} samples).")

    if tasks:
        if parallel:
            gpu_list = get_gpu_list(available_gpus)
            n_workers = len(gpu_list) * workers_per_gpu
            cmd_base = [sys.executable, os.path.abspath(__file__), "--worker", "--out-dir", out_dir]
            spawn_workers(tasks, n_workers, gpu_list, workers_per_gpu, os.path.join(out_dir, "_tmp_chunks_dataset"), cmd_base)
        else:
            pipe, vae = setup_sdxl(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
            seg_models = setup_seg_models(DEVICE)
            chunk_path = os.path.join(out_dir, "chunk_single.json")
            with open(chunk_path, "w") as f:
                json.dump(tasks, f)
            run_worker(chunk_path, out_dir, pipe, vae, seg_models, pca_basis, bands)

    # Merge
    out_csv = os.path.join(out_dir, f"dataset_mlp_pca_sdxl.csv")
    part_files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".csv"))
    seen_samples = set()
    rows_to_write = []
    for pf in part_files:
        try:
            with open(os.path.join(parts_dir, pf), newline="") as fin:
                for row in csv.DictReader(fin):
                    s_id = row.get("sample_id")
                    if s_id and s_id not in seen_samples:
                        seen_samples.add(s_id)
                        rows_to_write.append(row)
        except Exception as e:
            print(f"[MERGE WARNING] Could not read part {pf}: {e}")

    try:
        rows_to_write.sort(key=lambda r: int(r["sample_id"]))
    except Exception:
        pass

    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows_to_write:
            writer.writerow(r)

    print(f"\n>>> Successfully assembled complete dataset: {len(rows_to_write)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Phase C: Large-Scale Dataset Collection for SDXL in PCA Space")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-baselines", type=int, default=N_BASELINES_TOTAL)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chunk", default=None)
    args = parser.parse_args()

    if args.worker:
        pipe, vae = setup_sdxl(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
        seg_models = setup_seg_models(DEVICE)
        pca_basis = load_pca_basis()
        bands = load_bands()
        run_worker(args.chunk, args.out_dir, pipe, vae, seg_models, pca_basis, bands)
        return

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()] if args.gpus else None
    collect_dataset(args.out_dir, args.n_baselines, parallel=args.parallel, available_gpus=gpus, workers_per_gpu=args.workers_per_gpu)


if __name__ == "__main__":
    main()

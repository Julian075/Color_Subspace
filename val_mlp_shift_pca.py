"""
VAL_MLP_SHIFT_PCA.PY -- Full Closed-Loop Validation of Trained PCA MLP on FLUX.

Validates the trained MLP by generating real images with FLUX + SAM3 + PCA Shift
and measuring the actual color achieved vs target color:
  - CIEDE2000 (ΔE00)
  - Delta ab (Euclidean distance in chromatic plane)
  - Delta Chroma (|ΔC*|)
  - Hue Angle Error (|Δh°|)
  - SSIM & PSNR (Full & Masked)
  - Selected ISCC-NBS Prompt Color Name

Generates side-by-side visual panels and comprehensive CSV validation reports.
"""

import os
import sys
import csv
import json
import math
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# Ensure parent directory is in sys.path
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from flux_core import (
    build_envelope_bands, envelope_weight, compute_ref,
    latent_hw, unpack_to_4d, pack_from_4d, decode_latents_4d, build_mask_latent,
    PERFIL_GENERATORS, setup_flux, get_gpu_list, spawn_workers,
)
from utils import (
    setup_seg_models, get_object_mask, measure_color_gt, ciede2000,
    hex_to_rgb, rgb_to_lab_single_np, nearest_color_name
)
from model_pca import load_mlp_pca

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
RESOLUTION = 1024
STEPS = 28
GUIDANCE = 3.5
PROMPT_TEMPLATE = "a photo of a {color_name} {object_phrase} {scene}, studio lighting, high quality, 8k"

PCA_AXES_PATH = os.path.join(os.path.dirname(__file__), "fase_a_pca_out", "pca_axes.json")
WINNING_SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), "fase_b_pca_out", "fase_b_winning_schedule.json")
DEFAULT_CKPT_PATH = os.path.join(os.path.dirname(__file__), "mlp_training_out", "mlp_shift_pca_best.pt")

# Load PCA axes
with open(PCA_AXES_PATH) as f:
    pca_data = json.load(f)["axes"]
U1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
U2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
U3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)

# Load Winning Schedule
if os.path.exists(WINNING_SCHEDULE_PATH):
    with open(WINNING_SCHEDULE_PATH) as f:
        sched_cfg = json.load(f)
else:
    sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

BANDS = build_envelope_bands(
    sched_cfg["gate_frac"],
    PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
    "ramp_down"
)

# =========================== TEST OBJECT POOLS ===========================
OBJECT_SCENES_SEEN = [
    ("bicycle", "bicycle", "on a street"),
    ("mug", "mug", "on a wooden desk"),
    ("backpack", "backpack", "on the floor"),
    ("vase", "vase", "on a clean table"),
    ("umbrella", "umbrella", "in an entryway"),
    ("mailbox", "mailbox", "on a suburban street"),
    ("kite", "kite", "in the sky"),
    ("scarf", "scarf", "on a wooden chair"),
    ("balloon", "balloon", "floating in a room"),
    ("bottle", "bottle", "on a kitchen counter"),
    ("lamp", "lamp", "on a side table"),
    ("pillow", "pillow", "on a sofa"),
    ("apple", "apple", "on a wooden cutting board"),
    ("bell pepper", "pepper", "in a vegetable basket"),
    ("strawberry", "strawberry", "on a white ceramic saucer"),
]

OBJECT_SCENES_UNSEEN = [
    ("chair", "chair", "in a studio"),
    ("guitar", "guitar", "leaning against a wall"),
    ("camera", "camera", "on a wooden table"),
    ("hat", "hat", "on a shelf"),
    ("watch", "watch", "on a velvet pad"),
    ("shoe", "shoe", "on the floor"),
    ("cup", "cup", "on a marble countertop"),
    ("violin", "violin", "in a music case"),
    ("teapot", "teapot", "on a dining table"),
    ("headphones", "headphones", "on a desk"),
    ("blender", "blender", "on a kitchen counter"),
    ("drone", "drone", "on a launching pad"),
]

# 20 Challenging Test Colors (Diverse hues, pastels, saturated, and dark shades)
CHALLENGING_TEST_HEX = [
    ("#DC143C", "crimson"),
    ("#B94642", "brick red"),
    ("#E68697", "soft pink"),
    ("#DC7D34", "burnt orange"),
    ("#D9B451", "mustard gold"),
    ("#9ACD32", "yellow green"),
    ("#00FA9A", "spring mint"),
    ("#0000CD", "royal blue"),
    ("#F0FFF0", "honeydew"),
    ("#F5F5DC", "beige"),
    ("#5C4033", "dark brown"),
    ("#800000", "maroon"),
    ("#FF7F50", "coral"),
    ("#C71585", "magenta"),
    ("#6A0DAD", "deep purple"),
    ("#008080", "teal"),
    ("#40E0D0", "turquoise"),
    ("#87CEEB", "sky blue"),
    ("#556B2F", "olive green"),
    ("#708090", "slate gray"),
]


def calculate_psnr_masked(img_ref, img_mod, mask):
    diff = (img_ref.astype(np.float32) - img_mod.astype(np.float32)) ** 2
    mask_3d = np.repeat(mask[:, :, None], 3, axis=2)
    if mask_3d.sum() == 0:
        return 0.0
    mse = diff[mask_3d].mean()
    if mse == 0:
        return 100.0
    return float(10 * np.log10((255.0 ** 2) / mse))


# =========================== INFERENCE ROUTINE ===========================
def shift_pca_4d(latents_4d, m1, m2, m3, mask=None):
    out = latents_4d.clone()
    m_mask = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in range(16):
        delta_c = float(m1 * U1[c] + m2 * U2[c] + m3 * U3[c])
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


@torch.no_grad()
def generate_with_pca_mlp(pipe, vae, mlp_shift, seg_models, device,
                          obj_phrase, obj_word, scene, color_name, target_lab, seed,
                          height=RESOLUTION, width=RESOLUTION):
    prompt = PROMPT_TEMPLATE.format(color_name=color_name, object_phrase=obj_phrase, scene=scene)
    latent_h, latent_w = latent_hw(pipe, height, width)

    state = {
        "locked": False, "mask_latent": None, "mask_pixel": None,
        "init_lab": None, "m_pred": None, "img_x0": None, "failed": False, "reason": None
    }
    captured = {}

    orig_step = pipe.scheduler.step

    def patched_step(model_output, timestep, sample, *a, **k):
        if not state["locked"]:
            try:
                sched = pipe.scheduler
                idx = sched.index_for_timestep(timestep) if hasattr(sched, "index_for_timestep") else getattr(sched, "_step_index", None)
                sigma = sched.sigmas[idx] if idx is not None else None
                if sigma is not None:
                    captured["pending_x0"] = (sample.detach() - sigma * model_output.detach()).clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def cb(pipe_, step_index, timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        frac = step_index / max(STEPS - 1, 1)
        w = envelope_weight(frac, BANDS)

        if not state["locked"]:
            if w == 0.0:
                return callback_kwargs

            pred_x0 = captured.get("pending_x0", lat)
            img_x0 = decode_latents_4d(vae, unpack_to_4d(pipe_, pred_x0, height, width))
            state["img_x0"] = img_x0

            mask_pixel = get_object_mask(Image.fromarray(img_x0), obj_word, seg_models)
            state["locked"] = True

            if mask_pixel is None or mask_pixel.sum() < 50:
                state["failed"] = True
                state["reason"] = "Mask detection failed at gate step"
                return callback_kwargs

            init_lab = measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Color measurement failed at gate step"
                return callback_kwargs

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            # Predict PCA shift vector m = (m1, m2, m3)
            m_pred = mlp_shift.predict_m(init_lab, target_lab)
            state["m_pred"] = m_pred

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return callback_kwargs

        unpacked = unpack_to_4d(pipe_, lat, height, width)
        m = state["mask_latent"]
        if m.shape[-2:] != unpacked.shape[-2:]:
            m = F.interpolate(m, size=unpacked.shape[-2:], mode="nearest")

        m1, m2, m3 = state["m_pred"]
        shifted = shift_pca_4d(unpacked, m1 * w, m2 * w, m3 * w, mask=m)
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents = pipe(
            prompt, height=height, width=width,
            guidance_scale=GUIDANCE, num_inference_steps=STEPS,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent", callback_on_step_end=cb
        ).images
    finally:
        pipe.scheduler.step = orig_step

    if state["failed"]:
        return None, None, None, state["reason"]

    img_final = decode_latents_4d(vae, unpack_to_4d(pipe, latents, height, width))
    return img_final, state["img_x0"], state["m_pred"], state["init_lab"]


# =========================== VISUALIZATION PANEL ===========================
def create_comparison_panel(img_x0, img_final, mask_pixel, target_rgb, target_lab, init_lab, final_lab,
                            color_name, dist_to_name, prompt,
                            m_pred, dE_init, dE_final, dab_init, dab_final, dC_init, dC_final, hue_err, ssim_val, psnr_val):
    H, W = 512, 512
    im_base = Image.fromarray(img_x0).resize((W, H))
    im_final = Image.fromarray(img_final).resize((W, H))

    # Mask Overlay
    overlay = img_final.copy().astype(np.float32)
    red = np.array([255.0, 40.0, 40.0])
    if mask_pixel is not None:
        overlay[mask_pixel] = 0.45 * overlay[mask_pixel] + 0.55 * red
    im_mask = Image.fromarray(overlay.astype(np.uint8)).resize((W, H))

    # Target Color Swatch
    swatch = np.zeros((H, W, 3), dtype=np.uint8)
    swatch[:, :] = target_rgb
    im_swatch = Image.fromarray(swatch)

    # Panel Layout: 4 columns with top banner for prompt and metrics
    HEADER_H = 120
    panel = Image.new("RGB", (W * 4, H + HEADER_H), (20, 20, 20))
    panel.paste(im_base, (0, HEADER_H))
    panel.paste(im_swatch, (W, HEADER_H))
    panel.paste(im_final, (W * 2, HEADER_H))
    panel.paste(im_mask, (W * 3, HEADER_H))

    draw = ImageDraw.Draw(panel)

    # Banner: Prompt text
    draw.text((20, 12), f"PROMPT: \"{prompt}\"", fill=(255, 235, 130))

    # Column 1 (Baseline Gate Preview)
    draw.text((20, 45), "1. Unshifted Baseline (Gate x0)", fill=(210, 210, 210))
    draw.text((20, 70), f"Initial Lab: ({init_lab[0]:.1f}, {init_lab[1]:.1f}, {init_lab[2]:.1f})", fill=(180, 180, 180))
    draw.text((20, 92), f"Init ΔE: {dE_init:.1f} | Δab: {dab_init:.1f} | ΔC*: {dC_init:.1f}", fill=(240, 140, 140))

    # Column 2 (Target Color Swatch & Selected Prompt Word)
    draw.text((W + 20, 45), f"2. Target Swatch: '{color_name}'", fill=(210, 210, 210))
    draw.text((W + 20, 70), f"Target Lab: ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f})", fill=(180, 180, 180))
    draw.text((W + 20, 92), f"ISCC Name: '{color_name}' (dE={dist_to_name:.1f})", fill=(150, 220, 255))

    # Column 3 (Final Shifted Output)
    draw.text((W * 2 + 20, 45), "3. Final Shifted Output", fill=(255, 255, 255))
    draw.text((W * 2 + 20, 70), f"Achieved Lab: ({final_lab[0]:.1f}, {final_lab[1]:.1f}, {final_lab[2]:.1f})", fill=(100, 255, 100))
    draw.text((W * 2 + 20, 92), f"SSIM: {ssim_val:.3f} | PSNR: {psnr_val:.1f}dB", fill=(100, 255, 100))

    # Column 4 (SAM3 Segmentation & Full Shift Metrics)
    draw.text((W * 3 + 20, 45), "4. SAM3 Mask & Precision Metrics", fill=(210, 210, 210))
    draw.text((W * 3 + 20, 70), f"ΔE: {dE_init:.1f} → {dE_final:.1f} | Δab: {dab_init:.1f} → {dab_final:.1f}", fill=(255, 220, 80))
    draw.text((W * 3 + 20, 92), f"ΔC*: {dC_init:.1f} → {dC_final:.1f} | Δh: {hue_err:.1f}° | ||m||={np.linalg.norm(m_pred):.2f}", fill=(255, 220, 80))

    return panel


# =========================== MAIN VALIDATION RUNNER ===========================
def main():
    parser = argparse.ArgumentParser(description="Validate PCA MLP on FLUX")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--out-dir", type=str, default="./val_mlp_pca_out")
    parser.add_argument("--n-cases", type=int, default=20, help="Number of test cases to validate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print(">>> STARTING FLUX PCA MLP VALIDATION PIPELINE")
    print(f"    Checkpoint: {args.ckpt}")
    print(f"    Test Cases: {args.n_cases}")
    print(f"    Output Dir: {args.out_dir}")
    print("=" * 80)

    # 1. Load models
    pipe, vae = setup_flux(MODEL_ID, DEVICE, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = setup_seg_models(DEVICE)
    mlp_shift = load_mlp_pca(args.ckpt, DEVICE)

    records = []
    cases_ran = 0

    for i in range(args.n_cases):
        # Pick seen vs unseen pool
        is_seen = bool(rng.random() < 0.5)
        pool = OBJECT_SCENES_SEEN if is_seen else OBJECT_SCENES_UNSEEN
        obj_phrase, obj_word, scene = pool[rng.integers(len(pool))]

        target_hex, default_color_name = CHALLENGING_TEST_HEX[i % len(CHALLENGING_TEST_HEX)]
        target_rgb = hex_to_rgb(target_hex)
        target_lab = rgb_to_lab_single_np(target_rgb)
        seed = int(rng.integers(10000, 9999999))

        # Determine nearest ISCC-NBS color name used for the prompt
        color_name_selected, dist_to_name = nearest_color_name(target_lab)
        prompt_text = PROMPT_TEMPLATE.format(color_name=color_name_selected, object_phrase=obj_phrase, scene=scene)

        print(f"\n[{i+1}/{args.n_cases}] Validating '{obj_phrase}' with target {color_name_selected} ({target_hex})...")
        print(f"  Prompt: \"{prompt_text}\"")

        img_final, img_x0, m_pred, init_lab = generate_with_pca_mlp(
            pipe, vae, mlp_shift, seg_models, DEVICE,
            obj_phrase, obj_word, scene, color_name_selected, target_lab, seed
        )

        if img_final is None:
            print(f"  FAILED: Mask/Generation error, skipping case.")
            continue

        # Independent Ground-Truth Measurement on Final Image
        mask_final = get_object_mask(Image.fromarray(img_final), obj_word, seg_models)
        if mask_final is None:
            mask_final = get_object_mask(Image.fromarray(img_x0), obj_word, seg_models)

        final_lab = measure_color_gt(img_final, mask_final) if mask_final is not None else None
        if final_lab is None:
            print("  Could not measure final color, skipping.")
            continue

        # Metrics
        dE_init = float(ciede2000(init_lab, target_lab))
        dE_final = float(ciede2000(final_lab, target_lab))
        dE_reduction = dE_init - dE_final

        # Delta ab (Euclidean distance in chromatic plane a, b)
        dab_init = float(math.hypot(init_lab[1] - target_lab[1], init_lab[2] - target_lab[2]))
        dab_final = float(math.hypot(final_lab[1] - target_lab[1], final_lab[2] - target_lab[2]))
        dab_reduction = dab_init - dab_final

        # Chroma metric (|C* - C*_target|)
        c_init = math.hypot(init_lab[1], init_lab[2])
        c_target = math.hypot(target_lab[1], target_lab[2])
        c_final = math.hypot(final_lab[1], final_lab[2])
        dC_init = float(abs(c_init - c_target))
        dC_final = float(abs(c_final - c_target))
        dC_reduction = dC_init - dC_final

        # Hue angle metric
        h_target_deg = math.degrees(math.atan2(target_lab[2], target_lab[1])) % 360
        h_final_deg = math.degrees(math.atan2(final_lab[2], final_lab[1])) % 360
        h_diff = abs(h_final_deg - h_target_deg) % 360
        hue_error_deg = float(min(h_diff, 360 - h_diff))

        ssim_val = float(structural_similarity(img_x0, img_final, channel_axis=2, data_range=255))
        psnr_val = float(peak_signal_noise_ratio(img_x0, img_final, data_range=255))
        psnr_masked = calculate_psnr_masked(img_x0, img_final, mask_final)
        m_mag = float(np.linalg.norm(m_pred))

        print(f"  Result: ΔE: {dE_init:.1f} → {dE_final:.1f} | Δab: {dab_init:.1f} → {dab_final:.1f} | ΔC*: {dC_init:.1f} → {dC_final:.1f} | Δh: {hue_error_deg:.1f}° | SSIM: {ssim_val:.3f} | PSNR: {psnr_val:.1f}dB | ||m||={m_mag:.2f}")

        # Save Visual Panel
        panel = create_comparison_panel(
            img_x0, img_final, mask_final, target_rgb, target_lab, init_lab, final_lab,
            color_name_selected, dist_to_name, prompt_text,
            m_pred, dE_init, dE_final, dab_init, dab_final, dC_init, dC_final, hue_error_deg, ssim_val, psnr_val
        )
        panel_path = os.path.join(args.out_dir, f"val_case_{i:03d}_{obj_word}_{color_name_selected}.png")
        panel.save(panel_path)

        records.append({
            "case_id": i,
            "object": obj_phrase,
            "obj_word": obj_word,
            "pool": "seen" if is_seen else "unseen",
            "prompt": prompt_text,
            "color_name_selected": color_name_selected,
            "dist_to_color_name": f"{dist_to_name:.2f}",
            "target_hex": target_hex,
            "target_L": f"{target_lab[0]:.2f}",
            "target_a": f"{target_lab[1]:.2f}",
            "target_b": f"{target_lab[2]:.2f}",
            "init_L": f"{init_lab[0]:.2f}",
            "init_a": f"{init_lab[1]:.2f}",
            "init_b": f"{init_lab[2]:.2f}",
            "final_L": f"{final_lab[0]:.2f}",
            "final_a": f"{final_lab[1]:.2f}",
            "final_b": f"{final_lab[2]:.2f}",
            "seed": seed,
            "m1": f"{m_pred[0]:.4f}",
            "m2": f"{m_pred[1]:.4f}",
            "m3": f"{m_pred[2]:.4f}",
            "m_norm": f"{m_mag:.4f}",
            "dE_initial": f"{dE_init:.2f}",
            "dE_final": f"{dE_final:.2f}",
            "dE_reduction": f"{dE_reduction:.2f}",
            "delta_ab_initial": f"{dab_init:.2f}",
            "delta_ab_final": f"{dab_final:.2f}",
            "delta_ab_reduction": f"{dab_reduction:.2f}",
            "chroma_err_initial": f"{dC_init:.2f}",
            "chroma_err_final": f"{dC_final:.2f}",
            "chroma_err_reduction": f"{dC_reduction:.2f}",
            "hue_error_deg": f"{hue_error_deg:.2f}",
            "ssim": f"{ssim_val:.4f}",
            "psnr_full": f"{psnr_val:.2f}",
            "psnr_masked": f"{psnr_masked:.2f}",
            "panel_path": panel_path,
        })
        cases_ran += 1

    # Save CSV Report
    csv_path = os.path.join(args.out_dir, "validation_results.csv")
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    # Summary Statistics
    if records:
        dE_finals = [float(r["dE_final"]) for r in records]
        dE_inits = [float(r["dE_initial"]) for r in records]
        dabs_finals = [float(r["delta_ab_final"]) for r in records]
        dabs_inits = [float(r["delta_ab_initial"]) for r in records]
        dCs_finals = [float(r["chroma_err_final"]) for r in records]
        dCs_inits = [float(r["chroma_err_initial"]) for r in records]
        ssims = [float(r["ssim"]) for r in records]
        psnrs = [float(r["psnr_full"]) for r in records]
        hues = [float(r["hue_error_deg"]) for r in records]

        print("\n" + "=" * 80)
        print(">>> VALIDATION SUMMARY STATISTICS:")
        print("=" * 80)
        print(f"Total Validated Cases:     {cases_ran}")
        print(f"Initial ΔE (unassisted):   {np.mean(dE_inits):.2f} ± {np.std(dE_inits):.2f}")
        print(f"Final ΔE (with PCA MLP):   {np.mean(dE_finals):.2f} ± {np.std(dE_finals):.2f}")
        print(f"Average ΔE Reduction:      {np.mean(dE_inits) - np.mean(dE_finals):+.2f}")
        print(f"Initial Δab (chromatic):   {np.mean(dabs_inits):.2f} ± {np.std(dabs_inits):.2f}")
        print(f"Final Δab (chromatic):     {np.mean(dabs_finals):.2f} ± {np.std(dabs_finals):.2f} ({np.mean(dabs_inits) - np.mean(dabs_finals):+.2f})")
        print(f"Initial Chroma ΔC*:        {np.mean(dCs_inits):.2f} ± {np.std(dCs_inits):.2f}")
        print(f"Final Chroma ΔC*:          {np.mean(dCs_finals):.2f} ± {np.std(dCs_finals):.2f} ({np.mean(dCs_inits) - np.mean(dCs_finals):+.2f})")
        print(f"Mean Hue Error:            {np.mean(hues):.2f}°")
        print(f"Mean SSIM:                 {np.mean(ssims):.4f}")
        print(f"Mean PSNR:                 {np.mean(psnrs):.2f} dB")
        print(f"Full Report Saved to:      {csv_path}")
        print(f"Visual Panels Saved to:    {args.out_dir}/")
        print("=" * 80)


if __name__ == "__main__":
    main()

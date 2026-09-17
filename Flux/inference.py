"""
INFERENCE_APPLICATION.PY -- Targeted Object Color Steering on FLUX.1-dev.

Applies the paper's downstream application method:
1. Takes a text prompt describing a scene/object and a target color (Hex, RGB, or CIELAB).
2. Uses SAM3 to segment the target object at the gate step during denoising.
3. Measures the object's initial color in CIELAB space.
4. Uses the trained MLPShiftPCA to predict the optimal latent shift (m1, m2, m3).
5. Injects the spatial latent perturbation via the calibrated temporal schedule.
6. Decodes the final image where the targeted object achieves the exact desired color.
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Add current Flux directory to path
FLUX_DIR = Path(__file__).resolve().parent
if str(FLUX_DIR) not in sys.path:
    sys.path.insert(0, str(FLUX_DIR))

from flux_core import (
    build_envelope_bands,
    envelope_weight,
    latent_hw,
    unpack_to_4d,
    pack_from_4d,
    decode_latents_4d,
    build_mask_latent,
    PERFIL_GENERATORS,
    setup_flux,
)
import utils
from model_pca import load_mlp_pca

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
NUM_LATENT_CHANNELS = 16
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5

PCA_AXES_PATH = FLUX_DIR / "fase_a_pca_out" / "pca_axes.json"
WINNING_SCHEDULE_PATH = FLUX_DIR / "fase_b_pca_out" / "fase_b_winning_schedule.json"
DEFAULT_CKPT_PATH = FLUX_DIR / "mlp_training_out" / "mlp_shift_pca_best.pt"


extract_hex_from_text = utils.extract_hex_from_text
extract_color_spec_from_text = utils.extract_color_spec_from_text
extract_object_from_text = utils.extract_object_from_text


def shift_pca_4d(latents_4d: torch.Tensor, m1: float, m2: float, m3: float, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = latents_4d.clone()
    m_mask = mask[:, 0].to(out.dtype) if mask is not None else None
    for c in range(16):
        delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c])
        if m_mask is not None:
            out[:, c] += delta_c * m_mask
        else:
            out[:, c] += delta_c
    return out


def run_color_steering_inference(
    prompt: str,
    target_color_spec: Any = "#A52A2A",
    object_word: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "outs_paper",
    save_baseline: bool = True,
    save_comparison: bool = True,
    mlp_ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Parse Target Color & Object using utils
    target_lab = utils.parse_target_color(target_color_spec)
    target_rgb = utils.lab_to_rgb_single_np(target_lab)
    target_hex = f"#{target_rgb[0]:02X}{target_rgb[1]:02X}{target_rgb[2]:02X}"
    nearest_name, _ = utils.nearest_color_name(target_lab)

    if object_word is None or object_word.strip() == "":
        object_word = utils.extract_object_from_text(prompt)

    clean_prompt = utils.clean_prompt_for_diffusion(prompt, nearest_name)

    print("=" * 80)
    print("RUNNING FLUX DOWNSTREAM OBJECT COLOR STEERING (APPLICATION METHOD)")
    print(f"Raw Prompt:       \"{prompt}\"")
    print(f"Target Object:    \"{object_word}\"")
    print(f"Target Color:     {target_hex} (RGB: {target_rgb}) -> CIELAB: L*={target_lab[0]:.1f}, a*={target_lab[1]:.1f}, b*={target_lab[2]:.1f} (nearest: '{nearest_name}')")
    print(f"Semantic Prompt:  \"{clean_prompt}\"")
    print(f"Seed: {seed} | Steps: {steps} | Guidance: {guidance} | Res: {resolution}x{resolution} | Device: {device}")
    print("=" * 80)

    # 2. Load Models
    pipe, vae = setup_flux(MODEL_ID, device, DTYPE, NUM_LATENT_CHANNELS)
    seg_models = utils.setup_seg_models(device)

    ckpt_file = mlp_ckpt_path or str(DEFAULT_CKPT_PATH)
    mlp_shift = load_mlp_pca(ckpt_file, device)

    # Load PCA axes
    with open(PCA_AXES_PATH) as f:
        pca_data = json.load(f)["axes"]
    u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
    u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
    u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)

    # Load winning schedule
    if os.path.exists(WINNING_SCHEDULE_PATH):
        with open(WINNING_SCHEDULE_PATH) as f:
            sched_cfg = json.load(f)
    else:
        sched_cfg = {"gate_frac": 0.5, "n_partes": 1, "perfil_name": "ascendente"}

    bands = build_envelope_bands(
        sched_cfg["gate_frac"],
        PERFIL_GENERATORS[sched_cfg["perfil_name"]](sched_cfg["n_partes"]),
        "ramp_down"
    )

    latent_h, latent_w = latent_hw(pipe, resolution, resolution)

    # 3. Baseline generation (if requested)
    baseline_img = None
    if save_baseline or save_comparison:
        print("\n[Step 1/2] Generating unperturbed baseline image...")
        latents_base = pipe(
            clean_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
        ).images
        base_np = decode_latents_4d(vae, unpack_to_4d(pipe, latents_base, resolution, resolution))
        baseline_img = Image.fromarray(base_np)
        base_path = out_path / f"baseline_{object_word}_seed_{seed}.png"
        baseline_img.save(base_path)
        print(f"  Saved baseline: {base_path.name}")

    # 4. Steered generation (Phase B closed-loop object color intervention)
    print(f"\n[Step 2/2] Generating color-steered image with target {target_hex}...")
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
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)

        if not state["locked"]:
            if w == 0.0:
                return callback_kwargs

            pred_x0 = captured.get("pending_x0", lat)
            img_x0 = decode_latents_4d(vae, unpack_to_4d(pipe_, pred_x0, resolution, resolution))
            state["img_x0"] = img_x0

            mask_pixel = utils.get_object_mask(Image.fromarray(img_x0), object_word, seg_models)
            state["locked"] = True

            if mask_pixel is None or mask_pixel.sum() < 50:
                print(f"  [Warning] Mask detection for '{object_word}' yielded low pixel count ({mask_pixel.sum() if mask_pixel is not None else 0}). Proceeding without mask.")
                state["failed"] = True
                state["reason"] = "Mask detection failed"
                return callback_kwargs

            init_lab = utils.measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Initial color measurement failed"
                return callback_kwargs

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            m_pred = mlp_shift.predict_m(init_lab, target_lab)
            state["m_pred"] = m_pred
            print(f"  [Gate Triggered @ Step {step_index}] Initial Lab: L*={init_lab[0]:.1f}, a*={init_lab[1]:.1f}, b*={init_lab[2]:.1f}")
            print(f"  [MLP Predicted Shift]: m1={m_pred[0]:+.4f}, m2={m_pred[1]:+.4f}, m3={m_pred[2]:+.4f}")

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return callback_kwargs

        unpacked = unpack_to_4d(pipe_, lat, resolution, resolution)
        m = state["mask_latent"]
        if m.shape[-2:] != unpacked.shape[-2:]:
            m = F.interpolate(m, size=unpacked.shape[-2:], mode="nearest")

        m1, m2, m3 = state["m_pred"]
        shifted = shift_pca_4d(unpacked, m1 * w, m2 * w, m3 * w, u1, u2, u3, mask=m)
        callback_kwargs["latents"] = pack_from_4d(pipe_, shifted, latent_h, latent_w)
        return callback_kwargs

    pipe.scheduler.step = patched_step
    try:
        latents_steered = pipe(
            clean_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
            callback_on_step_end=cb,
        ).images
    finally:
        pipe.scheduler.step = orig_step

    steered_np = decode_latents_4d(vae, unpack_to_4d(pipe, latents_steered, resolution, resolution))
    steered_img = Image.fromarray(steered_np)

    hex_clean = target_hex.replace("#", "")
    steered_path = out_path / f"steered_{object_word}_{hex_clean}_seed_{seed}.png"
    steered_img.save(steered_path)
    print(f"\n  Saved steered image to: {steered_path}")

    # 5. Measure achieved color & Delta E
    achieved_lab = None
    delta_e = None
    if state["mask_pixel"] is not None:
        final_mask = utils.get_object_mask(steered_img, object_word, seg_models)
        eval_mask = final_mask if (final_mask is not None and final_mask.sum() >= 50) else state["mask_pixel"]
        achieved_lab = utils.measure_color_gt(steered_np, eval_mask)
        if achieved_lab is not None:
            delta_e = utils.ciede2000(target_lab, achieved_lab)
            print(f"  Achieved Color:  L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}")
            print(f"  CIEDE2000 ΔE00:  {delta_e:.2f} (Perceptual precision)")

    # 6. Save Comparison Panel if requested
    if save_comparison and baseline_img is not None:
        comp_path = out_path / f"comparison_{object_word}_{hex_clean}_seed_{seed}.png"
        create_comparison_figure(
            baseline_img=baseline_img,
            steered_img=steered_img,
            mask_np=state["mask_pixel"],
            target_rgb=target_rgb,
            target_hex=target_hex,
            target_lab=target_lab,
            init_lab=state["init_lab"],
            achieved_lab=achieved_lab,
            delta_e=delta_e,
            object_word=object_word,
            prompt=prompt,
            save_path=comp_path,
        )
        print(f"  Saved comparison panel to: {comp_path}")

    print("\n" + "=" * 80)
    print("FLUX Color Steering Application Completed Successfully!")
    return {
        "steered_img": steered_img,
        "baseline_img": baseline_img,
        "steered_path": str(steered_path),
        "target_lab": target_lab,
        "achieved_lab": achieved_lab,
        "delta_e00": delta_e,
    }


def create_comparison_figure(
    baseline_img: Image.Image,
    steered_img: Image.Image,
    mask_np: Optional[np.ndarray],
    target_rgb: Tuple[int, int, int],
    target_hex: str,
    target_lab: Tuple[float, float, float],
    init_lab: Optional[Tuple[float, float, float]],
    achieved_lab: Optional[Tuple[float, float, float]],
    delta_e: Optional[float],
    object_word: str,
    prompt: str,
    save_path: Path,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # Panel 1: Baseline
    axes[0].imshow(baseline_img)
    init_str = f"L*={init_lab[0]:.1f}, a*={init_lab[1]:.1f}, b*={init_lab[2]:.1f}" if init_lab else "N/A"
    axes[0].set_title(f"Baseline FLUX (Unsteered)\nMeasured {object_word}: {init_str}", fontsize=13, weight="bold")
    axes[0].axis("off")

    # Panel 2: Steered
    axes[1].imshow(steered_img)
    ach_str = f"L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}" if achieved_lab else "N/A"
    de_str = f" | ΔE00={delta_e:.2f}" if delta_e is not None else ""
    axes[1].set_title(f"Subspace Steered ({target_hex})\nAchieved {object_word}: {ach_str}{de_str}", fontsize=13, weight="bold", color="darkred")
    axes[1].axis("off")

    # Panel 3: Mask + Target Swatch
    swatch_h, swatch_w = 1024, 1024
    info_canvas = np.zeros((swatch_h, swatch_w, 3), dtype=np.uint8)
    # Top half: target color swatch
    info_canvas[:512, :] = target_rgb
    # Bottom half: mask overlay on base
    if mask_np is not None:
        base_arr = np.array(baseline_img.resize((swatch_w, 512)))
        mask_resized = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize((swatch_w, 512))) > 128
        overlay = base_arr.copy()
        overlay[mask_resized] = [0, 220, 0] # highlight mask in bright green
        blended = (base_arr * 0.4 + overlay * 0.6).astype(np.uint8)
        info_canvas[512:, :] = blended
    else:
        info_canvas[512:, :] = 40

    axes[2].imshow(info_canvas)
    axes[2].set_title(f"Target Swatch {target_hex} (Top)\nSAM3 Segmented Object Mask (Bottom)", fontsize=13, weight="bold")
    axes[2].axis("off")

    fig.suptitle(f"FLUX.1-dev Downstream Application: Target Object Color Steering\nPrompt: \"{prompt}\"", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FLUX Downstream Object Color Steering Application")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a dog at color #A52A2A doing skateboarding in the park",
                        help="Text prompt")
    parser.add_argument("--target-color", "--hex", type=str, default=None,
                        help="Target color specification (Hex, RGB, CIELAB, or name). Auto-detected from prompt if omitted.")
    parser.add_argument("--object", type=str, default=None,
                        help="Target object to segment (e.g. dog). Auto-detected if omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Image resolution")
    parser.add_argument("--out-dir", type=str, default="./inference_outputs",
                        help="Output directory")
    parser.add_argument("--no-baseline", action="store_true", help="Do not save unperturbed baseline image")
    parser.add_argument("--no-comparison", action="store_true", help="Do not save comparison figure")

    args = parser.parse_args()

    # Discover target color from prompt if not explicitly passed
    color_in_prompt = extract_color_spec_from_text(args.prompt)
    target_spec = args.target_color or color_in_prompt or "#A52A2A"

    run_color_steering_inference(
        prompt=args.prompt,
        target_color_spec=target_spec,
        object_word=args.object,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        guidance=args.guidance,
        resolution=args.resolution,
        out_dir=args.out_dir,
        save_baseline=not args.no_baseline,
        save_comparison=not args.no_comparison,
    )


if __name__ == "__main__":
    main()

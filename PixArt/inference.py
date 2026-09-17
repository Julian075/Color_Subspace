"""
INFERENCE.PY -- Targeted Object Color Steering on PixArt-Alpha and PixArt-Sigma.

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
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image

PIXART_DIR = Path(__file__).resolve().parent
if str(PIXART_DIR) not in sys.path:
    sys.path.insert(0, str(PIXART_DIR))

from pixart_core import (
    build_envelope_bands,
    envelope_weight,
    latent_hw,
    decode_latents_4d,
    build_mask_latent,
    PERFIL_GENERATORS,
    setup_pixart,
    MAX_SEQ_LEN_ALPHA,
    MAX_SEQ_LEN_SIGMA,
)
import utils
from model_pca import load_mlp_pca

DTYPE = torch.float16
NUM_LATENT_CHANNELS = 4
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.5


extract_hex_from_text = utils.extract_hex_from_text
extract_color_spec_from_text = utils.extract_color_spec_from_text
extract_object_from_text = utils.extract_object_from_text


def load_pca_basis(variant: str):
    sub = f"fase_a_{variant}_out"
    axes_path = PIXART_DIR / sub / "pca_axes.json"
    if axes_path.exists():
        with open(axes_path) as f:
            pca_data = json.load(f)["axes"]
        u1 = np.array(pca_data["PC1"]["loadings"], dtype=np.float32)
        u2 = np.array(pca_data["PC2"]["loadings"], dtype=np.float32)
        u3 = np.array(pca_data["PC3"]["loadings"], dtype=np.float32)
    else:
        u1 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u1[0] = 1.0
        u2 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u2[1] = 1.0
        u3 = np.zeros(NUM_LATENT_CHANNELS, dtype=np.float32); u3[2] = 1.0
    return u1, u2, u3


def load_winning_schedule(variant: str):
    sub = f"fase_b_{variant}_out"
    sched_path = PIXART_DIR / sub / "fase_b_winning_schedule.json"
    if sched_path.exists():
        with open(sched_path) as f:
            return json.load(f)
    if variant == "sigma":
        return {"gate_frac": 0.75, "n_partes": 3, "perfil_name": "triangular"}
    return {"gate_frac": 0.50, "n_partes": 3, "perfil_name": "triangular"}


def run_color_steering_inference(
    prompt: str,
    target_color_spec: Any = "#A52A2A",
    object_word: Optional[str] = None,
    variant: str = "sigma",
    seed: int = 42,
    device: str = "cuda:0",
    steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    resolution: int = DEFAULT_RESOLUTION,
    out_dir: str = "./inference_outputs",
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
    print(f"RUNNING PIXART-{variant.upper()} DOWNSTREAM OBJECT COLOR STEERING")
    print(f"Raw Prompt:       \"{prompt}\"")
    print(f"Target Object:    \"{object_word}\"")
    print(f"Target Color:     {target_hex} (RGB: {target_rgb}) -> CIELAB: L*={target_lab[0]:.1f}, a*={target_lab[1]:.1f}, b*={target_lab[2]:.1f} (nearest: '{nearest_name}')")
    print(f"Semantic Prompt:  \"{clean_prompt}\"")
    print(f"Seed: {seed} | Steps: {steps} | Guidance: {guidance} | Res: {resolution}x{resolution} | Device: {device}")
    print("=" * 80)

    pipe, vae = setup_pixart(model_type=variant, device=device, dtype=DTYPE)
    seg_models = utils.setup_seg_models(device)

    default_ckpt = PIXART_DIR / f"mlp_training_{variant}_out" / "mlp_shift_pca_best.pt"
    ckpt_file = mlp_ckpt_path or str(default_ckpt)
    mlp_shift = load_mlp_pca(ckpt_file, device)

    u1, u2, u3 = load_pca_basis(variant)
    sched_cfg = load_winning_schedule(variant)

    bands = build_envelope_bands(
        sched_cfg.get("gate_frac", 0.75 if variant == "sigma" else 0.5),
        PERFIL_GENERATORS[sched_cfg.get("perfil_name", "triangular")](sched_cfg.get("n_partes", 3)),
        "ramp_down"
    )

    latent_h, latent_w = latent_hw(resolution, resolution)
    max_seq_len = MAX_SEQ_LEN_SIGMA if variant == "sigma" else MAX_SEQ_LEN_ALPHA

    baseline_img = None
    if save_baseline or save_comparison:
        print("\n[Step 1/2] Generating unperturbed baseline image...")
        latents_base = pipe(
            clean_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=max_seq_len,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
        ).images
        base_np = decode_latents_4d(vae, latents_base)
        baseline_img = Image.fromarray(base_np)
        base_path = out_path / f"baseline_{variant}_{object_word}_seed_{seed}.png"
        baseline_img.save(base_path)
        print(f"  Saved baseline: {base_path.name}")

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
                if hasattr(sched, "alphas_cumprod"):
                    t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
                    alpha_prod_t = sched.alphas_cumprod[t].to(sample.device)
                    sqrt_alpha = alpha_prod_t ** 0.5
                    sqrt_one_minus_alpha = (1.0 - alpha_prod_t) ** 0.5
                    captured["pending_x0"] = ((sample.detach() - sqrt_one_minus_alpha * model_output.detach()) / sqrt_alpha).clone()
            except Exception:
                captured.pop("pending_x0", None)
        return orig_step(model_output, timestep, sample, *a, **k)

    def callback_fn(step_index, timestep, latents):
        frac = step_index / max(steps - 1, 1)
        w = envelope_weight(frac, bands)

        if not state["locked"]:
            if w == 0.0:
                return

            pred_x0 = captured.get("pending_x0", latents)
            img_x0 = decode_latents_4d(vae, pred_x0)
            state["img_x0"] = img_x0

            mask_pixel = utils.get_object_mask(Image.fromarray(img_x0), object_word, seg_models)
            state["locked"] = True

            if mask_pixel is None or mask_pixel.sum() < 50:
                print(f"  [Warning] Mask detection for '{object_word}' yielded low pixel count. Proceeding without steering.")
                state["failed"] = True
                state["reason"] = "Mask detection failed"
                return

            init_lab = utils.measure_color_gt(img_x0, mask_pixel)
            if init_lab is None:
                state["failed"] = True
                state["reason"] = "Initial color measurement failed"
                return

            state["mask_pixel"] = mask_pixel
            state["init_lab"] = init_lab
            state["mask_latent"] = build_mask_latent(mask_pixel, latent_h, latent_w, device)

            m_pred = mlp_shift(init_lab, target_lab)
            state["m_pred"] = m_pred
            print(f"  [Gate Triggered @ Step {step_index}] Initial Lab: L*={init_lab[0]:.1f}, a*={init_lab[1]:.1f}, b*={init_lab[2]:.1f}")
            print(f"  [MLP Predicted Shift]: m1={m_pred[0]:+.4f}, m2={m_pred[1]:+.4f}, m3={m_pred[2]:+.4f}")

        if state["failed"] or w == 0.0 or state["m_pred"] is None:
            return

        m1, m2, m3 = state["m_pred"]
        m_mask = state["mask_latent"][:, 0].to(latents.dtype)
        shifted = latents.clone()
        for c in range(NUM_LATENT_CHANNELS):
            delta_c = float(m1 * u1[c] + m2 * u2[c] + m3 * u3[c]) * w
            shifted[:, c] += delta_c * m_mask
        latents.copy_(shifted)

    pipe.scheduler.step = patched_step
    try:
        latents_steered = pipe(
            clean_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=max_seq_len,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
            callback=callback_fn,
            callback_steps=1,
        ).images
    finally:
        pipe.scheduler.step = orig_step

    steered_np = decode_latents_4d(vae, latents_steered)
    steered_img = Image.fromarray(steered_np)

    hex_clean = target_hex.replace("#", "")
    steered_path = out_path / f"steered_{variant}_{object_word}_{hex_clean}_seed_{seed}.png"
    steered_img.save(steered_path)
    print(f"\n  Saved steered image to: {steered_path}")

    achieved_lab = None
    delta_e = None
    if state["mask_pixel"] is not None:
        final_mask = utils.get_object_mask(steered_img, object_word, seg_models)
        eval_mask = final_mask if (final_mask is not None and final_mask.sum() >= 50) else state["mask_pixel"]
        achieved_lab = utils.measure_color_gt(steered_np, eval_mask)
        if achieved_lab is not None:
            delta_e = utils.ciede2000(target_lab, achieved_lab)
            print(f"  Achieved Color:  L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}")
            print(f"  CIEDE2000 ΔE00:  {delta_e:.2f}")

    if save_comparison and baseline_img is not None:
        comp_path = out_path / f"comparison_{variant}_{object_word}_{hex_clean}_seed_{seed}.png"
        create_comparison_figure(
            baseline_img=baseline_img,
            steered_img=steered_img,
            mask_np=state["mask_pixel"],
            target_rgb=target_rgb,
            target_hex=target_hex,
            init_lab=state["init_lab"],
            achieved_lab=achieved_lab,
            delta_e=delta_e,
            object_word=object_word,
            prompt=prompt,
            save_path=comp_path,
        )
        print(f"  Saved comparison panel to: {comp_path}")

    print("\n" + "=" * 80)
    print(f"PixArt-{variant.upper()} Color Steering Inference Completed Successfully!")
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
    init_lab: Optional[Tuple[float, float, float]],
    achieved_lab: Optional[Tuple[float, float, float]],
    delta_e: Optional[float],
    object_word: str,
    prompt: str,
    save_path: Path,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    axes[0].imshow(baseline_img)
    init_str = f"L*={init_lab[0]:.1f}, a*={init_lab[1]:.1f}, b*={init_lab[2]:.1f}" if init_lab else "N/A"
    axes[0].set_title(f"Baseline PixArt (Unsteered)\nMeasured {object_word}: {init_str}", fontsize=13, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(steered_img)
    ach_str = f"L*={achieved_lab[0]:.1f}, a*={achieved_lab[1]:.1f}, b*={achieved_lab[2]:.1f}" if achieved_lab else "N/A"
    de_str = f" | ΔE00={delta_e:.2f}" if delta_e is not None else ""
    axes[1].set_title(f"Subspace Steered ({target_hex})\nAchieved {object_word}: {ach_str}{de_str}", fontsize=13, weight="bold", color="darkred")
    axes[1].axis("off")

    swatch_h, swatch_w = 1024, 1024
    info_canvas = np.zeros((swatch_h, swatch_w, 3), dtype=np.uint8)
    info_canvas[:512, :] = target_rgb
    if mask_np is not None:
        base_arr = np.array(baseline_img.resize((swatch_w, 512)))
        mask_resized = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize((swatch_w, 512))) > 128
        overlay = base_arr.copy()
        overlay[mask_resized] = [0, 220, 0]
        blended = (base_arr * 0.4 + overlay * 0.6).astype(np.uint8)
        info_canvas[512:, :] = blended
    else:
        info_canvas[512:, :] = 40

    axes[2].imshow(info_canvas)
    axes[2].set_title(f"Target Swatch {target_hex} (Top)\nSAM3 Segmented Object Mask (Bottom)", fontsize=13, weight="bold")
    axes[2].axis("off")

    fig.suptitle(f"PixArt Color Subspace Steering\nPrompt: \"{prompt}\"", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PixArt Downstream Object Color Steering Application")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a ceramic mug on a wooden desk",
                        help="Text prompt")
    parser.add_argument("--target-color", "--hex", type=str, default=None,
                        help="Target color specification (Hex, RGB, CIELAB, or name). Auto-detected from prompt if omitted.")
    parser.add_argument("--variant", choices=["sigma", "alpha"], default="sigma",
                        help="PixArt variant (sigma or alpha)")
    parser.add_argument("--object", type=str, default=None,
                        help="Target object to segment (e.g. mug). Auto-detected if omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Image resolution")
    parser.add_argument("--out-dir", type=str, default="./inference_outputs", help="Output directory")
    parser.add_argument("--no-baseline", action="store_true", help="Do not save unperturbed baseline image")
    parser.add_argument("--no-comparison", action="store_true", help="Do not save comparison figure")

    args = parser.parse_args()

    color_in_prompt = extract_color_spec_from_text(args.prompt)
    target_spec = args.target_color or color_in_prompt or "#A52A2A"

    run_color_steering_inference(
        prompt=args.prompt,
        target_color_spec=target_spec,
        object_word=args.object,
        variant=args.variant,
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

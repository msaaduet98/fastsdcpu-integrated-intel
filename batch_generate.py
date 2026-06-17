#!/usr/bin/env python3
"""
batch_generate.py — batch flat-2D-cartoon / line-art asset generator.

Drives FastSD CPU's OWN pipeline internals (no reinvented diffusion code):
it reuses src/state.py -> get_settings()/get_context() and
context.generate_text_to_image(), exactly like src/app.py's CLI path.

What it does:
  * reads a list of prompts from a .json or .txt file
  * wraps every prompt in a fixed flat-cartoon style template
    (thick black outline, flat colors, white background, no shading)
    + a shared negative prompt
  * runs OpenVINO (SDXS-512 or SD-Turbo) at 1 step, 512x512
  * uses a reproducible seed range (seed = seed_start + prompt_index)
  * saves each image named after its prompt into ./output/

Usage (CPU or iGPU):
    # iGPU via OpenVINO (set by Dockerfile.intel-igpu / compose):
    DEVICE=GPU python batch_generate.py
    # or explicitly:
    python batch_generate.py --device GPU --model sdxs --prompts prompts.json

Inside the iGPU container:
    docker compose -f docker-compose.intel-igpu.yml run --rm \
        fastsdcpu-igpu python batch_generate.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed style template — tweak these to change the look of every asset.
# ---------------------------------------------------------------------------
STYLE_PREFIX = "flat 2D cartoon illustration of "
STYLE_SUFFIX = (
    ", thick bold black outline, flat solid colors, simple vector style, "
    "clean line art, white background, no shading, no gradient, centered, "
    "minimal, sticker style"
)
NEGATIVE_PROMPT = (
    "photo, photorealistic, 3d, render, realistic, shading, gradient, "
    "shadow, texture, noise, grain, blurry, lowres, jpeg artifacts, "
    "watermark, text, signature, frame, border, complex background, "
    "cluttered, busy"
)

# OpenVINO model ids (from configs/openvino-lcm-models.txt)
MODEL_IDS = {
    "sdxs": "rupeshs/sdxs-512-0.9-openvino",
    "sd-turbo": "rupeshs/sd-turbo-openvino",
}


def parse_args():
    p = argparse.ArgumentParser(description="Batch flat-cartoon asset generator (FastSD CPU)")
    p.add_argument("--prompts", default="prompts.json",
                   help="Path to prompts file (.json list, or .txt one-per-line). Default: prompts.json")
    p.add_argument("--model", choices=list(MODEL_IDS), default="sdxs",
                   help="OpenVINO model. Default: sdxs (SDXS-512)")
    p.add_argument("--output", default="output", help="Output folder. Default: output")
    p.add_argument("--steps", type=int, default=1, help="Inference steps. Default: 1")
    p.add_argument("--width", type=int, default=512, help="Image width. Default: 512")
    p.add_argument("--height", type=int, default=512, help="Image height. Default: 512")
    p.add_argument("--seed-start", type=int, default=0,
                   help="First seed; prompt i uses seed_start + i. Default: 0")
    p.add_argument("--seed-end", type=int, default=None,
                   help="Optional upper bound of the seed range (informational/clamp warning).")
    p.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale. Default: 1.0")
    p.add_argument("--device", default=None,
                   help="Override DEVICE env (e.g. GPU for the Intel iGPU, or CPU). "
                        "Must be set BEFORE the pipeline imports, so this script re-reads it.")
    p.add_argument("--tiny-autoencoder", action="store_true",
                   help="Use Tiny AutoEncoder (TAESD) for faster decode.")
    return p.parse_args()


def load_prompts(prompts_path: Path):
    """Return a list of {prompt, seed?} dicts from a .json or .txt file."""
    if not prompts_path.exists():
        sys.exit(f"Prompts file not found: {prompts_path}")
    raw = prompts_path.read_text(encoding="utf-8")
    items = []
    if prompts_path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]
        for entry in data:
            if isinstance(entry, str):
                items.append({"prompt": entry})
            elif isinstance(entry, dict) and entry.get("prompt"):
                items.append({"prompt": entry["prompt"], "seed": entry.get("seed")})
    else:  # plain text, one prompt per non-empty, non-comment line
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append({"prompt": line})
    if not items:
        sys.exit(f"No prompts found in {prompts_path}")
    return items


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "asset"


def main():
    args = parse_args()

    # DEVICE must be set in the environment BEFORE FastSD CPU's modules import,
    # because src/constants.py reads it at import time (DEVICE -> GPU/CPU).
    if args.device:
        os.environ["DEVICE"] = args.device
    device = os.environ.get("DEVICE", "cpu")

    # Make FastSD CPU's flat `src/` imports resolvable (mirrors `python src/app.py`).
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))

    from models.interface_types import InterfaceType  # noqa: E402
    from state import get_settings, get_context        # noqa: E402

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompts))
    print(f"FastSD CPU batch generator | device={device} | model={args.model} "
          f"({MODEL_IDS[args.model]}) | {len(prompts)} prompt(s)")

    # skip_file=True -> use in-memory defaults, don't read/write configs/settings.yaml
    app_settings = get_settings(skip_file=True)
    context = get_context(InterfaceType.CLI)
    cfg = app_settings.settings
    lcm = cfg.lcm_diffusion_setting

    # Fixed generation config for flat-cartoon assets via OpenVINO.
    lcm.use_openvino = True
    lcm.openvino_lcm_model_id = MODEL_IDS[args.model]
    lcm.use_tiny_auto_encoder = args.tiny_autoencoder
    lcm.image_width = args.width
    lcm.image_height = args.height
    lcm.inference_steps = args.steps
    lcm.guidance_scale = args.guidance_scale
    lcm.number_of_images = 1
    lcm.negative_prompt = NEGATIVE_PROMPT
    lcm.use_seed = True
    lcm.use_safety_checker = False
    cfg.generated_images.save_image = False  # we name + save the images ourselves

    seeds_used = []
    for i, item in enumerate(prompts):
        seed = item.get("seed")
        if seed is None:
            seed = args.seed_start + i
        if args.seed_end is not None and seed > args.seed_end:
            print(f"  ! seed {seed} exceeds --seed-end {args.seed_end} (still using it)")

        styled = f"{STYLE_PREFIX}{item['prompt']}{STYLE_SUFFIX}"
        lcm.prompt = styled
        lcm.seed = seed

        print(f"[{i + 1}/{len(prompts)}] seed={seed} :: {item['prompt']}")
        images = context.generate_text_to_image(
            settings=cfg,
            device=device,
            save_config=False,
        )
        if not images:
            print(f"  ! generation failed for: {item['prompt']} (error: {context.error})")
            continue

        base = f"{i:03d}-{slugify(item['prompt'])}-seed{seed}"
        for n, image in enumerate(images):
            name = f"{base}.png" if len(images) == 1 else f"{base}-{n + 1}.png"
            out = output_dir / name
            image.save(out)
            print(f"  -> {out}")
        seeds_used.append(seed)

    print(f"Done. {len(seeds_used)} image set(s) written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

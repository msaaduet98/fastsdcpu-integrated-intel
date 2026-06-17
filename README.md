# fastsdcpu-integrated-intel

Fast **local** generation of simple flat 2D cartoon / line-art assets on an
**Intel integrated GPU** (no discrete GPU), using [OpenVINO] + a 1-step
diffusion model (SDXS-512 / SD-Turbo).

Built for an Intel **Core Ultra** laptop (iGPU, 32 GB RAM, Ubuntu).

---

## Relationship to the original project

This repo is based on **FastSD CPU** by **Rupesh Sreeraman**:
👉 https://github.com/rupeshs/fastsdcpu (MIT licensed)

It is **not a GitHub fork** — it is an independent repo with its own clean git
history, customized for personal use. All the core diffusion code under `src/`
is the upstream project's work; the original developer documentation is kept
intact in **[`Readme.md`](./Readme.md)**.

**My additions on top of upstream:**

| Added file | What it does |
|---|---|
| `Dockerfile.intel-igpu` | Image with the Intel OpenCL + Level Zero GPU runtime, `DEVICE=GPU` |
| `docker-compose.intel-igpu.yml` | Passes `/dev/dri` + `render` group, persistent volumes |
| `batch_generate.py` | Batch flat-cartoon asset generator (reuses FastSD CPU internals) |
| `prompts.json` | Sample prompt list |
| `svg_postprocess.sh` | Optional PNG → SVG vector tracing |

License: original MIT preserved in [`LICENSE`](./LICENSE) (incl. the original
copyright); my modifications are added under the same MIT license alongside
the original copyright notice.

---

## 1. Intel iGPU Docker workflow

OpenVINO talks to the Intel iGPU through the OpenCL "NEO" driver + Level Zero,
which are installed **inside** the image (`Dockerfile.intel-igpu`). The host
just shares its GPU device node.

### One-time host setup — set your `render` group GID

The container user must be in the group that owns `/dev/dri/renderD128`. Find
your host's `render` group GID:

```bash
getent group render
# e.g.  render:x:992:   <-- 992 is the GID
```

Open `docker-compose.intel-igpu.yml` and set the GID under `group_add:`
(it defaults to `992`, the value on the machine this repo was generated on —
**change it if yours differs**).

### Build + run the web UI

```bash
docker compose -f docker-compose.intel-igpu.yml up --build
# open http://localhost:7860, choose an OpenVINO model, generation runs on the iGPU
```

Verify the GPU is actually visible inside the container:

```bash
docker compose -f docker-compose.intel-igpu.yml run --rm fastsdcpu-igpu clinfo | grep -i "device name"
# should list your Intel iGPU
```

### Persistent volumes

- `hf_cache` (named volume) → downloaded HuggingFace / OpenVINO models survive rebuilds
- `./lora_models` → your LoRA files
- `./output` → images from the batch generator

---

## 2. Batch asset generation

`batch_generate.py` reuses FastSD CPU's own pipeline (`get_settings()` /
`get_context()` / `context.generate_text_to_image()` — the same path
`src/app.py` uses), so it is not a reimplementation.

It wraps every prompt in a fixed flat-cartoon style template (thick black
outline, flat colors, white background, no shading) plus a shared negative
prompt, runs OpenVINO SDXS-512 at **1 step, 512×512**, with a reproducible
seed range, and saves each image named after its prompt into `output/`.

### Prompts file

`prompts.json` — a JSON list. Entries are plain strings, or objects with a
fixed per-prompt seed:

```json
[
  "a smiling red apple",
  { "prompt": "a green potted cactus", "seed": 42 }
]
```

A plain `.txt` file (one prompt per line, `#` for comments) also works via
`--prompts mylist.txt`.

### Run inside the container (recommended — uses the iGPU)

```bash
docker compose -f docker-compose.intel-igpu.yml run --rm \
    fastsdcpu-igpu python batch_generate.py
```

### Run on the host directly

```bash
# iGPU via OpenVINO:
DEVICE=GPU python batch_generate.py --model sdxs --prompts prompts.json
# plain CPU:
DEVICE=CPU python batch_generate.py
```

### Options

```
--prompts PATH       prompts .json or .txt        (default: prompts.json)
--model {sdxs,sd-turbo}  OpenVINO model           (default: sdxs = SDXS-512)
--output DIR         output folder                (default: output)
--steps N            inference steps              (default: 1)
--width / --height   image size                   (default: 512 x 512)
--seed-start N       prompt i uses seed_start + i (default: 0)
--seed-end N         optional upper bound (warns if exceeded)
--device GPU|CPU     overrides the DEVICE env var
--tiny-autoencoder   faster decode via TAESD
```

Reproducibility: prompt *i* uses `seed_start + i` unless the prompt object
sets its own `seed`. Same prompts + same `--seed-start` ⇒ same images.

Output files are named `NNN-<prompt-slug>-seed<N>.png`.

---

## 3. Optional: PNG → SVG

Convert the flat line-art PNGs to clean vector SVG with `potrace` +
ImageMagick. **Host dependencies:**

```bash
sudo apt-get install potrace imagemagick inkscape
```

Then:

```bash
./svg_postprocess.sh                 # output/*.png  ->  output/svg/*.svg
./svg_postprocess.sh INPUT_DIR OUT_DIR
THRESHOLD=60% ./svg_postprocess.sh   # tune the black/white cutoff
```

Tracing works well on the thick-outline flat assets this repo generates;
it is not meant for photographic images.

---

## Credits

- **FastSD CPU** — original project by Rupesh Sreeraman:
  https://github.com/rupeshs/fastsdcpu
- See [`Readme.md`](./Readme.md) for the full upstream documentation and the
  complete model/feature list.

[OpenVINO]: https://github.com/openvinotoolkit/openvino

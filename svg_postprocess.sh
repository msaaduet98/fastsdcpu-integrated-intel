#!/usr/bin/env bash
# svg_postprocess.sh — convert generated flat line-art PNGs to clean SVG.
#
# Pipeline per image:
#   PNG --(ImageMagick)--> high-contrast black/white BMP --(potrace)--> SVG
#
# This works best on the flat-2D / thick-black-outline assets produced by
# batch_generate.py (line art traces cleanly; photos do not).
#
# ---------------------------------------------------------------------------
# DEPENDENCIES (install on the HOST, not the container):
#     sudo apt-get install potrace imagemagick inkscape
#   * imagemagick -> `convert` (preprocess to bitmap)
#   * potrace     -> raster -> vector tracing
#   * inkscape    -> optional, only used here to report/clean SVGs if present
# ---------------------------------------------------------------------------
#
# Usage:
#     ./svg_postprocess.sh                 # input=output/  output=output/svg/
#     ./svg_postprocess.sh INPUT_DIR OUT_DIR
#     THRESHOLD=60% ./svg_postprocess.sh   # tweak black/white cutoff
set -euo pipefail

IN_DIR="${1:-output}"
OUT_DIR="${2:-output/svg}"
THRESHOLD="${THRESHOLD:-50%}"   # ImageMagick -threshold cutoff

# --- dependency check -------------------------------------------------------
missing=()
command -v convert  >/dev/null 2>&1 || missing+=("imagemagick (convert)")
command -v potrace  >/dev/null 2>&1 || missing+=("potrace")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing dependencies: ${missing[*]}" >&2
  echo "Install with:  sudo apt-get install potrace imagemagick inkscape" >&2
  exit 1
fi
command -v inkscape >/dev/null 2>&1 || echo "note: inkscape not found (optional) — continuing." >&2

if [ ! -d "$IN_DIR" ]; then
  echo "Input dir not found: $IN_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
shopt -s nullglob
pngs=("$IN_DIR"/*.png "$IN_DIR"/*.PNG)
if [ "${#pngs[@]}" -eq 0 ]; then
  echo "No PNGs found in $IN_DIR" >&2
  exit 1
fi

count=0
for png in "${pngs[@]}"; do
  name="$(basename "${png%.*}")"
  bmp="$(mktemp --suffix=.bmp)"
  svg="$OUT_DIR/$name.svg"

  # 1) flatten to white, grayscale, hard threshold -> 1-bit bitmap potrace likes
  convert "$png" -flatten -colorspace Gray -threshold "$THRESHOLD" "$bmp"
  # 2) trace to SVG (--flat keeps it simple; -k threshold already applied)
  potrace "$bmp" --svg --output "$svg"
  rm -f "$bmp"

  echo "  -> $svg"
  count=$((count + 1))
done

echo "Done. $count SVG file(s) written to $OUT_DIR"

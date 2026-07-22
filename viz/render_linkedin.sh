#!/usr/bin/env bash
# Render NpuKit systolic Manim scene → LinkedIn square GIF + MP4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/viz/.venv"
OUT="$ROOT/viz/out"
MEDIA="$OUT/manim"
SCENE=SystolicArrayScene

if [[ ! -x "$VENV/bin/manim" ]]; then
  echo "Create the venv first:"
  echo "  python3.13 -m venv viz/.venv && viz/.venv/bin/pip install manim numpy"
  exit 1
fi

mkdir -p "$OUT"

echo "==> Rendering Manim scene (1080×1080)…"
"$VENV/bin/manim" -qm --disable_caching \
  -r 1080,1080 \
  --media_dir "$MEDIA" \
  "$ROOT/viz/systolic_manim.py" "$SCENE"

# Prefer the newest / highest-res finished scene file (ignore partials)
MP4="$(find "$MEDIA/videos" -path '*/partial_movie_files/*' -prune -o \
  -name "${SCENE}.mp4" -print \
  | while read -r f; do
      w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$f")
      echo "$w $f"
    done \
  | sort -nr \
  | head -n1 \
  | awk '{print $2}')"

if [[ -z "${MP4:-}" || ! -f "$MP4" ]]; then
  echo "Could not find rendered MP4"
  exit 1
fi

cp "$MP4" "$OUT/systolic_8x8_manim.mp4"
echo "==> MP4: $OUT/systolic_8x8_manim.mp4 ($(du -h "$OUT/systolic_8x8_manim.mp4" | awk '{print $1}'))"

echo "==> Building palette-optimized GIF (720², ~LinkedIn-friendly size)…"
PALETTE="$OUT/_palette.png"
GIF="$OUT/systolic_8x8_manim.gif"

# 12 fps + 720px + fewer colors keeps the GIF under ~5–8 MB for feed sharing
ffmpeg -y -i "$MP4" \
  -vf "fps=12,scale=720:720:flags=lanczos,palettegen=max_colors=128:stats_mode=diff" \
  "$PALETTE"
ffmpeg -y -i "$MP4" -i "$PALETTE" \
  -lavfi "fps=12,scale=720:720:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
  -loop 0 "$GIF"
rm -f "$PALETTE"

echo "==> GIF: $GIF ($(du -h "$GIF" | awk '{print $1}'))"
echo "Tip: for LinkedIn, native video often looks sharper — upload the MP4."
echo "Done."

#!/usr/bin/env python3
"""Animated LinkedIn GIF: MNIST digits results (no intro title slide).

Beats:
  1) Fabric + FPGA util
  2) Tiny-ViT geometry / params
  3) Deploy split (CPU stem + FPGA GEMM + A9 float)
  4) DS-CNN peer vs ViT accuracy / KiB
  5) Inference latency (A9 bench)
  6) Board smoke result

Usage:
  python3 viz/edge_peers_anim.py
  → viz/out/edge_peers.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1080
FPS = 6
HOLD = 24  # ~4s per beat at 6 fps
FADE = 6

BG = (12, 18, 28)
PANEL = (20, 28, 42)
PANEL2 = (26, 36, 52)
TITLE = (245, 248, 252)
MUTED = (140, 156, 178)
MINT = (80, 200, 180)
AMBER = (240, 170, 70)
OK = (120, 210, 160)

OUT_DIR = Path(__file__).resolve().parent / "out"
GIF_PATH = OUT_DIR / "edge_peers.gif"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def text_center(d: ImageDraw.ImageDraw, y: int, s: str, f, fill=TITLE) -> None:
    bbox = d.textbbox((0, 0), s, font=f)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, y), s, font=f, fill=fill)


def rounded(d: ImageDraw.ImageDraw, xy, fill, r: int = 28) -> None:
    d.rounded_rectangle(xy, radius=r, fill=fill)


def draw_chip(
    d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, sub: str, accent
) -> None:
    rounded(d, (x, y, x + w, y + h), PANEL, 24)
    d.rounded_rectangle((x, y, x + 10, y + h), radius=4, fill=accent)
    d.text((x + 28, y + 22), label, font=font(32, True), fill=TITLE)
    d.text((x + 28, y + 72), sub, font=font(24), fill=MUTED)


def scene_fabric(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 60, "PYNQ-Z2  ·  Zynq-7020 fabric", font(40, True))
    text_center(d, 120, "MNIST digits  ·  100 MHz  ·  VERSION 0x300", font(26), MUTED)

    y0 = 200
    gx = int(lerp(-40, 90, ease(min(1.0, progress * 1.4))))
    draw_chip(
        d,
        gx,
        y0,
        900,
        150,
        "8×8 int8 systolic GEMM",
        "output-stationary · host-tiled DMA / MMIO",
        MINT,
    )
    gx2 = int(lerp(-40, 90, ease(max(0.0, progress * 1.4 - 0.2))))
    draw_chip(
        d,
        gx2,
        y0 + 180,
        900,
        150,
        "Transformer glue (optional)",
        "residual · GELU · RMSNorm · Softmax  ·  MAX_LEN=16",
        AMBER,
    )

    # Util strip
    cy = int(lerp(1100, 600, ease(max(0.0, progress * 1.2 - 0.3))))
    rounded(d, (90, cy, 990, cy + 320), PANEL2, 24)
    d.text((130, cy + 28), "Placed utilization", font=font(30, True), fill=TITLE)
    util = [
        ("LUT", "~18%", "9.6k / 53.2k"),
        ("FF", "~12%", "12.9k / 106k"),
        ("DSP", "~33%", "72 / 220"),
        ("BRAM", "~1.4%", "2 / 140"),
    ]
    for i, (name, pct, detail) in enumerate(util):
        x = 130 + (i % 2) * 420
        y = cy + 90 + (i // 2) * 100
        d.text((x, y), name, font=font(26, True), fill=MUTED)
        d.text((x + 110, y), pct, font=font(34, True), fill=MINT if name == "DSP" else OK)
        d.text((x + 110, y + 42), detail, font=font(22), fill=MUTED)
    return im


def scene_params(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 60, "Tiny-ViT parameters (digits)", font(40, True))
    text_center(d, 120, "native 28×28  ·  no resize", font(26), MUTED)

    rows = [
        ("Input", "28×28 grayscale MNIST"),
        ("CPU DS-stem", "MID=24 → D=16 tokens (T=16)"),
        ("Body", "T=16 · D=16 · MLP=32 · L=4"),
        ("Quant", "int8 GEMM + per-channel W scales"),
        ("Norms", "Softmax / RMSNorm / GELU on A9 float"),
        ("Size", "~12.2k params  ·  ~13 KiB weights"),
        ("Head", "10-class linear on CPU"),
    ]
    top = 190
    for ri, (k, v) in enumerate(rows):
        t = ease(max(0.0, min(1.0, progress * 1.5 - ri * 0.08)))
        if t <= 0:
            continue
        y = top + ri * 100
        rounded(d, (80, y, 1000, y + 84), PANEL, 18)
        d.text((110, y + 24), k, font=font(26, True), fill=MUTED)
        d.text((340, y + 24), v, font=font(26, True), fill=TITLE)
    return im


def scene_split_compute(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 70, "Where compute runs", font(42, True))
    text_center(d, 130, "CPU/FPGA combo for the ViT peer", font(28), MUTED)

    boxes = [
        ("28×28", "image"),
        ("DS-stem", "A9 int8"),
        ("GEMM", "FPGA int8"),
        ("norms", "A9 float"),
        ("head", "A9"),
    ]
    n = len(boxes)
    for i, (a, b) in enumerate(boxes):
        t = ease(max(0.0, min(1.0, progress * 1.5 - i * 0.1)))
        x = 55 + i * 200
        y = 280
        rounded(d, (x, y, x + 180, y + 170), PANEL if t > 0.05 else BG, 20)
        if t > 0.2:
            d.text((x + 18, y + 40), a, font=font(26, True), fill=TITLE)
            d.text((x + 18, y + 100), b, font=font(22), fill=MINT if "FPGA" in b else MUTED)
        if i < n - 1 and t > 0.5:
            d.polygon(
                [(x + 186, y + 75), (x + 198, y + 85), (x + 186, y + 95)],
                fill=AMBER,
            )

    cy = int(lerp(1100, 560, ease(max(0.0, progress * 1.2 - 0.35))))
    rounded(d, (90, cy, 990, cy + 280), PANEL2, 24)
    d.text((130, cy + 36), "Design choice", font=font(32, True), fill=AMBER)
    d.text(
        (130, cy + 100),
        "Keep the heavy matmuls on the 8×8 systolic array.\n"
        "Tiny DS-stem + Softmax/RMSNorm/GELU stay on the A9\n"
        "in float — better accuracy than LUT glue for this size.",
        font=font(26),
        fill=TITLE,
    )
    return im


def scene_peers(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 60, "MNIST edge peers", font(42, True))
    text_center(d, 120, "same digits task · not a param-matched bake-off", font(26), MUTED)

    slide = ease(progress)
    left_x = int(lerp(-500, 70, slide))
    right_x = int(lerp(W + 40, 560, slide))

    rounded(d, (left_x, 200, left_x + 450, 860), PANEL, 28)
    d.text((left_x + 36, 240), "MCU-class peer", font=font(26, True), fill=MINT)
    d.text((left_x + 36, 300), "DS-CNN", font=font(48, True), fill=TITLE)
    d.text((left_x + 36, 380), "depthwise-separable CNN\nhost int8 TinyML", font=font(26), fill=MUTED)
    d.text((left_x + 36, 520), "98.39%", font=font(52, True), fill=OK)
    d.text((left_x + 36, 590), "full 10k test · int8", font=font(24), fill=MUTED)
    d.text((left_x + 36, 680), "~9.0k params", font=font(28, True), fill=TITLE)
    d.text((left_x + 36, 740), "~8.3 KiB weights", font=font(28), fill=MUTED)
    d.text((left_x + 36, 800), "CPU only · not on FPGA", font=font(24), fill=MUTED)

    rounded(d, (right_x, 200, right_x + 450, 860), PANEL, 28)
    d.text((right_x + 36, 240), "CPU + NpuKit", font=font(26, True), fill=AMBER)
    d.text((right_x + 36, 300), "Tiny-ViT", font=font(48, True), fill=TITLE)
    d.text((right_x + 36, 380), "DS-stem + FPGA GEMM\n+ A9 float norms", font=font(26), fill=MUTED)
    d.text((right_x + 36, 520), "97.98%", font=font(52, True), fill=OK)
    d.text((right_x + 36, 590), "full 10k · deploy-quant", font=font(24), fill=MUTED)
    d.text((right_x + 36, 680), "~12.2k params", font=font(28, True), fill=TITLE)
    d.text((right_x + 36, 740), "~13 KiB weights", font=font(28), fill=MUTED)
    d.text((right_x + 36, 800), "board: ALL VIT PASS", font=font(24), fill=MUTED)
    return im


def scene_latency(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 60, "Inference time (A9 host)", font(42, True))
    text_center(d, 120, "PYNQ-Z2  ·  n=64  ·  includes host schedule / DMA", font(26), MUTED)

    rows = [
        ("Path", "ms / image"),
        ("DS-CNN int8 CPU (single)", "~230 ms"),
        ("DS-CNN int8 CPU (batch avg)", "~213 ms"),
        ("ViT deploy numpy CPU", "~118 ms"),
        ("ViT FPGA end-to-end", "~977 ms"),
    ]
    top = 220
    for ri, (a, b) in enumerate(rows):
        t = ease(max(0.0, min(1.0, progress * 1.5 - ri * 0.1)))
        if t <= 0:
            continue
        y = top + ri * 100
        bg = PANEL2 if ri == 0 else PANEL
        rounded(d, (80, y, 1000, y + 84), bg, 18)
        fill_a = MINT if ri == 0 else TITLE
        fill_b = MINT if ri == 0 else (AMBER if "FPGA" in a else OK)
        d.text((120, y + 24), a, font=font(28, True if ri == 0 else False), fill=fill_a)
        d.text((720, y + 24), b, font=font(30, True), fill=fill_b)

    text_center(
        d,
        780,
        "FPGA path = stem + tiled GEMM + norms + head (not PE-only)",
        font(24),
        MUTED,
    )
    text_center(
        d,
        840,
        "Untuned host schedule — latency is honest, not marketing FLOPs",
        font(24),
        MUTED,
    )
    return im


def scene_board(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 80, "Board smoke (digits)", font(42, True))
    text_center(d, 150, "npukit_board_smoke.ipynb  ·  same bitstream", font(26), MUTED)

    mark = ease(progress)
    text_center(d, 280, "ALL SMOKE PASS", font(56, True), OK if mark > 0.25 else MUTED)
    text_center(d, 360, "matmul · glue · e2e · ViT", font(28), MUTED)

    rounded(d, (120, 460, 960, 860), PANEL, 24)
    lines = [
        ("ViT sample n=64", "61/64 (95.3%) ref & hw"),
        ("ref ↔ hw agree", "64/64 (100%)"),
        ("tensor max|err|", "0  (GEMM bit-exact vs numpy)"),
        ("Full 10k deploy-quant", "97.98%"),
        ("vs DS-CNN int8", "−0.41 pp  ·  +~5 KiB"),
    ]
    for i, (k, v) in enumerate(lines):
        y = 500 + i * 60
        d.text((160, y), k, font=font(26), fill=MUTED)
        d.text((520, y), v, font=font(26, True), fill=TITLE)
    return im


def scene_why(progress: float) -> Image.Image:
    """Story beat: small FPGA + tiny transformer → sequential-data upside."""
    im, d = new_frame()
    text_center(d, 60, "Why a tiny transformer on cheap FPGA?", font(36, True))
    text_center(d, 120, "digits prove the path · sequence work is the upside", font(26), MUTED)

    # Left: small silicon
    t0 = ease(min(1.0, progress * 1.3))
    lx = int(lerp(-480, 70, t0))
    rounded(d, (lx, 200, lx + 450, 520), PANEL, 28)
    d.text((lx + 36, 240), "Small · low-cost FPGA", font=font(28, True), fill=MINT)
    d.text(
        (lx + 36, 310),
        "Zynq-7020 class board\n~18% LUT · ~33% DSP\n8×8 int8 GEMM fits with\nheadroom to spare",
        font=font(26),
        fill=TITLE,
    )

    # Right: tiny model
    t1 = ease(max(0.0, progress * 1.3 - 0.15))
    rx = int(lerp(W + 40, 560, t1))
    rounded(d, (rx, 200, rx + 450, 520), PANEL, 28)
    d.text((rx + 36, 240), "Tiny transformer", font=font(28, True), fill=AMBER)
    d.text(
        (rx + 36, 310),
        "~12k params · ~13 KiB\nT=16 · D=16 · L=4\nsame size class as a\nTinyML CNN peer",
        font=font(26),
        fill=TITLE,
    )

    # Bottom: sequential upside
    cy = int(lerp(1100, 580, ease(max(0.0, progress * 1.2 - 0.35))))
    rounded(d, (70, cy, 1010, cy + 360), PANEL2, 24)
    d.text((110, cy + 28), "Where transformers win", font=font(32, True), fill=OK)
    d.text(
        (110, cy + 90),
        "Attention is built for sequential / contextual data —\n"
        "tokens that depend on each other across time or space.\n\n"
        "MNIST is the smoke test. The same GEMM + tiny block\n"
        "is a path to sensor streams, short audio, control traces,\n"
        "and other edge sequences CNNs handle less naturally.",
        font=font(26),
        fill=TITLE,
    )
    return im


def hold(frames: list[Image.Image], im: Image.Image, n: int = HOLD) -> None:
    for _ in range(n):
        frames.append(im.copy())


def crossfade(frames: list[Image.Image], a: Image.Image, b: Image.Image, n: int = FADE) -> None:
    for i in range(1, n + 1):
        frames.append(Image.blend(a, b, i / n))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []

    # No intro title slide — start on fabric/util meat.
    for i in range(HOLD + 4):
        frames.append(scene_fabric((i + 1) / (HOLD + 4)))
    s0 = scene_fabric(1.0)
    hold(frames, s0, 8)

    for i in range(HOLD + 4):
        frames.append(scene_params((i + 1) / (HOLD + 4)))
    s1 = scene_params(1.0)
    hold(frames, s1, 8)
    crossfade(frames, s1, scene_split_compute(1.0))

    for i in range(HOLD + 6):
        frames.append(scene_split_compute((i + 1) / (HOLD + 6)))
    s2 = scene_split_compute(1.0)
    hold(frames, s2, 8)

    for i in range(HOLD + 4):
        frames.append(scene_peers(min(1.0, (i + 1) / 12)))
    s3 = scene_peers(1.0)
    hold(frames, s3, 10)

    for i in range(HOLD + 4):
        frames.append(scene_latency((i + 1) / (HOLD + 4)))
    s4 = scene_latency(1.0)
    hold(frames, s4, 10)

    for i in range(HOLD):
        frames.append(scene_board((i + 1) / HOLD))
    s5 = scene_board(1.0)
    hold(frames, s5, HOLD + 4)
    crossfade(frames, s5, scene_why(1.0))

    for i in range(HOLD + 6):
        frames.append(scene_why((i + 1) / (HOLD + 6)))
    hold(frames, scene_why(1.0), HOLD + 8)

    scaled = [fr.resize((720, 720), Image.Resampling.LANCZOS) for fr in frames]
    imageio.mimsave(GIF_PATH, scaled, fps=FPS, loop=0)
    print(f"wrote {GIF_PATH}  frames={len(scaled)}  size={GIF_PATH.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()

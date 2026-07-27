#!/usr/bin/env python3
"""Animated LinkedIn/README GIF: MCU DS-CNN vs MCU+NpuKit tiny-ViT.

Story beats (MNIST edge peers — not a param bake-off):
  1) NpuKit hardware (GEMM + glue)
  2) Two peers on the same task
  3) DS-CNN = MCU-class TinyML (~98% int8)
  4) Tiny-ViT: CPU DS-stem + FPGA int8 GEMM + A9 float norms (~97.4%)
  5) Compare accuracy / KiB / where compute runs

Usage:
  python3 viz/edge_peers_anim.py
  → viz/out/edge_peers.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1080
FPS = 10
HOLD = 14  # frames per beat (~1.4s)
FADE = 4

BG = (12, 18, 28)
PANEL = (20, 28, 42)
PANEL2 = (26, 36, 52)
TITLE = (245, 248, 252)
MUTED = (140, 156, 178)
MINT = (80, 200, 180)
AMBER = (240, 170, 70)
CORAL = (230, 120, 100)
OK = (120, 210, 160)
LINE = (48, 62, 84)

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


def draw_chip(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, sub: str, accent) -> None:
    rounded(d, (x, y, x + w, y + h), PANEL, 24)
    d.rounded_rectangle((x, y, x + 10, y + h), radius=4, fill=accent)
    d.text((x + 28, y + 22), label, font=font(34, True), fill=TITLE)
    d.text((x + 28, y + 70), sub, font=font(24), fill=MUTED)


def scene_title(alpha: float = 1.0) -> Image.Image:
    im, d = new_frame()
    text_center(d, 320, "NpuKit", font(72, True), TITLE)
    text_center(d, 420, "FPGA mini-NPU on PYNQ-Z2", font(36), MINT)
    text_center(d, 500, "8×8 int8 systolic GEMM  ·  transformer glue", font(28), MUTED)
    text_center(d, 700, "Edge peers on MNIST", font(32, True), AMBER)
    if alpha < 1.0:
        overlay = Image.new("RGB", (W, H), BG)
        im = Image.blend(overlay, im, alpha)
    return im


def scene_hw(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 80, "What’s on the fabric", font(44, True))
    y0 = 220
    # GEMM block slides in
    gx = int(lerp(-40, 90, ease(min(1.0, progress * 1.4))))
    draw_chip(d, gx, y0, 900, 160, "8×8 int8 systolic GEMM", "output-stationary · AXI DMA / MMIO tiling", MINT)
    # Glue block
    gx2 = int(lerp(-40, 90, ease(max(0.0, progress * 1.4 - 0.25))))
    draw_chip(
        d,
        gx2,
        y0 + 200,
        900,
        160,
        "Transformer glue  (MAX_LEN=16)",
        "residual · GELU · RMSNorm · Softmax  @ 100 MHz",
        AMBER,
    )
    text_center(d, 920, "A9 host: quant, scales, schedule, RoPE / masks", font(26), MUTED)
    return im


def scene_split(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 70, "Same task · two deployments", font(42, True))
    text_center(d, 130, "MNIST  ·  not a param-matched bake-off", font(26), MUTED)

    slide = ease(progress)
    left_x = int(lerp(-500, 70, slide))
    right_x = int(lerp(W + 40, 560, slide))

    # Left: MCU CNN
    rounded(d, (left_x, 240, left_x + 450, 820), PANEL, 28)
    d.text((left_x + 36, 280), "MCU path", font=font(28, True), fill=MINT)
    d.text((left_x + 36, 340), "DS-CNN", font=font(48, True), fill=TITLE)
    d.text((left_x + 36, 420), "TinyML depthwise-\nseparable CNN", font=font(28), fill=MUTED)
    d.text((left_x + 36, 560), "int8  ~98.4%", font=font(36, True), fill=OK)
    d.text((left_x + 36, 620), "~8.3 KiB weights", font=font(28), fill=MUTED)
    d.text((left_x + 36, 700), "runs on MCU-class host\n(not mapped to FPGA)", font=font(24), fill=MUTED)

    # Right: ViT + accel
    rounded(d, (right_x, 240, right_x + 450, 820), PANEL, 28)
    d.text((right_x + 36, 280), "MCU/MPU + NpuKit", font=font(28, True), fill=AMBER)
    d.text((right_x + 36, 340), "Tiny-ViT", font=font(48, True), fill=TITLE)
    d.text((right_x + 36, 420), "T=16 · D=16 · L=4\nMLP=32  deploy-quant", font=font(28), fill=MUTED)
    d.text((right_x + 36, 560), "quant  ~97.4%", font=font(36, True), fill=OK)
    d.text((right_x + 36, 620), "~11.5 KiB weights", font=font(28), fill=MUTED)
    d.text((right_x + 36, 700), "int8 GEMM on FPGA\nstem+norms on A9", font=font(24), fill=MUTED)
    return im


def scene_split_compute(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 80, "ViT deploy split (this kit)", font(42, True))
    text_center(d, 150, "int8 where the flops are · float where it helps", font(28), MUTED)

    boxes = [
        ("28×28", "image"),
        ("DS-stem", "CPU int8"),
        ("GEMM body", "FPGA int8"),
        ("norms", "A9 float"),
    ]
    n = len(boxes)
    for i, (a, b) in enumerate(boxes):
        t = ease(max(0.0, min(1.0, progress * 1.5 - i * 0.12)))
        x = 100 + i * 230
        y = 320
        rounded(d, (x, y, x + 200, y + 160), PANEL if t > 0.05 else BG, 20)
        if t > 0.2:
            d.text((x + 20, y + 36), a, font=font(26, True), fill=TITLE)
            d.text((x + 20, y + 90), b, font=font(22), fill=MUTED)
        if i < n - 1 and t > 0.5:
            d.polygon(
                [(x + 208, y + 70), (x + 228, y + 80), (x + 208, y + 90)],
                fill=MINT,
            )

    cy = int(lerp(1100, 620, ease(max(0.0, progress * 1.2 - 0.35))))
    rounded(d, (120, cy, 960, cy + 220), (28, 36, 32), 24)
    d.text((160, cy + 36), "Keep GEMM on the NPU", font=font(40, True), fill=MINT)
    d.text(
        (160, cy + 100),
        "Tiny CPU DS-stem + Softmax/RMSNorm/GELU on A9 float32.\n"
        "Transformer matmuls stay int8 on the systolic array.",
        font=font(26),
        fill=TITLE,
    )
    return im


def scene_table(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 70, "Compare what matters", font(42, True), TITLE)
    text_center(d, 130, "accuracy  ·  footprint  ·  where compute runs", font(26), MUTED)

    rows = [
        ("", "DS-CNN (MCU)", "Tiny-ViT (NpuKit)"),
        ("Task", "MNIST", "MNIST"),
        ("Deploy shape", "host int8 TinyML", "FPGA GEMM + A9"),
        ("Accuracy", "~98.4% int8", "~97.4% quant"),
        ("Weights", "~8.3 KiB", "~11.5 KiB"),
        ("Front-end", "is the CNN", "tiny DS-stem"),
    ]
    top = 220
    col_x = [80, 320, 700]
    for ri, row in enumerate(rows):
        t = ease(max(0.0, min(1.0, progress * 1.6 - ri * 0.08)))
        if t <= 0:
            continue
        y = top + ri * 90
        bg = PANEL2 if ri == 0 else PANEL
        rounded(d, (60, y, 1020, y + 78), bg, 16)
        for ci, cell in enumerate(row):
            fill = MINT if ri == 0 and ci > 0 else (AMBER if ri == 0 else TITLE)
            if ri > 0 and ci == 0:
                fill = MUTED
            d.text((col_x[ci], y + 22), cell, font=font(26, bold=(ri == 0 or ci == 0)), fill=fill)
    return im


def scene_end(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 300, "Board bring-up", font(48, True))
    text_center(d, 380, "npukit_board_smoke.ipynb", font(32), MINT)
    mark = ease(progress)
    text_center(d, 520, "ALL SMOKE PASS", font(52, True), OK if mark > 0.3 else MUTED)
    text_center(d, 620, "matmul · glue · e2e · ViT", font(28), MUTED)
    text_center(d, 820, "github.com/NetKit-Labs/npukit", font(26), AMBER)
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

    s0 = scene_title()
    hold(frames, s0, HOLD + 4)

    s1 = scene_hw(1.0)
    # build-up of hw scene
    for i in range(HOLD):
        frames.append(scene_hw((i + 1) / HOLD))
    hold(frames, s1, 6)
    crossfade(frames, s1, scene_split(1.0))

    for i in range(HOLD + 4):
        frames.append(scene_split(min(1.0, (i + 1) / 10)))
    s2 = scene_split(1.0)
    hold(frames, s2, 8)

    for i in range(HOLD + 6):
        frames.append(scene_split_compute((i + 1) / (HOLD + 6)))
    s3 = scene_split_compute(1.0)
    hold(frames, s3, 10)

    for i in range(HOLD + 4):
        frames.append(scene_table((i + 1) / (HOLD + 4)))
    s4 = scene_table(1.0)
    hold(frames, s4, 10)

    for i in range(HOLD):
        frames.append(scene_end((i + 1) / HOLD))
    hold(frames, scene_end(1.0), HOLD + 6)

    # Write GIF (Pillow) — LinkedIn-friendly scale 720²
    scaled = [fr.resize((720, 720), Image.Resampling.LANCZOS) for fr in frames]
    imageio.mimsave(GIF_PATH, scaled, fps=FPS, loop=0)
    print(f"wrote {GIF_PATH}  frames={len(scaled)}  size={GIF_PATH.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Animated GIF: speech peer findings (bag CNN vs order transformer).

No splash. Only the long race + the fair order-only race (no KWS/FSM peer B).

Beats:
  1) Same mel, two peers (A CNN · B Hybrid Transformer)
  2) Long scripts: bag CNN wins (84.5% vs 50%)
  3) Why: global pool = bag of features
  4) Fair setup (order-only, strip the bag)
  5) Fair results: transformer wins (96.5% vs 27.8%)
  6) Takeaway

Usage:
  python3 viz/speech_peers_anim.py
  → viz/out/speech_peers.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1080
FPS = 6
HOLD = 30  # ~5 s per settled beat at 6 fps
FADE = 8

BG = (12, 18, 28)
PANEL = (20, 28, 42)
PANEL2 = (26, 36, 52)
TITLE = (245, 248, 252)
MUTED = (140, 156, 178)
MINT = (80, 200, 180)
AMBER = (240, 170, 70)
OK = (120, 210, 160)
BAD = (230, 120, 110)
CNN = (100, 180, 220)
TR = (200, 150, 255)

OUT_DIR = Path(__file__).resolve().parent / "out"
GIF_PATH = OUT_DIR / "speech_peers.gif"


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


def bar(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    frac: float,
    fill,
    label: str,
    value: str,
) -> None:
    rounded(d, (x, y, x + w, y + h), PANEL, 18)
    inner = int(max(8, (w - 24) * max(0.0, min(1.0, frac))))
    d.rounded_rectangle((x + 12, y + h - 28, x + 12 + inner, y + h - 12), radius=6, fill=fill)
    d.text((x + 20, y + 16), label, font=font(26, True), fill=TITLE)
    bbox = d.textbbox((0, 0), value, font=font(34, True))
    d.text((x + w - 20 - (bbox[2] - bbox[0]), y + 12), value, font=font(34, True), fill=fill)


def scene_setup(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 48, "Same log-mel  ·  same phrases", font(38, True))
    text_center(d, 108, "16 kHz  ·  n_fft=512  ·  hop=256 (50%)  ·  32 mels", font(24), MUTED)

    t0 = ease(min(1.0, progress * 1.3))
    y_data = int(lerp(1100, 180, t0))
    rounded(d, (70, y_data, 1010, y_data + 200), PANEL2, 24)
    d.text((110, y_data + 36), "Dataset", font=font(28, True), fill=MINT)
    d.text(
        (110, y_data + 90),
        "Google Speech Commands words, stitched into\n"
        "robotics command phrases (concatenated clips)",
        font=font(26),
        fill=TITLE,
    )

    items = [
        ("A  Fat DS-CNN", "full-phrase classifier  ·  global pool", CNN),
        ("B  Hybrid Transformer", "stem → tokens → intent", TR),
    ]
    for i, (lab, sub, accent) in enumerate(items):
        t = ease(max(0.0, min(1.0, progress * 1.4 - 0.15 - i * 0.2)))
        y = int(lerp(1200, 420 + i * 240, t))
        rounded(d, (90, y, 990, y + 200), PANEL, 26)
        d.rounded_rectangle((90, y, 104, y + 200), radius=4, fill=accent)
        d.text((140, y + 48), lab, font=font(38, True), fill=TITLE)
        d.text((140, y + 118), sub, font=font(26), fill=MUTED)
    return im


def scene_long_wins(progress: float) -> Image.Image:
    """Long 9-word scripts: bag CNN wins."""
    im, d = new_frame()
    text_center(d, 50, "Long scripts  ·  ~9.7 s  ·  9 words", font(36, True))
    text_center(d, 112, "global-pool CNN still wins phrase ID", font(26), MUTED)

    t = ease(progress)
    y0 = int(lerp(1100, 220, t))
    rounded(d, (70, y0, 1010, y0 + 520), PANEL, 26)
    d.text((110, y0 + 36), "Accuracy (host)", font=font(28, True), fill=MINT)
    bar(d, 110, y0 + 110, 860, 120, 0.845, CNN, "A  Fat DS-CNN", "84.5%")
    bar(d, 110, y0 + 270, 860, 120, 0.50, TR, "B  Hybrid Transformer", "50.0%")
    d.text(
        (110, y0 + 430),
        "Same canvas, same labels — pooling CNN dominates.",
        font=font(26),
        fill=MUTED,
    )
    return im


def scene_why_bag(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 56, "Why the CNN looked unbeatable", font(36, True))
    text_center(d, 118, "global pool turns ~10 s into a bag of local cues", font(24), MUTED)

    t = ease(progress)
    boxes = [
        (120, "mel frames\n~1023", MUTED),
        (400, "stride 5/7\nkernels", CNN),
        (680, "AdaptiveAvgPool\n→ 1 vector", AMBER),
    ]
    for i, (x, lab, accent) in enumerate(boxes):
        yy = int(lerp(900, 280, ease(max(0.0, t * 1.3 - i * 0.12))))
        rounded(d, (x, yy, x + 240, yy + 200), PANEL, 22)
        d.rounded_rectangle((x, yy, x + 240, yy + 10), radius=4, fill=accent)
        d.text((x + 28, yy + 60), lab, font=font(26, True), fill=TITLE)
        if i < 2:
            ax = x + 250
            ay = yy + 100
            alpha = ease(max(0.0, t * 1.3 - i * 0.12 - 0.2))
            if alpha > 0.05:
                d.polygon([(ax, ay - 12), (ax + 36, ay), (ax, ay + 12)], fill=MUTED)

    cy = int(lerp(1100, 560, ease(max(0.0, t - 0.35))))
    rounded(d, (90, cy, 990, cy + 360), PANEL2, 24)
    d.text((130, cy + 36), "Bag of features", font=font(34, True), fill=AMBER)
    d.text(
        (130, cy + 110),
        "Order is mostly discarded.\n"
        "Great for “which sounds appear in this clip.”\n"
        "Weak when classes share the same words\n"
        "and only the sequence differs.",
        font=font(28),
        fill=TITLE,
    )
    return im


def scene_fair_setup(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 50, "Fair race: make order the signal", font(36, True))
    text_center(d, 112, "same 8-word multiset  ·  16 permutations", font(24), MUTED)

    t = ease(progress)
    left = int(lerp(-500, 70, t))
    right = int(lerp(W + 40, 560, t))
    rounded(d, (left, 200, left + 450, 420), PANEL, 24)
    d.text((left + 28, 230), "class 0", font=font(24), fill=MUTED)
    d.text(
        (left + 28, 280),
        "go left right forward\nstop on off learn",
        font=font(28, True),
        fill=TITLE,
    )

    rounded(d, (right, 200, right + 450, 420), PANEL, 24)
    d.text((right + 28, 230), "class 1  (same bag)", font=font(24), fill=MUTED)
    d.text(
        (right + 28, 280),
        "go right left forward\nstop on off learn",
        font=font(28, True),
        fill=TITLE,
    )

    cy = int(lerp(1100, 480, ease(max(0.0, t - 0.2))))
    rounded(d, (70, cy, 1010, cy + 460), PANEL2, 24)
    d.text((110, cy + 28), "Rules of the fair fight", font=font(32, True), fill=OK)
    rows = [
        ("A", "Causal DS-CNN — no global pool; last time column only", CNN),
        ("B", "Hybrid stem → T=128 → last-token + word-order aux", TR),
        ("", "Right-aligned mel so “last” = end of the command", MUTED),
    ]
    for i, (tag, line, col) in enumerate(rows):
        y = cy + 100 + i * 100
        if tag:
            d.text((110, y), tag, font=font(30, True), fill=col)
            d.text((170, y), line, font=font(26), fill=TITLE)
        else:
            d.text((110, y), line, font=font(26), fill=MUTED)
    return im


def scene_fair_results(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 50, "Fair results", font(42, True))
    text_center(d, 112, "order-only commands  ·  bagging disallowed", font(24), MUTED)

    t = ease(progress)
    ax = int(lerp(-480, 70, t))
    cx = int(lerp(W + 40, 560, ease(max(0.0, t - 0.1))))

    rounded(d, (ax, 220, ax + 450, 720), PANEL, 28)
    d.text((ax + 40, 270), "A  Causal CNN", font=font(30, True), fill=CNN)
    d.text((ax + 40, 360), "27.8%", font=font(72, True), fill=BAD)
    d.text(
        (ax + 40, 470),
        "last-frame only\nno AdaptiveAvgPool\n~16.7 ms / phrase",
        font=font(26),
        fill=MUTED,
    )

    rounded(d, (cx, 220, cx + 450, 720), PANEL, 28)
    d.text((cx + 40, 270), "B  Hybrid Transformer", font=font(28, True), fill=TR)
    d.text((cx + 40, 360), "96.5%", font=font(72, True), fill=OK)
    d.text(
        (cx + 40, 470),
        "T=128 · last-token\nword-sequence aux\n~15.4 ms / phrase",
        font=font(26),
        fill=MUTED,
    )

    if t > 0.55:
        text_center(d, 780, "B ≫ A   when order is the label", font(32, True), OK)
    return im


def scene_takeaway(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 56, "What this means", font(40, True))
    text_center(
        d,
        118,
        "GSC words stitched into robot commands",
        font(24),
        MUTED,
    )

    t = ease(progress)
    rows = [
        (
            "Long scripts + bag pool",
            "Fat DS-CNN owns fixed-canvas phrase ID\n84.5%  vs  Hybrid Transformer 50%",
            CNN,
        ),
        (
            "Order-only + no bag",
            "Hybrid Transformer owns the sequence\n96.5%  vs  causal CNN 27.8%",
            TR,
        ),
    ]
    for i, (head, body, accent) in enumerate(rows):
        y = int(lerp(1100, 220 + i * 320, ease(max(0.0, t * 1.25 - i * 0.18))))
        rounded(d, (70, y, 1010, y + 280), PANEL, 26)
        d.rounded_rectangle((70, y, 84, y + 280), radius=4, fill=accent)
        d.text((120, y + 48), head, font=font(34, True), fill=accent)
        d.text((120, y + 120), body, font=font(28), fill=TITLE)
    return im


def hold(frames: list[Image.Image], im: Image.Image, n: int = HOLD) -> None:
    for _ in range(n):
        frames.append(im.copy())


def crossfade(frames: list[Image.Image], a: Image.Image, b: Image.Image, n: int = FADE) -> None:
    for i in range(1, n + 1):
        frames.append(Image.blend(a, b, i / n))


def animate_scene(frames: list[Image.Image], scene_fn, build: int, settle: int) -> Image.Image:
    for i in range(build):
        frames.append(scene_fn((i + 1) / build))
    final = scene_fn(1.0)
    hold(frames, final, settle)
    return final


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []

    s0 = animate_scene(frames, scene_setup, HOLD + 6, HOLD)
    s1 = animate_scene(frames, scene_long_wins, HOLD + 8, HOLD + 6)
    crossfade(frames, s1, scene_why_bag(0.0))
    s2 = animate_scene(frames, scene_why_bag, HOLD + 6, HOLD + 4)
    crossfade(frames, s2, scene_fair_setup(0.0))
    s3 = animate_scene(frames, scene_fair_setup, HOLD + 6, HOLD + 4)
    crossfade(frames, s3, scene_fair_results(0.0))
    s4 = animate_scene(frames, scene_fair_results, HOLD + 6, HOLD + 10)
    crossfade(frames, s4, scene_takeaway(0.0))
    animate_scene(frames, scene_takeaway, HOLD + 8, HOLD + 12)
    _ = s0

    scaled = [fr.resize((720, 720), Image.Resampling.LANCZOS) for fr in frames]
    imageio.mimsave(GIF_PATH, scaled, fps=FPS, loop=0)
    dur = len(scaled) / FPS
    print(
        f"wrote {GIF_PATH}  frames={len(scaled)}  "
        f"~{dur:.1f}s  size={GIF_PATH.stat().st_size // 1024} KiB"
    )


if __name__ == "__main__":
    main()

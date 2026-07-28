#!/usr/bin/env python3
"""LinkedIn GIF: Hybrid Transformer on order-sensitive robot speech commands.

Meat only (no splash). No peer comparisons — HT story alone.
Dataset: Google Speech Commands words stitched into robotics phrases.

Usage:
  python3 viz/hybrid_transformer_anim.py
  → viz/out/hybrid_transformer.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1080
FPS = 6
HOLD = 32
FADE = 8

BG = (12, 18, 28)
PANEL = (20, 28, 42)
PANEL2 = (26, 36, 52)
TITLE = (245, 248, 252)
MUTED = (140, 156, 178)
MINT = (80, 200, 180)
AMBER = (240, 170, 70)
OK = (120, 210, 160)
TR = (200, 150, 255)

OUT_DIR = Path(__file__).resolve().parent / "out"
GIF_PATH = OUT_DIR / "hybrid_transformer.gif"

# Accuracy only (no host/board timing in the GIF)
ACC = 96.9


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
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


def new_frame():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def text_center(d, y, s, f, fill=TITLE):
    bbox = d.textbbox((0, 0), s, font=f)
    d.text(((W - (bbox[2] - bbox[0])) // 2, y), s, font=f, fill=fill)


def rounded(d, xy, fill, r=28):
    d.rounded_rectangle(xy, radius=r, fill=fill)


def load_accuracy():
    global ACC
    p = Path(__file__).resolve().parents[1] / "host" / "speech_peers_fair_board.json"
    if not p.is_file():
        p = Path(__file__).resolve().parents[1] / "host" / "speech_peers_metrics_fair.json"
    if not p.is_file():
        return
    import json

    d = json.loads(p.read_text())
    for key in ("torch_acc_export_subset", "cpu_i8_acc"):
        if d.get(key) is not None:
            ACC = 100.0 * float(d[key])
            break
    # fair metrics file nests peers
    for peer in d.get("peers", []):
        if "transformer" in str(peer.get("name", "")).lower():
            ACC = 100.0 * float(peer["accuracy"])
            break


def scene_dataset(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 48, "Hybrid Transformer", font(42, True), TR)
    text_center(d, 110, "robot command phrases from speech", font(26), MUTED)

    t = ease(progress)
    y = int(lerp(1100, 180, t))
    rounded(d, (70, y, 1010, y + 760), PANEL, 28)
    d.text((110, y + 36), "Dataset", font=font(32, True), fill=MINT)
    d.text(
        (110, y + 95),
        "Google Speech Commands words\n"
        "(go, up, down, left, right, stop, …)\n"
        "stitched into multi-word scripts",
        font=font(28),
        fill=TITLE,
    )
    d.text((110, y + 280), "Example spoken phrases", font=font(24, True), fill=MUTED)
    d.text(
        (110, y + 330),
        "go up  ·  go down  ·  go left\n"
        "go right · stop · go forward\n"
        "go up · go down · go left · stop\n"
        "go left · stop · go right · go forward",
        font=font(26),
        fill=TITLE,
    )
    d.text(
        (110, y + 580),
        "Same mel front-end every phrase:\n"
        "16 kHz · n_fft=512 · 50% hop · 32 mels",
        font=font(22),
        fill=MUTED,
    )
    return im


def scene_order(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 48, "Word order is the command", font(36, True))
    text_center(d, 110, "same words  ·  different robot action", font(24), MUTED)

    t = ease(progress)
    left = int(lerp(-500, 70, t))
    right = int(lerp(W + 40, 560, ease(max(0.0, t - 0.08))))

    # Minimal pair — path order matters
    rounded(d, (left, 200, left + 450, 640), PANEL, 26)
    d.text((left + 32, 230), "Command A", font=font(24), fill=MUTED)
    d.text((left + 32, 280), "raise, then lower", font=font(22, True), fill=MINT)
    d.text(
        (left + 32, 350),
        "go up\n"
        "go down\n"
        "go left\n"
        "stop",
        font=font(34, True),
        fill=TITLE,
    )

    rounded(d, (right, 200, right + 450, 640), PANEL, 26)
    d.text((right + 32, 230), "Command B", font=font(24), fill=MUTED)
    d.text((right + 32, 280), "lower, then raise", font=font(22, True), fill=AMBER)
    d.text(
        (right + 32, 350),
        "go down\n"
        "go up\n"
        "go left\n"
        "stop",
        font=font(34, True),
        fill=TITLE,
    )

    if t > 0.5:
        text_center(d, 720, "same bag of words  →  opposite motion sequence", font(26), AMBER)
        text_center(d, 800, "pooling away time cannot tell these apart", font(24), MUTED)
    return im


def _arrow_down(d, x: int, y: int, col=MUTED) -> None:
    d.polygon([(x - 14, y), (x + 14, y), (x, y + 22)], fill=col)


def scene_arch(progress: float) -> Image.Image:
    """High-level Hybrid Transformer block diagram (vertical flow)."""
    im, d = new_frame()
    text_center(d, 40, "Architecture", font(38, True))
    text_center(d, 96, "Hybrid Transformer  ·  keep the time axis", font(24), MUTED)

    t = ease(progress)
    blocks = [
        ("Stitched waveform", "spoken words concatenated → ~10 s clip", (100, 160, 190)),
        ("Log-mel front-end", "32 mels × 1023 frames · 50% hop", MINT),
        ("Hybrid DS-CNN stem", "strided DS blocks → T=128 × D=32 tokens", MINT),
        ("Transformer body ×6", "self-attention + FFN · D=32 · MLP=64", TR),
        ("Last-token head", "order-aware command classification", AMBER),
    ]

    y0 = 150
    box_h = 110
    gap = 28
    for i, (title, sub, accent) in enumerate(blocks):
        appear = ease(max(0.0, min(1.0, t * 1.35 - i * 0.12)))
        y = int(lerp(1100, y0 + i * (box_h + gap), appear))
        x0, x1 = 100, 980
        rounded(d, (x0, y, x1, y + box_h), PANEL, 20)
        d.rounded_rectangle((x0, y, x0 + 14, y + box_h), radius=4, fill=accent)
        d.text((x0 + 40, y + 22), title, font=font(28, True), fill=TITLE)
        d.text((x0 + 40, y + 62), sub, font=font(22), fill=MUTED)
        if i < len(blocks) - 1 and appear > 0.55:
            _arrow_down(d, W // 2, y + box_h + 2, MUTED)
    return im


def scene_results(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 48, "Results", font(42, True))
    text_center(d, 110, "order-only robotics phrases", font(24), MUTED)

    t = ease(progress)
    y = int(lerp(1100, 220, t))
    rounded(d, (90, y, 990, y + 620), PANEL, 28)
    d.text((130, y + 48), "Hybrid Transformer", font=font(32, True), fill=TR)
    d.text((130, y + 140), f"{ACC:.1f}%", font=font(96, True), fill=OK)
    d.text((130, y + 280), "command accuracy", font=font(28), fill=MUTED)

    d.text(
        (130, y + 380),
        "Geometry: T=128 · D=32 · L=6\n"
        "Task: same words, 16 order permutations\n"
        "Source: Google Speech Commands → stitched phrases",
        font=font(24),
        fill=TITLE,
    )
    return im


def scene_takeaway(progress: float) -> Image.Image:
    im, d = new_frame()
    text_center(d, 56, "Takeaway", font(40, True))

    t = ease(progress)
    y = int(lerp(1100, 200, t))
    rounded(d, (70, y, 1010, y + 680), PANEL2, 28)
    d.text((120, y + 50), "For multi-word robot speech…", font=font(30, True), fill=MINT)
    d.text(
        (120, y + 140),
        "Keep the time axis.\n"
        "Don’t globally pool the phrase\n"
        "into a bag of sounds.\n\n"
        "A Hybrid Transformer can learn\n"
        "that word order is the command.",
        font=font(32),
        fill=TITLE,
    )
    d.text(
        (120, y + 520),
        f"Order-only robotics phrases → {ACC:.1f}%",
        font=font(26, True),
        fill=OK,
    )
    return im


def hold(frames, im, n=HOLD):
    for _ in range(n):
        frames.append(im.copy())


def crossfade(frames, a, b, n=FADE):
    for i in range(1, n + 1):
        frames.append(Image.blend(a, b, i / n))


def animate_scene(frames, scene_fn, build, settle):
    for i in range(build):
        frames.append(scene_fn((i + 1) / build))
    final = scene_fn(1.0)
    hold(frames, final, settle)
    return final


def main() -> None:
    load_accuracy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []

    s0 = animate_scene(frames, scene_dataset, HOLD + 6, HOLD + 4)
    crossfade(frames, s0, scene_order(0.0))
    s1 = animate_scene(frames, scene_order, HOLD + 6, HOLD + 6)
    crossfade(frames, s1, scene_arch(0.0))
    s2 = animate_scene(frames, scene_arch, HOLD + 8, HOLD + 8)
    crossfade(frames, s2, scene_results(0.0))
    s3 = animate_scene(frames, scene_results, HOLD + 6, HOLD + 10)
    crossfade(frames, s3, scene_takeaway(0.0))
    animate_scene(frames, scene_takeaway, HOLD + 6, HOLD + 12)

    scaled = [fr.resize((720, 720), Image.Resampling.LANCZOS) for fr in frames]
    imageio.mimsave(GIF_PATH, scaled, fps=FPS, loop=0)
    print(f"wrote {GIF_PATH}  frames={len(scaled)}  ~{len(scaled)/FPS:.1f}s  acc={ACC:.1f}%")


if __name__ == "__main__":
    main()

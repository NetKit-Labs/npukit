#!/usr/bin/env python3
"""Polished LinkedIn GIF: 8x8 output-stationary systolic array.

Matches the skewed feed schedule in sim/systolic_array_tb.sv (3N-2 cycles).

Usage:
  python3 viz/systolic_anim.py
  → viz/out/systolic_8x8.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 8
CYCLES = 3 * N - 2  # 22

# Canvas — square works well in LinkedIn feed
W, H = 1080, 1080
FPS = 12

# Palette: deep slate + mint/amber (avoid purple-glow cliché)
BG = (12, 18, 28)
PANEL = (20, 28, 42)
GRID_LINE = (48, 62, 84)
PE_IDLE = (28, 38, 56)
PE_ACTIVE = (36, 72, 78)
ACC_TEXT = (230, 236, 244)
MUTED = (140, 156, 178)
A_COLOR = (80, 200, 180)      # mint — A west→east
B_COLOR = (240, 170, 70)      # amber — B north→south
MAC_GLOW = (255, 220, 140)
TITLE = (245, 248, 252)
ACCENT = (80, 200, 180)
C_DONE = (120, 210, 160)

OUT_DIR = Path(__file__).resolve().parent / "out"
GIF_PATH = OUT_DIR / "systolic_8x8.gif"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def matrices():
    """Same stimulus as systolic_array_tb: A[i][k]=i+1, B[k][j]=1 → C[i][j]=N*(i+1)."""
    A = np.zeros((N, N), dtype=np.int32)
    B = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            A[i, j] = i + 1
            B[i, j] = 1
    C_ref = A @ B
    return A, B, C_ref


def west_north(A, B, t: int):
    a_west = np.zeros(N, dtype=np.int32)
    b_north = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if t >= i and (t - i) < N:
            a_west[i] = A[i, t - i]
            b_north[i] = B[t - i, i]
    return a_west, b_north


class ArraySim:
    """Cycle-accurate model of rtl/pe.sv + rtl/systolic_array.sv."""

    def __init__(self):
        self.acc = np.zeros((N, N), dtype=np.int32)
        self.a_pipe = np.zeros((N, N), dtype=np.int32)
        self.b_pipe = np.zeros((N, N), dtype=np.int32)
        self.active = np.zeros((N, N), dtype=bool)
        self.last_a_in = np.zeros((N, N), dtype=np.int32)
        self.last_b_in = np.zeros((N, N), dtype=np.int32)

    def step(self, a_west, b_north, enable: bool = True):
        a_in = np.zeros((N, N), dtype=np.int32)
        b_in = np.zeros((N, N), dtype=np.int32)
        for r in range(N):
            for c in range(N):
                a_in[r, c] = a_west[r] if c == 0 else self.a_pipe[r, c - 1]
                b_in[r, c] = b_north[c] if r == 0 else self.b_pipe[r - 1, c]

        self.active[:] = False
        self.last_a_in = a_in.copy()
        self.last_b_in = b_in.copy()
        if not enable:
            return

        new_a = np.zeros_like(self.a_pipe)
        new_b = np.zeros_like(self.b_pipe)
        for r in range(N):
            for c in range(N):
                prod = int(a_in[r, c]) * int(b_in[r, c])
                if a_in[r, c] != 0 or b_in[r, c] != 0:
                    self.active[r, c] = True
                self.acc[r, c] += prod
                new_a[r, c] = a_in[r, c]
                new_b[r, c] = b_in[r, c]
        self.a_pipe = new_a
        self.b_pipe = new_b


def lerp_color(c0, c1, u: float):
    u = max(0.0, min(1.0, u))
    return tuple(int(c0[i] + (c1[i] - c0[i]) * u) for i in range(3))


def draw_round_rect(draw, xy, radius, fill, outline=None, width=1):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


class Scene:
    def __init__(self):
        self.A, self.B, self.C_ref = matrices()
        self.f_title = font(42, bold=True)
        self.f_sub = font(22, bold=False)
        self.f_label = font(18, bold=True)
        self.f_small = font(15, bold=False)
        self.f_cell = font(16, bold=True)
        self.f_tiny = font(13, bold=False)
        self.f_big = font(28, bold=True)

        # Grid geometry
        self.margin_l = 120
        self.margin_t = 210
        self.cell = 78
        self.gap = 8
        grid_w = N * self.cell + (N - 1) * self.gap
        self.origin_x = (W - grid_w) // 2 + 20
        self.origin_y = self.margin_t

    def pe_box(self, r, c):
        x0 = self.origin_x + c * (self.cell + self.gap)
        y0 = self.origin_y + r * (self.cell + self.gap)
        return x0, y0, x0 + self.cell, y0 + self.cell

    def blank(self):
        img = Image.new("RGB", (W, H), BG)
        return img, ImageDraw.Draw(img)

    def header(self, draw, cycle: int | None = None, phase: str = ""):
        draw.text((56, 48), "NpuKit", font=self.f_title, fill=ACCENT)
        draw.text((56, 100), "8×8 output-stationary systolic array", font=self.f_sub, fill=TITLE)
        draw.text(
            (56, 134),
            "A → east   ·   B ↓ south   ·   C stays in each PE",
            font=self.f_small,
            fill=MUTED,
        )
        if cycle is not None:
            draw_round_rect(draw, (W - 280, 52, W - 56, 118), 14, PANEL, GRID_LINE, 2)
            draw.text((W - 260, 62), f"cycle  {cycle + 1} of {CYCLES}", font=self.f_label, fill=TITLE)
            draw.text((W - 260, 88), phase, font=self.f_tiny, fill=MUTED)

    def legend(self, draw):
        y = H - 86
        draw_round_rect(draw, (56, y - 10, W - 56, H - 40), 16, PANEL, GRID_LINE, 2)
        draw.ellipse((80, y + 8, 98, y + 26), fill=A_COLOR)
        draw.text((110, y + 6), "A stream (west → east)", font=self.f_small, fill=A_COLOR)
        draw.ellipse((420, y + 8, 438, y + 26), fill=B_COLOR)
        draw.text((450, y + 6), "B stream (north → south)", font=self.f_small, fill=B_COLOR)
        draw.ellipse((760, y + 8, 778, y + 26), fill=C_DONE)
        draw.text((790, y + 6), "C accumulator", font=self.f_small, fill=C_DONE)

    def draw_flow_arrows(self, draw):
        # Left A label
        ax = self.origin_x - 70
        ay = self.origin_y + (N * (self.cell + self.gap) - self.gap) / 2
        draw.text((ax - 10, ay - 40), "A", font=self.f_big, fill=A_COLOR)
        draw.polygon(
            [(ax + 28, ay - 8), (ax + 52, ay), (ax + 28, ay + 8)],
            fill=A_COLOR,
        )
        # Top B label
        bx = self.origin_x + (N * (self.cell + self.gap) - self.gap) / 2
        by = self.origin_y - 58
        draw.text((bx - 10, by - 28), "B", font=self.f_big, fill=B_COLOR)
        draw.polygon(
            [(bx - 8, by + 8), (bx, by + 32), (bx + 8, by + 8)],
            fill=B_COLOR,
        )

    def draw_inputs(self, draw, a_west, b_north, pulse: float):
        for i in range(N):
            x0, y0, x1, y1 = self.pe_box(i, 0)
            # A pills on left
            px = x0 - 54
            py = (y0 + y1) // 2
            val = int(a_west[i])
            alpha = 0.35 + 0.65 * pulse if val else 0.25
            col = lerp_color(BG, A_COLOR, alpha)
            draw_round_rect(draw, (px - 18, py - 16, px + 18, py + 16), 8, col)
            if val:
                draw.text((px - 6, py - 9), str(val), font=self.f_tiny, fill=TITLE)

            x0, y0, x1, y1 = self.pe_box(0, i)
            px = (x0 + x1) // 2
            py = y0 - 36
            val = int(b_north[i])
            alpha = 0.35 + 0.65 * pulse if val else 0.25
            col = lerp_color(BG, B_COLOR, alpha)
            draw_round_rect(draw, (px - 18, py - 16, px + 18, py + 16), 8, col)
            if val:
                draw.text((px - 4, py - 9), str(val), font=self.f_tiny, fill=BG if val else MUTED)

    def draw_grid(self, draw, sim: ArraySim, pulse: float, show_final: bool = False):
        for r in range(N):
            for c in range(N):
                box = self.pe_box(r, c)
                active = bool(sim.active[r, c]) and not show_final
                if show_final:
                    fill = lerp_color(PE_IDLE, C_DONE, 0.35)
                    outline = C_DONE
                elif active:
                    fill = lerp_color(PE_IDLE, PE_ACTIVE, 0.55 + 0.45 * pulse)
                    outline = lerp_color(GRID_LINE, MAC_GLOW, pulse)
                else:
                    fill = PE_IDLE
                    outline = GRID_LINE
                draw_round_rect(draw, box, 12, fill, outline, 2 if active or show_final else 1)

                acc = int(sim.acc[r, c])
                label = str(acc)
                # Center accumulator
                tw = draw.textlength(label, font=self.f_cell)
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                color = TITLE if active or show_final or acc else MUTED
                draw.text((cx - tw / 2, cy - 10), label, font=self.f_cell, fill=color)

                if active and pulse > 0.4:
                    a_v = int(sim.last_a_in[r, c])
                    b_v = int(sim.last_b_in[r, c])
                    mac = f"{a_v}×{b_v}"
                    tw2 = draw.textlength(mac, font=self.f_tiny)
                    draw.text((cx - tw2 / 2, box[3] - 20), mac, font=self.f_tiny, fill=MAC_GLOW)

    def formula_panel(self, draw):
        draw_round_rect(draw, (56, H - 170, W - 56, H - 100), 14, PANEL, GRID_LINE, 2)
        draw.text(
            (80, H - 155),
            "C[i][j]  +=  A[i][k] · B[k][j]     each PE, each cycle     ·     finish in 3N−2 = 22 clocks",
            font=self.f_small,
            fill=MUTED,
        )

    def render_title(self, u: float) -> Image.Image:
        img, draw = self.blank()
        fade = ease_out_cubic(u)
        # Soft vignette bar
        draw_round_rect(draw, (80, 360, W - 80, 720), 28, lerp_color(BG, PANEL, fade), GRID_LINE, 2)
        draw.text((120, 410), "NpuKit", font=font(56, True), fill=lerp_color(BG, ACCENT, fade))
        draw.text(
            (120, 490),
            "Watch an 8×8 systolic array",
            font=font(34, True),
            fill=lerp_color(BG, TITLE, fade),
        )
        draw.text(
            (120, 545),
            "multiply two int8 matrices in 22 clock cycles",
            font=self.f_sub,
            fill=lerp_color(BG, MUTED, fade),
        )
        draw.text(
            (120, 620),
            "output-stationary  ·  A →  ·  B ↓  ·  C in each PE",
            font=self.f_small,
            fill=lerp_color(BG, A_COLOR, fade * 0.9),
        )
        return img

    def render_cycle(self, sim: ArraySim, a_west, b_north, cycle: int, pulse: float) -> Image.Image:
        img, draw = self.blank()
        self.header(draw, cycle, "streaming MACs")
        self.draw_flow_arrows(draw)
        self.draw_inputs(draw, a_west, b_north, pulse)
        self.draw_grid(draw, sim, pulse)
        self.formula_panel(draw)
        self.legend(draw)
        return img

    def render_final(self, sim: ArraySim, u: float) -> Image.Image:
        img, draw = self.blank()
        self.header(draw, CYCLES - 1, "C complete")
        self.draw_flow_arrows(draw)
        self.draw_grid(draw, sim, 1.0, show_final=True)
        # Result callout
        draw_round_rect(draw, (56, H - 170, W - 56, H - 40), 16, PANEL, C_DONE, 2)
        draw.text(
            (80, H - 145),
            "Done — each of 64 PE accumulators holds one C[i][j]",
            font=self.f_label,
            fill=TITLE,
        )
        draw.text(
            (80, H - 110),
            f"Example: A[i][k]=i+1, B=ones  →  C[i][j]={N}·(i+1)   (matches sim/systolic_array_tb)",
            font=self.f_small,
            fill=MUTED,
        )
        # Soft fade-in overlay feel via alpha on a bar
        if u < 1:
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            alpha = int(180 * (1 - ease_out_cubic(u)))
            od.rectangle((0, 0, W, H), fill=(12, 18, 28, alpha))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        return img


def hold(frames, img, n):
    for _ in range(n):
        frames.append(np.asarray(img))


def build_gif():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    frames: list[np.ndarray] = []

    sim = ArraySim()
    A, B, _ = scene.A, scene.B, scene.C_ref

    for t in range(CYCLES):
        a_west, b_north = west_north(A, B, t)

        # Advance one cycle into a temp state, then animate old → new acc
        pre_acc = sim.acc.copy()
        tmp = ArraySim()
        tmp.acc = sim.acc.copy()
        tmp.a_pipe = sim.a_pipe.copy()
        tmp.b_pipe = sim.b_pipe.copy()
        tmp.step(a_west, b_north, enable=True)

        for s in range(6):
            pulse = ease_out_cubic(s / 5)
            view = ArraySim()
            view.acc = pre_acc if s < 3 else tmp.acc
            view.active = tmp.active
            view.last_a_in = tmp.last_a_in
            view.last_b_in = tmp.last_b_in
            frames.append(np.asarray(scene.render_cycle(view, a_west, b_north, t, pulse)))

        sim.acc = tmp.acc
        sim.a_pipe = tmp.a_pipe
        sim.b_pipe = tmp.b_pipe
        sim.active = tmp.active
        sim.last_a_in = tmp.last_a_in
        sim.last_b_in = tmp.last_b_in

    # Settle
    settle = ArraySim()
    settle.acc = sim.acc.copy()
    for i in range(16):
        frames.append(np.asarray(scene.render_final(settle, i / 15)))
    hold(frames, scene.render_final(settle, 1.0), 18)

    # Verify against reference
    assert np.array_equal(sim.acc, scene.C_ref), (sim.acc, scene.C_ref)

    print(f"Writing {len(frames)} frames → {GIF_PATH}")
    imageio.mimsave(
        GIF_PATH,
        frames,
        fps=FPS,
        loop=0,
        palettesize=256,
    )
    size_mb = GIF_PATH.stat().st_size / (1024 * 1024)
    print(f"Done: {GIF_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_gif()

#!/usr/bin/env python3
"""Multi-page PDF: 8×8 output-stationary systolic array, cycle by cycle.

Same stimulus as sim/systolic_array_tb.sv:
  A[i][k] = i+1,  B[k][j] = 1  →  C[i][j] = N*(i+1)

Each cycle page shows A and B values in every PE as they flow through,
plus the running accumulator (C) in the same cell.

Usage:
  viz/.venv/bin/python viz/systolic_pdf.py
  → viz/out/systolic_8x8_timesteps.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 8
CYCLES = 3 * N - 2  # 22

# Landscape letter-ish canvas (points-ish pixels at 150 dpi feel)
W, H = 1600, 1100

BG = (250, 251, 252)
INK = (22, 28, 36)
MUTED = (100, 112, 128)
PANEL = (236, 240, 245)
GRID = (180, 190, 204)
PE_IDLE = (255, 255, 255)
PE_ACTIVE = (220, 242, 236)
A_COLOR = (16, 130, 118)
B_COLOR = (180, 100, 20)
C_COLOR = (28, 110, 72)
ACCENT = (16, 130, 118)
DONE_FILL = (228, 244, 234)
DONE_EDGE = (72, 160, 120)

OUT_DIR = Path(__file__).resolve().parent / "out"
PDF_PATH = OUT_DIR / "systolic_8x8_timesteps.pdf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def matrices():
    """Same stimulus as systolic_array_tb: A[i][k]=i+1, B=ones → C=N*(i+1)."""
    A = np.zeros((N, N), dtype=np.int32)
    B = np.ones((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            A[i, j] = i + 1
    return A, B, A @ B


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
        self.last_a = np.zeros((N, N), dtype=np.int32)
        self.last_b = np.zeros((N, N), dtype=np.int32)

    def step(self, a_west, b_north):
        a_in = np.zeros((N, N), dtype=np.int32)
        b_in = np.zeros((N, N), dtype=np.int32)
        for r in range(N):
            for c in range(N):
                a_in[r, c] = a_west[r] if c == 0 else self.a_pipe[r, c - 1]
                b_in[r, c] = b_north[c] if r == 0 else self.b_pipe[r - 1, c]

        self.active[:] = False
        self.last_a = a_in.copy()
        self.last_b = b_in.copy()
        new_a = np.zeros_like(self.a_pipe)
        new_b = np.zeros_like(self.b_pipe)
        for r in range(N):
            for c in range(N):
                if a_in[r, c] != 0 or b_in[r, c] != 0:
                    self.active[r, c] = True
                self.acc[r, c] += int(a_in[r, c]) * int(b_in[r, c])
                new_a[r, c] = a_in[r, c]
                new_b[r, c] = b_in[r, c]
        self.a_pipe = new_a
        self.b_pipe = new_b


class PdfScene:
    def __init__(self):
        self.A, self.B, self.C_ref = matrices()
        self.f_title = font(36, bold=True)
        self.f_h2 = font(24, bold=True)
        self.f_body = font(16, bold=False)
        self.f_small = font(13, bold=False)
        self.f_tiny = font(11, bold=False)
        self.f_cell_ab = font(12, bold=True)
        self.f_cell_acc = font(15, bold=True)
        self.f_matrix = font(14, bold=True)
        self.f_label = font(18, bold=True)

        self.cell = 88
        self.gap = 6
        grid_w = N * self.cell + (N - 1) * self.gap
        grid_h = grid_w
        self.origin_x = (W - grid_w) // 2 + 40
        self.origin_y = 210

    def pe_box(self, r, c):
        x0 = self.origin_x + c * (self.cell + self.gap)
        y0 = self.origin_y + r * (self.cell + self.gap)
        return x0, y0, x0 + self.cell, y0 + self.cell

    def blank(self):
        img = Image.new("RGB", (W, H), BG)
        return img, ImageDraw.Draw(img)

    def footer(self, draw, page: str):
        draw.text((48, H - 36), "NpuKit · output-stationary systolic array · A→east  B↓south  C stays", font=self.f_tiny, fill=MUTED)
        tw = draw.textlength(page, font=self.f_tiny)
        draw.text((W - 48 - tw, H - 36), page, font=self.f_tiny, fill=MUTED)

    def draw_matrix(self, draw, M, x, y, title, title_color, cell=36):
        draw.text((x, y), title, font=self.f_label, fill=title_color)
        y0 = y + 28
        for r in range(N):
            for c in range(N):
                bx = x + c * (cell + 3)
                by = y0 + r * (cell + 3)
                draw.rounded_rectangle((bx, by, bx + cell, by + cell), radius=6, fill=PE_IDLE, outline=GRID, width=1)
                label = str(int(M[r, c]))
                tw = draw.textlength(label, font=self.f_matrix)
                draw.text((bx + (cell - tw) / 2, by + 9), label, font=self.f_matrix, fill=INK)

    def render_intro(self) -> Image.Image:
        img, draw = self.blank()
        draw.text((48, 40), "NpuKit", font=self.f_title, fill=ACCENT)
        draw.text((48, 90), "8×8 systolic array — cycle-by-cycle walkthrough", font=self.f_h2, fill=INK)
        draw.text(
            (48, 130),
            "Same stimulus as sim/systolic_array_tb.sv · skewed feed · finish in 3N−2 = 22 clocks",
            font=self.f_body,
            fill=MUTED,
        )
        draw.text(
            (48, 165),
            "Each PE keeps a running C[i][j] accumulator. A streams west→east; B streams north→south.",
            font=self.f_body,
            fill=MUTED,
        )

        self.draw_matrix(draw, self.A, 80, 230, "A  (A[i][k] = i+1)", A_COLOR)
        self.draw_matrix(draw, self.B, 560, 230, "B  (all ones)", B_COLOR)
        self.draw_matrix(draw, self.C_ref, 1040, 230, "C = A·B  (expected)", C_COLOR)

        draw.rounded_rectangle((48, 620, W - 48, 980), radius=16, fill=PANEL, outline=GRID, width=1)
        lines = [
            "How to read the following pages",
            "",
            "• One page per clock cycle (t = 0 … 21).",
            "• Left pills = A values entering from the west; top pills = B values entering from the north.",
            "• Inside each PE cell:",
            "      A×B   — operands present this cycle (blank if idle / zeros)",
            "      Σ     — running accumulator after this cycle’s MAC  (this is C[i][j])",
            "• Active cells (doing a multiply-accumulate) are tinted mint.",
            "• Final page shows C held in the 64 PE accumulators.",
            "",
            f"Expected result: C[i][j] = {N}·(i+1)  →  row 0 = 8, row 1 = 16, …, row 7 = 64",
        ]
        y = 645
        for i, line in enumerate(lines):
            f = self.f_label if i == 0 else self.f_body
            col = INK if i == 0 else MUTED if line.startswith("•") or line.startswith(" ") or line.startswith("Expected") or line == "" else INK
            if line.startswith("•") or line.startswith(" ") or line.startswith("Expected"):
                col = MUTED if not line.startswith("Expected") else C_COLOR
            if i == 0:
                col = INK
            draw.text((80, y), line, font=f, fill=col if line else MUTED)
            y += 28 if i == 0 else 26

        self.footer(draw, "intro")
        return img

    def draw_feeds(self, draw, a_west, b_north):
        for i in range(N):
            x0, y0, x1, y1 = self.pe_box(i, 0)
            px = x0 - 52
            py = (y0 + y1) // 2
            val = int(a_west[i])
            fill = (200, 232, 224) if val else PANEL
            draw.rounded_rectangle((px - 22, py - 18, px + 22, py + 18), radius=8, fill=fill, outline=A_COLOR if val else GRID, width=1)
            label = str(val) if val else "·"
            tw = draw.textlength(label, font=self.f_cell_ab)
            draw.text((px - tw / 2, py - 8), label, font=self.f_cell_ab, fill=A_COLOR if val else MUTED)

            x0, y0, x1, y1 = self.pe_box(0, i)
            px = (x0 + x1) // 2
            py = y0 - 40
            val = int(b_north[i])
            fill = (255, 230, 200) if val else PANEL
            draw.rounded_rectangle((px - 22, py - 18, px + 22, py + 18), radius=8, fill=fill, outline=B_COLOR if val else GRID, width=1)
            label = str(val) if val else "·"
            tw = draw.textlength(label, font=self.f_cell_ab)
            draw.text((px - tw / 2, py - 8), label, font=self.f_cell_ab, fill=B_COLOR if val else MUTED)

        # Axis labels
        ax = self.origin_x - 100
        ay = self.origin_y + (N * (self.cell + self.gap) - self.gap) / 2
        draw.text((ax, ay - 20), "A", font=self.f_h2, fill=A_COLOR)
        draw.text((ax + 28, ay - 12), "→", font=self.f_label, fill=A_COLOR)

        bx = self.origin_x + (N * (self.cell + self.gap) - self.gap) / 2 - 10
        by = self.origin_y - 95
        draw.text((bx, by), "B", font=self.f_h2, fill=B_COLOR)
        draw.text((bx + 4, by + 28), "↓", font=self.f_label, fill=B_COLOR)

    def draw_grid(self, draw, sim: ArraySim, show_final: bool = False):
        for r in range(N):
            for c in range(N):
                box = self.pe_box(r, c)
                active = bool(sim.active[r, c]) and not show_final
                a_v = int(sim.last_a[r, c])
                b_v = int(sim.last_b[r, c])
                acc = int(sim.acc[r, c])

                if show_final:
                    fill, outline, ow = DONE_FILL, DONE_EDGE, 2
                elif active:
                    fill, outline, ow = PE_ACTIVE, A_COLOR, 2
                else:
                    fill, outline, ow = PE_IDLE, GRID, 1

                draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=ow)

                cx = (box[0] + box[2]) / 2
                if show_final:
                    label = str(acc)
                    tw = draw.textlength(label, font=self.f_cell_acc)
                    draw.text((cx - tw / 2, box[1] + 32), label, font=self.f_cell_acc, fill=C_COLOR)
                    continue

                # A×B operands on top line
                if active and (a_v != 0 or b_v != 0):
                    ab = f"{a_v}×{b_v}"
                    tw = draw.textlength(ab, font=self.f_cell_ab)
                    draw.text((cx - tw / 2, box[1] + 12), ab, font=self.f_cell_ab, fill=B_COLOR if b_v else A_COLOR)
                else:
                    ab = "—"
                    tw = draw.textlength(ab, font=self.f_tiny)
                    draw.text((cx - tw / 2, box[1] + 14), ab, font=self.f_tiny, fill=MUTED)

                # Running accumulator (C) on bottom line
                sigma = f"Σ {acc}"
                tw = draw.textlength(sigma, font=self.f_cell_acc)
                col = C_COLOR if acc or active else MUTED
                draw.text((cx - tw / 2, box[1] + 48), sigma, font=self.f_cell_acc, fill=col)

    def legend_bar(self, draw, y: int):
        draw.rounded_rectangle((48, y, W - 48, y + 52), radius=10, fill=PANEL, outline=GRID, width=1)
        draw.text((70, y + 16), "cell:  A×B  (operands this cycle)", font=self.f_small, fill=B_COLOR)
        draw.text((420, y + 16), "Σ  = running C[i][j] accumulator", font=self.f_small, fill=C_COLOR)
        draw.text((820, y + 16), "mint cell = MAC this cycle", font=self.f_small, fill=A_COLOR)
        draw.text((1180, y + 16), "C stays in PE", font=self.f_small, fill=MUTED)

    def render_cycle(self, sim: ArraySim, a_west, b_north, t: int) -> Image.Image:
        img, draw = self.blank()
        draw.text((48, 36), "NpuKit", font=self.f_title, fill=ACCENT)
        draw.text((48, 82), f"Cycle  {t + 1}  of  {CYCLES}   (t = {t})", font=self.f_h2, fill=INK)
        draw.text(
            (48, 118),
            "A and B flowing through each PE · Σ is the running accumulator (becomes C)",
            font=self.f_body,
            fill=MUTED,
        )

        # Cycle badge
        draw.rounded_rectangle((W - 280, 40, W - 48, 110), radius=12, fill=PANEL, outline=GRID, width=1)
        draw.text((W - 255, 52), f"clock  {t + 1}/{CYCLES}", font=self.f_label, fill=INK)
        draw.text((W - 255, 80), "streaming MACs", font=self.f_small, fill=MUTED)

        self.draw_feeds(draw, a_west, b_north)
        self.draw_grid(draw, sim, show_final=False)
        self.legend_bar(draw, H - 110)
        self.footer(draw, f"cycle {t + 1}/{CYCLES}")
        return img

    def render_final(self, sim: ArraySim) -> Image.Image:
        img, draw = self.blank()
        draw.text((48, 36), "NpuKit", font=self.f_title, fill=ACCENT)
        draw.text((48, 82), "Done — C is the PE accumulators", font=self.f_h2, fill=INK)
        draw.text(
            (48, 118),
            f"After {CYCLES} clocks each PE holds one C[i][j].  A[i][k]=i+1, B=ones → C[i][j]={N}·(i+1)",
            font=self.f_body,
            fill=MUTED,
        )

        draw.rounded_rectangle((W - 280, 40, W - 48, 110), radius=12, fill=DONE_FILL, outline=DONE_EDGE, width=2)
        draw.text((W - 255, 52), f"clock  {CYCLES}/{CYCLES}", font=self.f_label, fill=C_COLOR)
        draw.text((W - 255, 80), "C complete", font=self.f_small, fill=C_COLOR)

        # Shift grid up a bit / reuse geometry; label as C
        bx = self.origin_x + (N * (self.cell + self.gap) - self.gap) / 2 - 10
        by = self.origin_y - 70
        draw.text((bx - 10, by), "C", font=self.f_h2, fill=C_COLOR)

        self.draw_grid(draw, sim, show_final=True)

        draw.rounded_rectangle((48, H - 120, W - 48, H - 48), radius=12, fill=DONE_FILL, outline=DONE_EDGE, width=2)
        draw.text(
            (70, H - 100),
            "Result verified against A @ B  ·  matches sim/systolic_array_tb.sv",
            font=self.f_body,
            fill=C_COLOR,
        )
        self.footer(draw, "final · C")
        return img


def build_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = PdfScene()
    pages: list[Image.Image] = [scene.render_intro()]

    sim = ArraySim()
    A, B = scene.A, scene.B

    for t in range(CYCLES):
        a_west, b_north = west_north(A, B, t)
        sim.step(a_west, b_north)
        pages.append(scene.render_cycle(sim, a_west, b_north, t))

    assert np.array_equal(sim.acc, scene.C_ref), (sim.acc, scene.C_ref)
    pages.append(scene.render_final(sim))

    first, rest = pages[0], pages[1:]
    first.save(
        PDF_PATH,
        "PDF",
        resolution=120.0,
        save_all=True,
        append_images=rest,
    )
    size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(pages)} pages → {PDF_PATH} ({size_mb:.2f} MB)")
    print(f"Verified C == A @ B: row0={sim.acc[0,0]}, row7={sim.acc[7,0]}")


if __name__ == "__main__":
    build_pdf()

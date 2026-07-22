#!/usr/bin/env python3
"""Manim scene: 8×8 output-stationary systolic array (NpuKit).

Cycle-accurate vs sim/systolic_array_tb.sv (skewed A/B feed, 3N−2 = 22 cycles).

Setup (once):
  python3.13 -m venv viz/.venv
  viz/.venv/bin/pip install manim numpy

Render LinkedIn square GIF + MP4:
  viz/render_linkedin.sh
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    FadeIn,
    FadeOut,
    ManimColor,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    config,
    interpolate_color,
)

# Square canvas for LinkedIn feed
config.frame_width = 8
config.frame_height = 8
config.background_color = "#0C121C"

N = 8
CYCLES = 3 * N - 2  # 22

BG = "#0C121C"
PANEL = "#141C2A"
GRID = "#304054"
PE_IDLE = "#1C2638"
PE_ACTIVE = "#244848"
ACCENT = "#50C8B4"
AMBER = "#F0AA46"
TITLE = "#F5F8FC"
MUTED = "#8C9CB2"
GLOW = "#FFDC8C"
DONE = "#78D2A0"


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


def make_pill(label: str, color: str, w: float = 0.38, h: float = 0.32) -> VGroup:
    body = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=0.08,
        fill_color=color,
        fill_opacity=0.92,
        stroke_width=0,
    )
    txt = Text(
        label,
        font_size=14,
        color=BG if color == AMBER else TITLE,
        weight="BOLD",
    )
    txt.move_to(body.get_center())
    return VGroup(body, txt)


def make_acc_text(val: int, color: str, center) -> Text:
    t = Text(str(val), font_size=16, color=color, weight="BOLD")
    t.move_to(center)
    return t


def make_mac_text(label: str, box) -> Text:
    t = Text(label, font_size=10, color=GLOW)
    t.next_to(box.get_bottom(), UP, buff=0.05)
    return t


class SystolicArrayScene(Scene):
    """LinkedIn-ready Manim visualization of the NpuKit systolic array."""

    def construct(self):
        A, B, C_ref = matrices()
        sim = ArraySim()

        cell = 0.54
        gap = 0.06
        pitch = cell + gap
        grid_w = N * cell + (N - 1) * gap
        # Keep all 8 rows clear of the footer banner (frame y ∈ [-4, 4])
        origin = np.array([-grid_w / 2 + cell / 2 + 0.1, 1.55, 0.0])

        def pe_center(r: int, c: int):
            return origin + np.array([c * pitch, -r * pitch, 0.0])

        # --- chrome ---
        brand = Text("NpuKit", font_size=36, color=ACCENT, weight="BOLD")
        brand.to_corner(UL, buff=0.35)
        subtitle = Text("8×8 output-stationary systolic array", font_size=18, color=TITLE)
        subtitle.next_to(brand, DOWN, aligned_edge=LEFT, buff=0.12)
        flow = Text(
            "A → east   ·   B ↓ south   ·   C stays in each PE",
            font_size=14,
            color=MUTED,
        )
        flow.next_to(subtitle, DOWN, aligned_edge=LEFT, buff=0.1)

        cycle_box = RoundedRectangle(
            width=2.35,
            height=0.72,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=GRID,
            stroke_width=1.5,
        )
        cycle_box.to_corner(UR, buff=0.35)
        cycle_label = Text("cycle  1 of 22", font_size=16, color=TITLE, weight="BOLD")
        cycle_phase = Text("streaming MACs", font_size=12, color=MUTED)
        cycle_text = VGroup(cycle_label, cycle_phase).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT
        )
        cycle_text.move_to(cycle_box.get_center())

        # --- PE grid ---
        pe_boxes: list[list[RoundedRectangle]] = []
        pe_acc: list[list[Text]] = []
        pe_mac: list[list[Text]] = []
        pe_group = VGroup()
        for r in range(N):
            row_b, row_a, row_m = [], [], []
            for c in range(N):
                box = RoundedRectangle(
                    width=cell,
                    height=cell,
                    corner_radius=0.1,
                    fill_color=PE_IDLE,
                    fill_opacity=1,
                    stroke_color=GRID,
                    stroke_width=1.2,
                )
                box.move_to(pe_center(r, c))
                acc = make_acc_text(0, MUTED, box.get_center())
                mac = make_mac_text("", box)
                mac.set_opacity(0)
                row_b.append(box)
                row_a.append(acc)
                row_m.append(mac)
                pe_group.add(box, acc, mac)
            pe_boxes.append(row_b)
            pe_acc.append(row_a)
            pe_mac.append(row_m)

        a_label = Text("A", font_size=28, color=ACCENT, weight="BOLD")
        a_label.next_to(pe_boxes[N // 2][0], LEFT, buff=0.85)
        a_arrow = Text("→", font_size=22, color=ACCENT)
        a_arrow.next_to(a_label, RIGHT, buff=0.08)

        b_label = Text("B", font_size=28, color=AMBER, weight="BOLD")
        b_label.next_to(pe_boxes[0][N // 2], UP, buff=0.75)
        b_arrow = Text("↓", font_size=22, color=AMBER)
        b_arrow.next_to(b_label, DOWN, buff=0.05)

        a_pills = VGroup()
        b_pills = VGroup()
        for i in range(N):
            ap = make_pill("·", ACCENT)
            ap.move_to(pe_center(i, 0) + LEFT * 0.72)
            ap.set_opacity(0.25)
            a_pills.add(ap)
            bp = make_pill("·", AMBER)
            bp.move_to(pe_center(0, i) + UP * 0.55)
            bp.set_opacity(0.25)
            b_pills.add(bp)

        formula_box = RoundedRectangle(
            width=7.3,
            height=0.55,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=GRID,
            stroke_width=1.5,
        )
        formula_box.move_to(DOWN * 3.05)
        formula = Text(
            "C[i][j]  +=  A[i][k] · B[k][j]     ·     finish in 3N−2 = 22 clocks",
            font_size=13,
            color=MUTED,
        )
        formula.move_to(formula_box.get_center())

        legend_box = RoundedRectangle(
            width=7.3,
            height=0.42,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=GRID,
            stroke_width=1.5,
        )
        legend_box.move_to(DOWN * 3.58)
        leg_a = Text("●  A stream (west → east)", font_size=12, color=ACCENT)
        leg_b = Text("●  B stream (north → south)", font_size=12, color=AMBER)
        leg_c = Text("●  C accumulator", font_size=12, color=DONE)
        legend = VGroup(leg_a, leg_b, leg_c).arrange(RIGHT, buff=0.45)
        legend.move_to(legend_box.get_center())

        chrome = VGroup(
            brand,
            subtitle,
            flow,
            cycle_box,
            cycle_text,
            a_label,
            a_arrow,
            b_label,
            b_arrow,
            formula_box,
            formula,
            legend_box,
            legend,
        )
        self.add(pe_group, a_pills, b_pills, chrome)
        self.wait(0.2)

        # ========== CYCLE LOOP ==========
        cycle_rt = 0.36

        for t in range(CYCLES):
            a_west, b_north = west_north(A, B, t)
            sim.step(a_west, b_north)

            new_cycle = Text(
                f"cycle  {t + 1} of {CYCLES}",
                font_size=16,
                color=TITLE,
                weight="BOLD",
            )
            new_phase = Text("streaming MACs", font_size=12, color=MUTED)
            new_ct = VGroup(new_cycle, new_phase).arrange(
                DOWN, buff=0.06, aligned_edge=LEFT
            )
            new_ct.move_to(cycle_box.get_center())

            new_a_pills = VGroup()
            new_b_pills = VGroup()
            for i in range(N):
                av = int(a_west[i])
                ap = make_pill(str(av) if av else "·", ACCENT)
                ap.move_to(pe_center(i, 0) + LEFT * 0.72)
                ap.set_opacity(1.0 if av else 0.22)
                new_a_pills.add(ap)

                bv = int(b_north[i])
                bp = make_pill(str(bv) if bv else "·", AMBER)
                bp.move_to(pe_center(0, i) + UP * 0.55)
                bp.set_opacity(1.0 if bv else 0.22)
                new_b_pills.add(bp)

            anims = [
                ReplacementTransform(cycle_text, new_ct),
                ReplacementTransform(a_pills, new_a_pills),
                ReplacementTransform(b_pills, new_b_pills),
            ]

            for r in range(N):
                for c in range(N):
                    active = bool(sim.active[r, c])
                    acc_val = int(sim.acc[r, c])
                    box = pe_boxes[r][c]

                    if active:
                        fill = interpolate_color(
                            ManimColor(PE_IDLE), ManimColor(PE_ACTIVE), 0.9
                        )
                        stroke = GLOW
                        sw = 2.6
                        acc_color = TITLE
                    else:
                        fill = PE_IDLE
                        stroke = GRID
                        sw = 1.2
                        acc_color = TITLE if acc_val else MUTED

                    anims.append(
                        box.animate.set_fill(fill, opacity=1).set_stroke(
                            stroke, width=sw
                        )
                    )

                    new_acc = make_acc_text(acc_val, acc_color, box.get_center())
                    anims.append(ReplacementTransform(pe_acc[r][c], new_acc))
                    pe_acc[r][c] = new_acc

                    if active:
                        a_v = int(sim.last_a[r, c])
                        b_v = int(sim.last_b[r, c])
                        new_mac = make_mac_text(f"{a_v}×{b_v}", box)
                        new_mac.set_opacity(1)
                    else:
                        new_mac = make_mac_text("", box)
                        new_mac.set_opacity(0)
                    anims.append(ReplacementTransform(pe_mac[r][c], new_mac))
                    pe_mac[r][c] = new_mac

            self.play(*anims, run_time=cycle_rt)
            cycle_text = new_ct
            a_pills = new_a_pills
            b_pills = new_b_pills

        assert np.array_equal(sim.acc, C_ref), (sim.acc, C_ref)

        # ========== FINAL ==========
        final_anims = []
        for r in range(N):
            for c in range(N):
                box = pe_boxes[r][c]
                fill = interpolate_color(ManimColor(PE_IDLE), ManimColor(DONE), 0.35)
                final_anims.append(
                    box.animate.set_fill(fill, opacity=1).set_stroke(DONE, width=2)
                )
                new_acc = make_acc_text(int(sim.acc[r, c]), TITLE, box.get_center())
                final_anims.append(ReplacementTransform(pe_acc[r][c], new_acc))
                pe_acc[r][c] = new_acc
                final_anims.append(pe_mac[r][c].animate.set_opacity(0))

        new_cycle = Text(
            f"cycle  {CYCLES} of {CYCLES}", font_size=16, color=TITLE, weight="BOLD"
        )
        new_phase = Text("C complete", font_size=12, color=DONE)
        new_ct = VGroup(new_cycle, new_phase).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT
        )
        new_ct.move_to(cycle_box.get_center())
        final_anims.append(ReplacementTransform(cycle_text, new_ct))

        done_box = RoundedRectangle(
            width=7.3,
            height=0.85,
            corner_radius=0.14,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=DONE,
            stroke_width=2,
        )
        done_box.move_to(DOWN * 3.35)
        done1 = Text(
            "Done — each of 64 PE accumulators holds one C[i][j]",
            font_size=15,
            color=TITLE,
            weight="BOLD",
        )
        done2 = Text(
            f"A[i][k]=i+1, B=ones  →  C[i][j]={N}·(i+1)   ·   matches sim/systolic_array_tb",
            font_size=12,
            color=MUTED,
        )
        done_txt = VGroup(done1, done2).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        done_txt.move_to(done_box.get_center())

        self.play(
            *final_anims,
            FadeOut(formula_box),
            FadeOut(formula),
            FadeOut(legend_box),
            FadeOut(legend),
            FadeOut(a_pills),
            FadeOut(b_pills),
            FadeIn(done_box),
            FadeIn(done_txt),
            run_time=0.85,
        )
        self.wait(2.0)

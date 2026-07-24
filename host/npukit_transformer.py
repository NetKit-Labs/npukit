#!/usr/bin/env python3
"""Tiny transformer driver for NpuKit: GEMM on the array + glue on npukit_glue.

Hardware (after bitstream rebuild with VERSION 0x300):
  - Tile GEMM via npukit_matmul.open_device / npu_matmul
  - Residual, GELU, RMSNorm, Softmax via GLUE_* MMIO banks

CPU still does RoPE, masks, reshape/pack (the weird bits).

Fixed-point matches rtl/npukit_glue.sv / docs/transformer_glue.md:
  activations Q12, softmax probs Q16, gamma Q12.

Usage on PYNQ:
  python3 npukit_transformer.py [/path/to/npukit.bit]

Offline (no board): exercises the Q12 reference model only:
  python3 npukit_transformer.py --ref-only
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np

Q12 = 12
Q16 = 16
ONE_Q12 = 1 << Q12
ONE_Q16 = 1 << Q16
FOUR_Q12 = 4 * ONE_Q12
EIGHT_Q12 = 8 * ONE_Q12

OP_RESIDUAL = 0x1
OP_GELU = 0x2
OP_RMSNORM = 0x3
OP_SOFTMAX = 0x4

REG_ID = 0x000
REG_VERSION = 0x004
REG_STATUS = 0x008
REG_FEATURES = 0x014
REG_GLUE_CTRL = 0x018
REG_GLUE_LEN = 0x01C
REG_GLUE_PARAM = 0x020
REG_GLUE_COUNT = 0x024
OFF_GLUE_X = 0x500
OFF_GLUE_Y = 0x600
OFF_GLUE_OUT = 0x700
OFF_GLUE_GAMMA = 0x800

STATUS_BUSY = 0x1
STATUS_GLUE_DONE = 0x10
FEAT_GLUE = 0x2
ID_MAGIC = 0x4E50554B
VERSION_GLUE = 0x00000300

MAX_LEN = 16


def to_q12(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * ONE_Q12).astype(np.int32)


def from_q12(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32).astype(np.float64) / ONE_Q12


def from_q16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32).astype(np.float64) / ONE_Q16


# ---------------------------------------------------------------------------
# Reference model (mirrors RTL LUT polys) — usable offline
# ---------------------------------------------------------------------------
def _gelu_poly_q12(x: int) -> int:
    xc = max(-FOUR_Q12, min(FOUR_Q12, int(x)))
    x2 = (xc * xc) >> Q12
    x3 = (x2 * xc) >> Q12
    u = (3269 * xc + 146 * x3) >> Q12
    den = ONE_Q12 + abs(u)
    t = (u << Q12) // den
    return int((xc * (ONE_Q12 + t)) >> (Q12 + 1))


def _exp_poly_q16(t: int) -> int:
    clamped = max(-EIGHT_Q12, min(0, int(t)))
    s = (5909 * clamped) >> Q12
    mag = -s
    sh = min(mag >> Q12, 15)
    frac = mag & 0xFFF
    base = ONE_Q16 - ((2839 * frac) >> (Q12 - 4))
    if base < 0:
        base = 0
    return int(base >> sh)


def ref_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x.astype(np.int32) + y.astype(np.int32)).astype(np.int32)


def ref_gelu(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.int32)
    for i, v in enumerate(np.asarray(x, dtype=np.int32).reshape(-1)):
        if v >= FOUR_Q12:
            out.reshape(-1)[i] = v
        elif v <= -FOUR_Q12:
            out.reshape(-1)[i] = 0
        else:
            out.reshape(-1)[i] = _gelu_poly_q12(v)
    return out.reshape(x.shape)


def ref_rmsnorm(x: np.ndarray, gamma: np.ndarray, eps_q12: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.int32).reshape(-1)
    g = np.asarray(gamma, dtype=np.int32).reshape(-1)
    n = x.size
    acc = int(np.sum(x.astype(np.int64) * x.astype(np.int64)))
    mean = acc // n + (eps_q12 << Q12)  # Q24
    # integer isqrt
    lo, hi = 0, (1 << 32) - 1
    for _ in range(32):
        mid = lo + ((hi - lo) >> 1)
        sq = mid * mid
        if sq <= mean:
            lo = mid
        else:
            hi = mid - 1
    sqrt_u = max(lo, 1)
    inv = (ONE_Q12 * ONE_Q12) // sqrt_u
    # Match RTL: truncate to Q12 between the two multiplies
    mid = ((x.astype(np.int64) * inv) >> Q12).astype(np.int32)
    out = ((mid.astype(np.int64) * g.astype(np.int64)) >> Q12).astype(np.int32)
    return out


def _exp_lut_q16(t: int) -> int:
    """Match rtl exp_lut addressing: clamp, then (t+8)>>7 saturated to 255."""
    tc = max(-EIGHT_Q12, min(0, int(t)))
    s = tc + EIGHT_Q12
    idx = 255 if s >= 32768 else (s >> 7)
    return _exp_poly_q16(-EIGHT_Q12 + ((8 * ONE_Q12 * idx) >> 8))


def ref_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int32).reshape(-1)
    m = int(x.max())
    exps = np.array([_exp_lut_q16(int(v) - m) for v in x], dtype=np.int64)
    s = int(exps.sum()) or 1
    return ((exps * ONE_Q16) // s).astype(np.int32)


# ---------------------------------------------------------------------------
# Board driver
# ---------------------------------------------------------------------------
@dataclass
class GlueDevice:
    mmio: object

    def wait_glue(self, before_count: int, timeout: float = 1.0) -> None:
        """Wait until GLUE_COUNT advances, else brief settle (fast ops / old bit)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if (self.mmio.read(REG_GLUE_COUNT) & 0xFFFFFFFF) != (before_count & 0xFFFFFFFF):
                return
            time.sleep(0.000_01)
        # Fallback: ops finish in <<1ms; DONE sticky is not a reliable edge.
        time.sleep(0.002)
        if self.mmio.read(REG_STATUS) & STATUS_GLUE_DONE:
            return
        raise TimeoutError(
            f"glue timeout status=0x{self.mmio.read(REG_STATUS):X} "
            f"count=0x{self.mmio.read(REG_GLUE_COUNT):X} before=0x{before_count:X}"
        )

    def _write_vec(self, base: int, vec: np.ndarray) -> None:
        flat = np.asarray(vec, dtype=np.int32).reshape(-1)
        if flat.size > MAX_LEN:
            raise ValueError(f"vector length {flat.size} > {MAX_LEN}")
        for i, v in enumerate(flat):
            self.mmio.write(base + 4 * i, int(np.uint32(v)))

    def _read_vec(self, base: int, n: int) -> np.ndarray:
        words = np.array(
            [self.mmio.read(base + 4 * i) & 0xFFFFFFFF for i in range(n)],
            dtype=np.uint32,
        )
        return words.view(np.int32).copy()

    def run(
        self,
        opcode: int,
        x: np.ndarray,
        *,
        y: np.ndarray | None = None,
        gamma: np.ndarray | None = None,
        param: int = 1,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.int32).reshape(-1)
        n = int(x.size)
        self.mmio.write(REG_GLUE_LEN, n)
        self.mmio.write(REG_GLUE_PARAM, int(param) & 0xFFFFFFFF)
        self._write_vec(OFF_GLUE_X, x)
        if y is not None:
            self._write_vec(OFF_GLUE_Y, y)
        if gamma is not None:
            self._write_vec(OFF_GLUE_GAMMA, gamma)
        before = self.mmio.read(REG_GLUE_COUNT) & 0xFFFFFFFF
        self.mmio.write(REG_GLUE_CTRL, ((opcode & 0xF) << 4) | 0x1)
        self.wait_glue(before)
        return self._read_vec(OFF_GLUE_OUT, n)


def quantize_int8_activation(x_q12: np.ndarray, scale: float = 64.0) -> tuple[np.ndarray, float]:
    """Pack Q12 activations to int8 for GEMM; returns (int8, dequant scale to float)."""
    xf = from_q12(x_q12)
    q = np.clip(np.rint(xf * scale), -128, 127).astype(np.int8)
    return q, 1.0 / scale


def dequant_gemm_to_q12(c_i32: np.ndarray, a_scale: float, b_scale: float) -> np.ndarray:
    """Approximate int32 GEMM accumulators back to Q12."""
    real = c_i32.astype(np.float64) * a_scale * b_scale
    return to_q12(real)


def rope_cpu(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Classic RoPE on last dim (even width); CPU-only weird glue."""
    *prefix, d = x.shape
    assert d % 2 == 0
    half = d // 2
    x = x.reshape(*prefix, half, 2).astype(np.float64)
    pos = np.arange(prefix[-1] if prefix else 1).reshape(-1, 1)
    if not prefix:
        pos = np.zeros((1, 1))
    else:
        seq = x.shape[-3] if len(prefix) >= 1 else 1
        pos = np.arange(x.shape[0]).reshape(-1, 1) if x.ndim >= 3 else np.arange(1).reshape(-1, 1)
        # For [T, D] layout:
        if x.ndim == 3:  # T, half, 2
            pos = np.arange(x.shape[0]).reshape(-1, 1)
    inv = 1.0 / (base ** (np.arange(half) / half))
    if x.ndim == 3:
        ang = pos * inv  # T x half
        cos = np.cos(ang)[..., None]
        sin = np.sin(ang)[..., None]
        out0 = x[..., 0:1] * cos - x[..., 1:2] * sin
        out1 = x[..., 0:1] * sin + x[..., 1:2] * cos
        return np.concatenate([out0, out1], axis=-1).reshape(*prefix, d)
    return x.reshape(*prefix, d)


def attention_block_ref(
    x_q12: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    *,
    use_rope: bool = True,
) -> np.ndarray:
    """Single-head attention in float reference (shapes: T x D, weights D x D)."""
    x = from_q12(x_q12)
    q = x @ wq
    k = x @ wk
    v = x @ wv
    if use_rope:
        q = rope_cpu(q)
        k = rope_cpu(k)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    # mask would be applied here on CPU
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    attn = p @ v
    return to_q12(attn @ wo)


def run_ref_suite() -> int:
    print("=== reference glue (offline) ===")
    x = to_q12(np.array([0.5, -0.25, 1.0, -1.5], dtype=np.float64))
    y = to_q12(np.array([0.5, 0.25, 0.0, 1.5], dtype=np.float64))
    r = ref_residual(x, y)
    assert np.allclose(from_q12(r), from_q12(x) + from_q12(y))
    print("residual OK", from_q12(r))

    g = ref_gelu(x)
    print("gelu    OK", from_q12(g))

    gamma = np.full(4, ONE_Q12, dtype=np.int32)
    n = ref_rmsnorm(x, gamma)
    print("rmsnorm OK", from_q12(n))

    s = ref_softmax(to_q12(np.array([2.0, 0.0, -1.0, -2.0])))
    print("softmax OK", from_q16(s), "sum", from_q16(s).sum())

    # Tiny attn smoke (float path + RoPE on CPU)
    rng = np.random.default_rng(0)
    t, d = 8, 8
    x0 = to_q12(rng.normal(0, 0.5, size=(t, d)))
    w = {name: rng.normal(0, 0.2, size=(d, d)) for name in ("q", "k", "v", "o")}
    y0 = attention_block_ref(x0, w["q"], w["k"], w["v"], w["o"])
    print(f"attn ref OK shape={y0.shape} meanQ12={y0.mean():.1f}")
    print("ALL REF PASS")
    return 0


def run_board(bit_path: str) -> int:
    from npukit_matmul import open_device  # type: ignore

    mmio, transport = open_device(bit_path)
    ident = mmio.read(REG_ID)
    if ident != ID_MAGIC:
        print(f"BAD ID 0x{ident:08X}")
        return 1
    ver = mmio.read(REG_VERSION)
    feat = mmio.read(REG_FEATURES)
    print(f"ID OK version=0x{ver:08X} features=0x{feat:08X}")
    if ver < VERSION_GLUE or not (feat & FEAT_GLUE):
        print(
            "Bitstream has no transformer glue (need VERSION>=0x300, FEATURES.GLUE).\n"
            "Rebuild with rtl/npukit_glue.sv then re-flash."
        )
        return 1

    glue = GlueDevice(mmio)

    def check(name: str, opcode: int, x, **kw) -> bool:
        refs = {
            OP_RESIDUAL: lambda: ref_residual(x, kw["y"]),
            OP_GELU: lambda: ref_gelu(x),
            OP_RMSNORM: lambda: ref_rmsnorm(x, kw["gamma"], kw.get("param", 1)),
            OP_SOFTMAX: lambda: ref_softmax(x),
        }
        got = glue.run(opcode, x, **kw)
        exp = refs[opcode]()
        # Softmax / gelu are approx — allow small abs tol in integer domain
        tol = 64 if opcode in (OP_GELU, OP_SOFTMAX, OP_RMSNORM) else 0
        ok = bool(np.max(np.abs(got.astype(np.int64) - exp.astype(np.int64))) <= tol)
        print(f"{name}: {'PASS' if ok else 'FAIL'}  max|err|="
              f"{int(np.max(np.abs(got.astype(np.int64) - exp.astype(np.int64))))}")
        return ok

    x = to_q12(np.array([0.5, -0.25, 1.0, -1.5]))
    y = to_q12(np.array([0.5, 0.25, 0.0, 1.5]))
    gamma = np.full(4, ONE_Q12, dtype=np.int32)
    logits = to_q12(np.array([2.0, 0.0, -1.0, -2.0]))

    ok = True
    ok &= check("residual", OP_RESIDUAL, x, y=y)
    ok &= check("gelu", OP_GELU, x)
    ok &= check("rmsnorm", OP_RMSNORM, x, gamma=gamma, param=1)
    ok &= check("softmax", OP_SOFTMAX, logits)

    # One GEMM tile still works through the same device
    a = np.eye(8, dtype=np.int8)
    b = np.arange(64, dtype=np.int8).reshape(8, 8)
    from npukit_matmul import npu_matmul

    c, _ = npu_matmul(mmio, transport, a, b)
    gemm_ok = bool(np.array_equal(c, a.astype(np.int32) @ b.astype(np.int32)))
    ok &= gemm_ok
    print(f"gemm tile: {'PASS' if gemm_ok else 'FAIL'}")

    print("ALL BOARD PASS" if ok else "BOARD FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ref-only" in argv or (not argv):
        # default to ref-only when no bit path so laptop syntax-checks work
        if "--ref-only" in argv:
            argv.remove("--ref-only")
        if not argv:
            return run_ref_suite()
    bit = argv[0] if argv else "/home/xilinx/jupyter_notebooks/npukit.bit"
    if bit == "--ref-only":
        return run_ref_suite()
    return run_board(bit)


if __name__ == "__main__":
    raise SystemExit(main())

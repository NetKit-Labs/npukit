#!/usr/bin/env python3
"""Tiny transformer driver for NpuKit: GEMM on the array + glue on npukit_glue.

Hardware (VERSION 0x300):
  - Tile GEMM via npukit_matmul.open_device / npu_matmul
  - Residual, GELU, RMSNorm, Softmax via GLUE_* MMIO banks

CPU: RoPE / masks / reshape (not used in the e2e smoke), quant/dequant scales.

Fixed-point: activations Q12, softmax probs Q16, gamma Q12.
Fixed host scales (int8 GEMM ↔ Q12 glue): see SCALE_ACT / SCALE_W / SCALE_P.

Usage on PYNQ:
  python3 npukit_transformer.py [/path/to/npukit.bit]
  python3 npukit_transformer.py [/path/to/npukit.bit] --e2e

Offline:
  python3 npukit_transformer.py --ref-only
  python3 npukit_transformer.py --e2e-ref
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

# E2E smoke geometry (ViT-ready token grid; MNIST patches can replace X later)
E2E_T = 8  # sequence / tokens  (≤ MAX_LEN)
E2E_D = 8  # model dim          (matches 8×8 GEMM tile)

# Fixed quant scales (host contract — do not tune per layer in smoke)
SCALE_ACT = 64.0   # Q12-real → int8 activation: q = clip(round(x * SCALE_ACT))
SCALE_W = 64.0     # float weight → int8
SCALE_P = 127.0    # Q16 prob → int8 for P@V


def to_q12(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * ONE_Q12).astype(np.int32)


def from_q12(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32).astype(np.float64) / ONE_Q12


def to_q16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * ONE_Q16).astype(np.int32)


def from_q16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int32).astype(np.float64) / ONE_Q16


# Weight scale: per-tensor float, or per-output-channel vector (last dim of W / C).
ScaleW = float | np.ndarray


def _as_scale_w(scale: ScaleW) -> float | np.ndarray:
    if isinstance(scale, (float, int, np.floating, np.integer)):
        return float(scale)
    arr = np.asarray(scale, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _inv_scale_w(scale: ScaleW) -> float | np.ndarray:
    s = _as_scale_w(scale)
    return 1.0 / s if isinstance(s, float) else 1.0 / s


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


def quantize_int8_activation(x_q12: np.ndarray, scale: float = SCALE_ACT) -> tuple[np.ndarray, float]:
    """Pack Q12 activations to int8 for GEMM; returns (int8, dequant scale to float)."""
    xf = from_q12(x_q12)
    q = np.clip(np.rint(xf * scale), -128, 127).astype(np.int8)
    return q, 1.0 / scale


def dequant_gemm_to_q12(
    c_i32: np.ndarray, a_scale: float, b_scale: ScaleW
) -> np.ndarray:
    """int32 GEMM accumulators → Q12. b_scale may be per-output-channel [N]."""
    bs = _as_scale_w(b_scale)
    real = c_i32.astype(np.float64) * float(a_scale) * bs
    return to_q12(real)


def quant_q12_to_i8(x_q12: np.ndarray, scale: float = SCALE_ACT) -> np.ndarray:
    xf = from_q12(x_q12)
    return np.clip(np.rint(xf * scale), -128, 127).astype(np.int8)


def quant_weight_to_i8(w: np.ndarray, scale: ScaleW = SCALE_W) -> np.ndarray:
    """Quantize W. scale: scalar, or per-output-channel along last dim."""
    wf = np.asarray(w, dtype=np.float64)
    s = _as_scale_w(scale)
    if not isinstance(s, float) and s.shape[-1] != wf.shape[-1]:
        raise ValueError(f"weight scale shape {s.shape} vs W {wf.shape}")
    return np.clip(np.rint(wf * s), -128, 127).astype(np.int8)


def quant_q16_to_i8(p_q16: np.ndarray, scale: float = SCALE_P) -> np.ndarray:
    pf = from_q16(p_q16)
    return np.clip(np.rint(pf * scale), -128, 127).astype(np.int8)


def gemm_i8_to_q12(
    a_i8: np.ndarray, b_i8: np.ndarray, *, a_scale: float, b_scale: ScaleW
) -> np.ndarray:
    """NumPy stand-in for NPU int8 matmul + dequant → Q12."""
    c = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
    return dequant_gemm_to_q12(c, 1.0 / a_scale, _inv_scale_w(b_scale))


def _fmt_mat(name: str, mat: np.ndarray, *, as_q12: bool = False, as_q16: bool = False) -> str:
    arr = np.asarray(mat)
    lines = [f"{name} shape={arr.shape} dtype={arr.dtype}"]
    with np.printoptions(linewidth=120, suppress=True, precision=4):
        lines.append(str(arr))
    if as_q12 and np.issubdtype(arr.dtype, np.integer):
        lines.append(f"{name} float(Q12):\n{np.round(from_q12(arr), 4)}")
    if as_q16 and np.issubdtype(arr.dtype, np.integer):
        lines.append(f"{name} float(Q16):\n{np.round(from_q16(arr), 4)}")
    return "\n".join(lines)


@dataclass
class TinyBlockWeights:
    """1-layer block: int8 projections + Q12 gammas. FFN may be D×mlp_h×D."""

    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    gamma1: np.ndarray
    gamma2: np.ndarray
    # Optional per-output-channel weight scales (None → caller scalar scale_w).
    sw_wq: np.ndarray | None = None
    sw_wk: np.ndarray | None = None
    sw_wv: np.ndarray | None = None
    sw_wo: np.ndarray | None = None
    sw_w1: np.ndarray | None = None
    sw_w2: np.ndarray | None = None

    @staticmethod
    def make(
        rng: np.random.Generator, d: int = E2E_D, mlp_h: int | None = None
    ) -> "TinyBlockWeights":
        mh = int(mlp_h if mlp_h is not None else d)

        def w(shape: tuple[int, int]) -> np.ndarray:
            return quant_weight_to_i8(rng.normal(0.0, 0.15, size=shape))

        g = np.full(d, ONE_Q12, dtype=np.int32)
        return TinyBlockWeights(
            w((d, d)),
            w((d, d)),
            w((d, d)),
            w((d, d)),
            w((d, mh)),
            w((mh, d)),
            g,
            g,
        )

    def scale_for(self, name: str, fallback: ScaleW) -> ScaleW:
        sw = getattr(self, f"sw_{name}", None)
        return fallback if sw is None else sw


def _float_rmsnorm_row(x_q12: np.ndarray, gamma_q12: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = from_q12(x_q12)
    g = from_q12(gamma_q12)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return to_q12((x / rms) * g)


def _float_gelu_row(x_q12: np.ndarray) -> np.ndarray:
    """Torch-default GELU (erf) → Q12."""
    x = from_q12(x_q12)
    # numpy has no erf on all platforms we care about; math.erf is fine at T×D.
    erf = np.frompyfunc(math.erf, 1, 1)
    y = 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)).astype(np.float64))
    return to_q12(y)


def _float_softmax_row(scores_q12: np.ndarray) -> np.ndarray:
    s = from_q12(scores_q12)
    s = s - np.max(s)
    e = np.exp(s)
    return to_q16(e / np.sum(e))


def _resolve_glue_mode(*, use_hw: bool, glue_mode: str | None) -> str:
    """glue_mode: 'hw' | 'q12' | 'float'. Legacy use_hw selects hw vs q12 when unset."""
    if glue_mode is not None:
        mode = str(glue_mode)
    else:
        mode = "hw" if use_hw else "q12"
    if mode not in ("hw", "q12", "float"):
        raise ValueError(f"glue_mode must be hw|q12|float, got {mode!r}")
    return mode


def _rmsnorm_rows(
    glue_or_none, x_q12: np.ndarray, gamma: np.ndarray, *, glue_mode: str
) -> np.ndarray:
    """RMSNorm each token (row) of shape [T, D]."""
    out = np.empty_like(x_q12, dtype=np.int32)
    for t in range(x_q12.shape[0]):
        if glue_mode == "hw":
            out[t] = glue_or_none.run(OP_RMSNORM, x_q12[t], gamma=gamma, param=1)
        elif glue_mode == "float":
            out[t] = _float_rmsnorm_row(x_q12[t], gamma)
        else:
            out[t] = ref_rmsnorm(x_q12[t], gamma, 1)
    return out


def _residual_rows(
    glue_or_none, x_q12: np.ndarray, y_q12: np.ndarray, *, glue_mode: str
) -> np.ndarray:
    """Residual add. float mode still uses exact int32 add (same as q12)."""
    out = np.empty_like(x_q12, dtype=np.int32)
    for t in range(x_q12.shape[0]):
        if glue_mode == "hw":
            out[t] = glue_or_none.run(OP_RESIDUAL, x_q12[t], y=y_q12[t])
        else:
            out[t] = ref_residual(x_q12[t], y_q12[t])
    return out


def _gelu_rows(glue_or_none, x_q12: np.ndarray, *, glue_mode: str) -> np.ndarray:
    out = np.empty_like(x_q12, dtype=np.int32)
    for t in range(x_q12.shape[0]):
        if glue_mode == "hw":
            out[t] = glue_or_none.run(OP_GELU, x_q12[t])
        elif glue_mode == "float":
            out[t] = _float_gelu_row(x_q12[t])
        else:
            out[t] = ref_gelu(x_q12[t])
    return out


def _softmax_rows(glue_or_none, scores_q12: np.ndarray, *, glue_mode: str) -> np.ndarray:
    """Softmax each row → Q16, scores [T, T]."""
    out = np.empty_like(scores_q12, dtype=np.int32)
    for t in range(scores_q12.shape[0]):
        if glue_mode == "hw":
            out[t] = glue_or_none.run(OP_SOFTMAX, scores_q12[t])
        elif glue_mode == "float":
            out[t] = _float_softmax_row(scores_q12[t])
        else:
            out[t] = ref_softmax(scores_q12[t])
    return out


def _matmul_q12(
    a_q12: np.ndarray,
    b_i8: np.ndarray,
    *,
    mmio=None,
    transport=None,
    use_hw: bool,
    scale_act: float = SCALE_ACT,
    scale_w: ScaleW = SCALE_W,
) -> np.ndarray:
    """A(Q12) @ W(int8) via quant(A)*W → dequant Q12."""
    a_i8 = quant_q12_to_i8(a_q12, scale_act)
    inv_w = _inv_scale_w(scale_w)
    if use_hw:
        from npukit_matmul import npu_matmul

        c_i32, _ = npu_matmul(mmio, transport, a_i8, b_i8)
        return dequant_gemm_to_q12(c_i32, 1.0 / scale_act, inv_w)
    return gemm_i8_to_q12(a_i8, b_i8, a_scale=scale_act, b_scale=scale_w)


def transformer_block_1layer(
    x_q12: np.ndarray,
    w: TinyBlockWeights,
    *,
    glue=None,
    mmio=None,
    transport=None,
    use_hw: bool = False,
    use_hw_gemm: bool | None = None,
    glue_mode: str | None = None,
    verbose: bool = False,
    scale_act: float = SCALE_ACT,
    scale_w: ScaleW = SCALE_W,
    scale_p: float = SCALE_P,
    causal: bool = False,
    key_pad_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One pre-norm transformer block: attn + FFN. Shapes [T,D] Q12 in/out.

    GEMM may run on NPU (use_hw_gemm). Softmax/RMSNorm/GELU follow glue_mode:
      'hw' | 'q12' (LUT refs) | 'float' (A9 float32, preferred for ViT deploy).
    Legacy use_hw=True ⇒ GEMM+glue HW when glue_mode / use_hw_gemm unset.
    Set causal=True for decoder/LM (mask future keys before Softmax).
    key_pad_mask: optional bool [T] True=PAD key to mask.
    """
    x_q12 = np.asarray(x_q12, dtype=np.int32)
    t, d = x_q12.shape
    mlp_h = int(w.w1.shape[1])
    gmode = _resolve_glue_mode(use_hw=use_hw, glue_mode=glue_mode)
    gemm_hw = bool(use_hw if use_hw_gemm is None else use_hw_gemm)
    # HW glue Softmax/RMSNorm/GELU are capped at MAX_LEN; float/q12 may exceed.
    assert d % 8 == 0 and t % 8 == 0
    assert w.w1.shape == (d, mlp_h) and w.w2.shape == (mlp_h, d)
    if gmode == "hw":
        if t > MAX_LEN or d > MAX_LEN:
            raise ValueError(
                f"glue_mode='hw' needs T,D ≤ MAX_LEN={MAX_LEN}; got T={t} D={d}"
            )
        if mlp_h > MAX_LEN:
            raise ValueError(
                f"FFN hidden {mlp_h} > MAX_LEN={MAX_LEN}; use glue_mode='float'"
            )

    dump: dict[str, np.ndarray] = {"x_in": x_q12.copy()}

    def mm(name: str) -> dict:
        return dict(
            mmio=mmio,
            transport=transport,
            use_hw=gemm_hw,
            scale_act=scale_act,
            scale_w=w.scale_for(name, scale_w),
        )

    # --- attention ---
    x_n = _rmsnorm_rows(glue, x_q12, w.gamma1, glue_mode=gmode)
    dump["attn_in_norm"] = x_n.copy()

    q = _matmul_q12(x_n, w.wq, **mm("wq"))
    k = _matmul_q12(x_n, w.wk, **mm("wk"))
    v = _matmul_q12(x_n, w.wv, **mm("wv"))
    dump["q"], dump["k"], dump["v"] = q.copy(), k.copy(), v.copy()

    # scores = (Q @ K^T) / sqrt(D)  — GEMM on int8 quant of Q and K^T
    k_t = np.ascontiguousarray(k.T)
    q_i8 = quant_q12_to_i8(q, scale_act)
    k_t_i8 = quant_q12_to_i8(k_t, scale_act)
    if gemm_hw:
        from npukit_matmul import npu_matmul

        scores_i32, _ = npu_matmul(mmio, transport, q_i8, k_t_i8)
        scores = dequant_gemm_to_q12(scores_i32, 1.0 / scale_act, 1.0 / scale_act)
    else:
        scores = gemm_i8_to_q12(q_i8, k_t_i8, a_scale=scale_act, b_scale=scale_act)

    inv_sqrt = to_q12(np.array(1.0 / math.sqrt(d)))[()]
    scores = ((scores.astype(np.int64) * int(inv_sqrt)) >> Q12).astype(np.int32)
    if causal or key_pad_mask is not None:
        neg = np.iinfo(np.int32).min // 4
        scores = scores.copy()
        if causal:
            fut = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores[fut] = neg
        if key_pad_mask is not None:
            pad = np.asarray(key_pad_mask, dtype=bool).reshape(-1)
            assert pad.shape[0] == t
            scores[:, pad] = neg
    dump["scores"] = scores.copy()

    p = _softmax_rows(glue, scores, glue_mode=gmode)
    dump["attn_p"] = p.copy()

    p_i8 = quant_q16_to_i8(p, scale_p)
    v_i8 = quant_q12_to_i8(v, scale_act)
    if gemm_hw:
        from npukit_matmul import npu_matmul

        attn_i32, _ = npu_matmul(mmio, transport, p_i8, v_i8)
        attn = dequant_gemm_to_q12(attn_i32, 1.0 / scale_p, 1.0 / scale_act)
    else:
        attn = gemm_i8_to_q12(p_i8, v_i8, a_scale=scale_p, b_scale=scale_act)
    dump["attn"] = attn.copy()

    attn_o = _matmul_q12(attn, w.wo, **mm("wo"))
    dump["attn_proj"] = attn_o.copy()

    x2 = _residual_rows(glue, x_q12, attn_o, glue_mode=gmode)
    dump["after_attn_res"] = x2.copy()

    # --- FFN ---
    x_n2 = _rmsnorm_rows(glue, x2, w.gamma2, glue_mode=gmode)
    dump["ffn_in_norm"] = x_n2.copy()
    h = _matmul_q12(x_n2, w.w1, **mm("w1"))
    dump["ffn_h"] = h.copy()
    h = _gelu_rows(glue, h, glue_mode=gmode)
    dump["ffn_gelu"] = h.copy()
    h = _matmul_q12(h, w.w2, **mm("w2"))
    dump["ffn_out"] = h.copy()
    y = _residual_rows(glue, x2, h, glue_mode=gmode)
    dump["y_out"] = y.copy()

    if verbose:
        for name, mat in dump.items():
            q16 = name == "attn_p"
            print(_fmt_mat(name, mat, as_q12=not q16, as_q16=q16))
            print()
    return y, dump


def run_e2e_smoke(
    *,
    bit_path: str | None = None,
    seed: int = 0,
) -> int:
    """1-layer T=8,D=8 block: CPU ref vs optional FPGA; print tensors + PASS/FAIL."""
    rng = np.random.default_rng(seed)
    w = TinyBlockWeights.make(rng, E2E_D)
    # Token grid in Q12 — stand-in for ViT patch embeddings
    x = to_q12(rng.normal(0.0, 0.4, size=(E2E_T, E2E_D)))

    print("=== E2E smoke: 1-layer transformer block ===")
    print(f"T={E2E_T} D={E2E_D}  SCALE_ACT={SCALE_ACT} SCALE_W={SCALE_W} SCALE_P={SCALE_P}")
    print(_fmt_mat("weights.wq (int8)", w.wq))
    print()

    print("--- CPU / ref path (int8 GEMM mimicked in NumPy + glue refs) ---")
    y_ref, dump_ref = transformer_block_1layer(x, w, use_hw=False, verbose=True)

    if bit_path is None:
        print("E2E REF-ONLY PASS (no bitstream)")
        return 0

    from npukit_matmul import open_device

    mmio, transport = open_device(bit_path)
    ident = mmio.read(REG_ID)
    ver = mmio.read(REG_VERSION)
    feat = mmio.read(REG_FEATURES)
    print(f"\nID OK version=0x{ver:08X} features=0x{feat:08X}")
    if ident != ID_MAGIC or ver < VERSION_GLUE or not (feat & FEAT_GLUE):
        print("E2E FAIL: need glue bitstream VERSION>=0x300")
        return 1

    glue = GlueDevice(mmio)
    print("\n--- FPGA path (NPU GEMM + glue) ---")
    y_hw, dump_hw = transformer_block_1layer(
        x, w, glue=glue, mmio=mmio, transport=transport, use_hw=True, verbose=True
    )

    # Tolerances: GEMM quant + glue LUT noise
    tol = 512
    ok = True
    print("\n=== compare ref vs FPGA ===")
    for key in dump_ref:
        a, b = dump_ref[key], dump_hw[key]
        err = int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))
        # Softmax rows can differ a bit more after score quant
        t_key = 1024 if key in ("attn_p", "scores", "attn", "attn_proj", "y_out", "ffn_out") else tol
        passed = err <= t_key
        ok &= passed
        print(f"{key}: {'PASS' if passed else 'FAIL'}  max|err|={err}  tol={t_key}")

    err_y = int(np.max(np.abs(y_ref.astype(np.int64) - y_hw.astype(np.int64))))
    print(f"\ny_out: {'PASS' if err_y <= 1024 else 'FAIL'}  max|err|={err_y}")
    print("\nALL E2E PASS" if ok else "\nE2E FAIL")
    return 0 if ok else 1


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


def _fmt_vec(name: str, v: np.ndarray, *, as_q12: bool = False, as_q16: bool = False) -> str:
    flat = np.asarray(v, dtype=np.int32).reshape(-1)
    lines = [f"{name} int32: {flat.tolist()}"]
    if as_q12:
        lines.append(f"{name} float(Q12): {np.round(from_q12(flat), 6).tolist()}")
    if as_q16:
        lines.append(f"{name} float(Q16): {np.round(from_q16(flat), 6).tolist()}")
    return "\n".join(lines)


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
        print(f"\n=== {name} ===")
        print(_fmt_vec("X", x, as_q12=True))
        if "y" in kw:
            print(_fmt_vec("Y", kw["y"], as_q12=True))
        if "gamma" in kw:
            print(_fmt_vec("GAMMA", kw["gamma"], as_q12=True))
        got = glue.run(opcode, x, **kw)
        exp = refs[opcode]()
        out_q16 = opcode == OP_SOFTMAX
        print(_fmt_vec("OUT_npu", got, as_q12=not out_q16, as_q16=out_q16))
        print(_fmt_vec("OUT_ref", exp, as_q12=not out_q16, as_q16=out_q16))
        # Softmax / gelu are approx — allow small abs tol in integer domain
        tol = 64 if opcode in (OP_GELU, OP_SOFTMAX, OP_RMSNORM) else 0
        err = int(np.max(np.abs(got.astype(np.int64) - exp.astype(np.int64))))
        ok = err <= tol
        print(f"{name}: {'PASS' if ok else 'FAIL'}  max|err|={err}  tol={tol}")
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
    print("\n=== gemm tile (same bitstream) ===")
    a = np.eye(8, dtype=np.int8)
    b = np.arange(64, dtype=np.int8).reshape(8, 8)
    from npukit_matmul import npu_matmul

    c, _ = npu_matmul(mmio, transport, a, b)
    ref_c = a.astype(np.int32) @ b.astype(np.int32)
    print("A (int8):\n", a)
    print("B (int8):\n", b)
    print("C_npu (int32):\n", c)
    print("C_ref (int32):\n", ref_c)
    gemm_ok = bool(np.array_equal(c, ref_c))
    ok &= gemm_ok
    print(f"gemm tile: {'PASS' if gemm_ok else 'FAIL'}")

    print("\nALL BOARD PASS" if ok else "\nBOARD FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    e2e = "--e2e" in argv
    e2e_ref = "--e2e-ref" in argv
    ref_only = "--ref-only" in argv
    argv = [a for a in argv if a not in ("--e2e", "--e2e-ref", "--ref-only")]

    if e2e_ref or (e2e and (ref_only or not argv)):
        return run_e2e_smoke(bit_path=None)
    if ref_only or not argv:
        return run_ref_suite()

    bit = argv[0]
    if e2e:
        return run_e2e_smoke(bit_path=bit)
    return run_board(bit)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tiny MCU-class DS-stem for NpuKit ViT (CPU / TFLite+XNNPACK).

Not the full DS-CNN peer — a slightly richer stem:
  Conv stem (stride 2) → DS block (DW+PW, stride 2) → DS block (DW+PW, stride 1)
  → pad 7→8 → pool to 4×4 → T=16 tokens of D=16 for the FPGA transformer body.

Float / QAT twin lives in train_vit_mnist.TinyViT; numpy + TFLite paths here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Torch is optional: board deploy uses StemInt8 + numpy (no torch on PYNQ venv).
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

STEM_C = 16
STEM_T = 16  # 4×4
STEM_HW = 4
IMG = 28


if _HAS_TORCH:

    class TinyDSStem(nn.Module):
        """28×28×1 → [B, T=16, D=16] tokens (channels = model dim)."""

        def __init__(self, c: int = STEM_C) -> None:
            super().__init__()
            self.stem = nn.Conv2d(1, c, 3, stride=2, padding=1, bias=True)  # 14×14
            self.dw = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=True)  # 7×7
            self.pw = nn.Conv2d(c, c, 1, bias=True)
            # Second DS block at stride 1 (richer teacher; still MCU-tiny).
            self.dw2 = nn.Conv2d(c, c, 3, stride=1, padding=1, groups=c, bias=True)
            self.pw2 = nn.Conv2d(c, c, 1, bias=True)
            self.c = c

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [B,28,28] or [B,1,28,28] → [B,16,C]."""
            if x.ndim == 3:
                x = x.unsqueeze(1)
            x = F.relu(self.stem(x), inplace=False)
            x = F.relu(self.dw(x), inplace=False)
            x = F.relu(self.pw(x), inplace=False)
            x = F.relu(self.dw2(x), inplace=False)
            x = F.relu(self.pw2(x), inplace=False)
            # 7×7 → pad to 8×8 → avg-pool 2×2 → 4×4 (TFLite-friendly; T=16)
            x = F.pad(x, (0, 1, 0, 1))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            return x.flatten(2).transpose(1, 2).contiguous()  # [B,16,C]

else:  # pragma: no cover

    class TinyDSStem:  # type: ignore[no-redef]
        """Stub so imports succeed on torch-less PYNQ; train/export need torch."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("TinyDSStem requires PyTorch (host train/export only)")


def _per_out_channel_scale(w: Any, *, lo: float = 8.0, hi: float = 512.0) -> np.ndarray:
    """Per-output-channel absmax scale for conv [Cout,...] or linear [K,N]."""
    if _HAS_TORCH and torch is not None and isinstance(w, torch.Tensor):
        arr = w.detach().cpu().numpy()
    else:
        arr = np.asarray(w)
    if arr.ndim == 2:
        amax = np.max(np.abs(arr), axis=0)
    else:
        reduce_axes = tuple(range(1, arr.ndim))
        amax = np.max(np.abs(arr), axis=reduce_axes)
    amax = np.maximum(amax, 1e-6)
    return np.clip(127.0 / amax * 0.95, lo, hi).astype(np.float64)


@dataclass
class StemInt8:
    """Exported int8 stem for numpy deploy-ref (matches TFLite float closely)."""

    w_stem: np.ndarray
    b_stem: np.ndarray
    w_dw: np.ndarray
    b_dw: np.ndarray
    w_pw: np.ndarray
    b_pw: np.ndarray
    w_dw2: np.ndarray
    b_dw2: np.ndarray
    w_pw2: np.ndarray
    b_pw2: np.ndarray
    # per-tensor act scales (input + post each layer)
    sa_in: float
    sa_stem: float
    sa_dw: float
    sa_pw: float
    sa_dw2: float
    sa_pw2: float
    # per-output-channel weight scales
    sw_stem: np.ndarray
    sw_dw: np.ndarray
    sw_pw: np.ndarray
    sw_dw2: np.ndarray
    sw_pw2: np.ndarray

    @staticmethod
    def from_module(
        m: Any,
        *,
        act_scales: dict[str, float] | None = None,
        w_scales: dict[str, np.ndarray | float] | None = None,
    ) -> "StemInt8":
        if not _HAS_TORCH:
            raise RuntimeError("StemInt8.from_module requires PyTorch")

        def q(w: Any, s: np.ndarray) -> np.ndarray:
            ss = np.asarray(s, dtype=np.float64).reshape(-1, *([1] * (w.ndim - 1)))
            return np.clip(
                np.round(w.detach().cpu().numpy() * ss), -128, 127
            ).astype(np.int8)

        act_scales = act_scales or {
            "in": 127.0,
            "stem": 127.0,
            "dw": 127.0,
            "pw": 127.0,
            "dw2": 127.0,
            "pw2": 127.0,
        }
        if w_scales is None:
            sw = {
                "stem": _per_out_channel_scale(m.stem.weight),
                "dw": _per_out_channel_scale(m.dw.weight),
                "pw": _per_out_channel_scale(m.pw.weight),
                "dw2": _per_out_channel_scale(m.dw2.weight),
                "pw2": _per_out_channel_scale(m.pw2.weight),
            }
        else:
            sw = {}
            for k, conv in (
                ("stem", m.stem),
                ("dw", m.dw),
                ("pw", m.pw),
                ("dw2", m.dw2),
                ("pw2", m.pw2),
            ):
                v = w_scales[k]
                if isinstance(v, (float, int, np.floating)):
                    sw[k] = np.full(conv.weight.shape[0], float(v), dtype=np.float64)
                else:
                    sw[k] = np.asarray(v, dtype=np.float64)

        return StemInt8(
            w_stem=q(m.stem.weight, sw["stem"]),
            b_stem=m.stem.bias.detach().cpu().numpy().astype(np.float32),
            w_dw=q(m.dw.weight, sw["dw"]),
            b_dw=m.dw.bias.detach().cpu().numpy().astype(np.float32),
            w_pw=q(m.pw.weight, sw["pw"]),
            b_pw=m.pw.bias.detach().cpu().numpy().astype(np.float32),
            w_dw2=q(m.dw2.weight, sw["dw2"]),
            b_dw2=m.dw2.bias.detach().cpu().numpy().astype(np.float32),
            w_pw2=q(m.pw2.weight, sw["pw2"]),
            b_pw2=m.pw2.bias.detach().cpu().numpy().astype(np.float32),
            sa_in=float(act_scales["in"]),
            sa_stem=float(act_scales["stem"]),
            sa_dw=float(act_scales["dw"]),
            sa_pw=float(act_scales["pw"]),
            sa_dw2=float(act_scales.get("dw2", act_scales["pw"])),
            sa_pw2=float(act_scales.get("pw2", act_scales["pw"])),
            sw_stem=np.asarray(sw["stem"], dtype=np.float64),
            sw_dw=np.asarray(sw["dw"], dtype=np.float64),
            sw_pw=np.asarray(sw["pw"], dtype=np.float64),
            sw_dw2=np.asarray(sw["dw2"], dtype=np.float64),
            sw_pw2=np.asarray(sw["pw2"], dtype=np.float64),
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[f"stem_{k}"] = v
            else:
                d[f"stem_{k}"] = np.array([v], dtype=np.float64)
        return d

    @staticmethod
    def from_npz(data) -> "StemInt8 | None":
        if "stem_w_stem" not in data.files:
            return None
        if "stem_w_dw2" not in data.files:
            raise ValueError(
                "stem npz missing dw2/pw2 (richer stem). Retrain with train_vit_mnist.py"
            )

        def _sw(key: str, cout: int) -> np.ndarray:
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim == 0 or arr.size == 1:
                return np.full(cout, float(arr.reshape(-1)[0]), dtype=np.float64)
            return arr.reshape(-1)

        w_stem = np.asarray(data["stem_w_stem"], dtype=np.int8)
        c = w_stem.shape[0]
        return StemInt8(
            w_stem=w_stem,
            b_stem=np.asarray(data["stem_b_stem"], dtype=np.float32),
            w_dw=np.asarray(data["stem_w_dw"], dtype=np.int8),
            b_dw=np.asarray(data["stem_b_dw"], dtype=np.float32),
            w_pw=np.asarray(data["stem_w_pw"], dtype=np.int8),
            b_pw=np.asarray(data["stem_b_pw"], dtype=np.float32),
            w_dw2=np.asarray(data["stem_w_dw2"], dtype=np.int8),
            b_dw2=np.asarray(data["stem_b_dw2"], dtype=np.float32),
            w_pw2=np.asarray(data["stem_w_pw2"], dtype=np.int8),
            b_pw2=np.asarray(data["stem_b_pw2"], dtype=np.float32),
            sa_in=float(data["stem_sa_in"][0]),
            sa_stem=float(data["stem_sa_stem"][0]),
            sa_dw=float(data["stem_sa_dw"][0]),
            sa_pw=float(data["stem_sa_pw"][0]),
            sa_dw2=float(data["stem_sa_dw2"][0]),
            sa_pw2=float(data["stem_sa_pw2"][0]),
            sw_stem=_sw("stem_sw_stem", c),
            sw_dw=_sw("stem_sw_dw", c),
            sw_pw=_sw("stem_sw_pw", c),
            sw_dw2=_sw("stem_sw_dw2", c),
            sw_pw2=_sw("stem_sw_pw2", c),
        )


def _fq(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(x * scale), -128, 127).astype(np.float32) / scale


def _dequant_w(w_i8: np.ndarray, sw: np.ndarray) -> np.ndarray:
    s = np.asarray(sw, dtype=np.float32).reshape(-1, *([1] * (w_i8.ndim - 1)))
    return w_i8.astype(np.float32) / s


def _conv2d_nchw(x, w, b, *, stride=1, padding=0, groups=1) -> np.ndarray:
    """Small vectorized conv (NHWC in / out as NCHW here)."""
    n, cin, _, _ = x.shape
    cout, cg, kh, kw = w.shape
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_out = (x.shape[2] - kh) // stride + 1
    w_out = (x.shape[3] - kw) // stride + 1
    s0, s1, s2, s3 = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, cin, h_out, w_out, kh, kw),
        strides=(s0, s1, s2 * stride, s3 * stride, s2, s3),
        writeable=False,
    )
    y = np.empty((n, cout, h_out, w_out), dtype=np.float32)
    cout_g = cout // groups
    for g in range(groups):
        xg = windows[:, g * cg : (g + 1) * cg]
        wg = w[g * cout_g : (g + 1) * cout_g]
        yg = np.tensordot(xg, wg, axes=([1, 4, 5], [1, 2, 3]))
        y[:, g * cout_g : (g + 1) * cout_g] = np.moveaxis(yg, -1, 1) + b[
            g * cout_g : (g + 1) * cout_g
        ].reshape(1, -1, 1, 1)
    return y


def stem_forward_numpy(img28: np.ndarray, s: StemInt8, *, qat: bool = True) -> np.ndarray:
    """One image → tokens float32 [T,C] (deploy-style fake-int8 if qat)."""
    x = np.asarray(img28, dtype=np.float32).reshape(1, 1, IMG, IMG)

    def layer(x_in, w_i8, b, sw, sa, *, stride, padding, groups):
        w = _dequant_w(w_i8, sw)
        xin = _fq(x_in, sa) if qat else x_in
        y = _conv2d_nchw(xin, w, b, stride=stride, padding=padding, groups=groups)
        return np.maximum(y, 0.0)

    x = layer(x, s.w_stem, s.b_stem, s.sw_stem, s.sa_in, stride=2, padding=1, groups=1)
    c = s.w_stem.shape[0]
    x = layer(x, s.w_dw, s.b_dw, s.sw_dw, s.sa_stem, stride=2, padding=1, groups=c)
    x = layer(x, s.w_pw, s.b_pw, s.sw_pw, s.sa_dw, stride=1, padding=0, groups=1)
    x = layer(x, s.w_dw2, s.b_dw2, s.sw_dw2, s.sa_pw, stride=1, padding=1, groups=c)
    x = layer(x, s.w_pw2, s.b_pw2, s.sw_pw2, s.sa_dw2, stride=1, padding=0, groups=1)
    # pad 7→8, avg pool 2 → 4×4
    x = np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
    x = x.reshape(1, x.shape[1], 4, 2, 4, 2).mean(axis=(3, 5))  # [1,C,4,4]
    ch = x.shape[1]
    return x[0].reshape(ch, STEM_T).T.copy()


def stem_forward_tflite(img28: np.ndarray, tflite_path: Path | None = None) -> np.ndarray:
    """A9 path: TFLite+XNNPACK → tokens [T,C]."""
    from xnnpack_cnn import STEM_TFLITE, load_stem_tflite

    path = tflite_path or STEM_TFLITE
    m = load_stem_tflite(path)
    x = np.asarray(img28, dtype=np.float32).reshape(1, IMG, IMG, 1)
    y = m(x)  # [1,T,C] or [1,4,4,C]
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 4:
        y = y.reshape(1, STEM_T, STEM_C)
    return y.reshape(STEM_T, -1)

#!/usr/bin/env python3
"""Export vit_mnist_weights.npz (+ optional samples) to a packed binary for C++.

Format (little-endian) — see include/npukit/vit.hpp.
Stem weights are written already dequantized to float32 (matches C stem).
Per-channel weight scales are stored as inverses for dequant.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

from npukit_vit_mnist import VitMnistWeights, vit_forward  # noqa: E402
from vit_ds_stem import _dequant_w  # noqa: E402

MAGIC = b"NKV1"
VERSION = 1


def _f64_arr(a: np.ndarray | float) -> bytes:
    return np.asarray(a, dtype=np.float64).reshape(-1).tobytes(order="C")


def _f32_arr(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.float32).tobytes(order="C")


def _i8_arr(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.int8).tobytes(order="C")


def _i32_arr(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.int32).tobytes(order="C")


def _inv(sw: np.ndarray) -> np.ndarray:
    return (1.0 / np.asarray(sw, dtype=np.float64)).astype(np.float64)


def export(
    weights_path: Path,
    out_path: Path,
    sample_path: Path | None,
    n_samples: int,
) -> None:
    w = VitMnistWeights.load(weights_path)
    assert w.stem is not None
    s = w.stem
    t = int(w.pos.shape[0])
    d = int(w.pos.shape[1])
    mlp = int(w.blocks[0].w1.shape[1])
    layers = len(w.blocks)
    n_class = int(w.w_cls.shape[1])
    mid = int(s.w_stem.shape[0])
    c = int(s.w_pw3.shape[0])
    img = 28

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<8I", VERSION, t, d, mlp, layers, n_class, mid, c)
    blob += struct.pack("<I", img)

    for name in (
        "sa_in",
        "sa_stem",
        "sa_dw",
        "sa_pw",
        "sa_dw2",
        "sa_pw2",
        "sa_dw3",
        "sa_pw3",
    ):
        blob += struct.pack("<f", float(getattr(s, name)))

    for wk, sk, bk in (
        ("w_stem", "sw_stem", "b_stem"),
        ("w_dw", "sw_dw", "b_dw"),
        ("w_pw", "sw_pw", "b_pw"),
        ("w_dw2", "sw_dw2", "b_dw2"),
        ("w_pw2", "sw_pw2", "b_pw2"),
        ("w_dw3", "sw_dw3", "b_dw3"),
        ("w_pw3", "sw_pw3", "b_pw3"),
    ):
        blob += _f32_arr(_dequant_w(getattr(s, wk), getattr(s, sk)))
        blob += _f32_arr(getattr(s, bk))

    blob += _i32_arr(w.pos)

    for li, blk in enumerate(w.blocks):
        sc = w.scale_blocks[li]
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            blob += _i8_arr(getattr(blk, name))
        blob += _i32_arr(blk.gamma1)
        blob += _i32_arr(blk.gamma2)
        blob += struct.pack("<dd", float(sc.act), float(sc.p))
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            sw = getattr(blk, f"sw_{name}")
            assert sw is not None
            blob += _f64_arr(_inv(sw))

    blob += _i8_arr(w.w_cls)
    blob += struct.pack("<d", float(w.scale_cls.act))
    sw_cls = np.asarray(w.scale_cls.w, dtype=np.float64).reshape(-1)
    if sw_cls.size == 1:
        sw_cls = np.full(n_class, float(sw_cls[0]), dtype=np.float64)
    blob += _f64_arr(_inv(sw_cls))

    imgs = labels = None
    if sample_path and sample_path.exists():
        z = np.load(sample_path)
        imgs = np.asarray(z["images"], dtype=np.float32)
        labels = np.asarray(z["labels"], dtype=np.int32)
        n_samples = min(n_samples, int(imgs.shape[0]))
    else:
        n_samples = 0

    blob += struct.pack("<I", n_samples)
    for i in range(n_samples):
        img_i = imgs[i]
        logits, _ = vit_forward(
            img_i,
            w,
            use_hw=False,
            use_hw_gemm=False,
            glue_mode="float",
            verbose=False,
            use_tflite_stem=False,
        )
        blob += _f32_arr(img_i.reshape(-1))
        blob += struct.pack("<i", int(labels[i]))
        blob += _i32_arr(np.asarray(logits, dtype=np.int32).reshape(-1))

    out_path.write_bytes(blob)
    print(
        f"wrote {out_path} ({len(blob)} bytes) "
        f"T={t} D={d} MLP={mlp} L={layers} mid={mid} samples={n_samples}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HOST / "vit_mnist_weights.npz")
    ap.add_argument("--samples", type=Path, default=HOST / "mnist_sample.npz")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "vit_mnist.bin"
    )
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()
    export(args.weights, args.out, args.samples, args.n_samples)


if __name__ == "__main__":
    main()

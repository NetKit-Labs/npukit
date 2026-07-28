#!/usr/bin/env python3
"""Pack command_lm_weights.npz (+ samples) to NKL1 binary for C++."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

import gsc_commands as gsc  # noqa: E402
from npukit_command_lm import CommandLmWeights  # noqa: E402

MAGIC = b"NKL1"
VERSION = 1


def _f64_arr(a: np.ndarray | float) -> bytes:
    return np.asarray(a, dtype=np.float64).reshape(-1).tobytes(order="C")


def _i8_arr(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.int8).tobytes(order="C")


def _i32_arr(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.int32).tobytes(order="C")


def _inv(sw: np.ndarray) -> np.ndarray:
    return (1.0 / np.asarray(sw, dtype=np.float64)).astype(np.float64)


def export(weights_path: Path, out_path: Path, sample_path: Path | None, n_samples: int) -> None:
    w = CommandLmWeights.load(weights_path)
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack(
        "<7I",
        VERSION,
        w.t,
        w.d,
        w.mlp,
        w.layers,
        w.vocab,
        int(gsc.PAD_ID),
    )
    blob += _i32_arr(w.pos)
    blob += _i8_arr(w.w_emb)
    blob += struct.pack("<d", float(w.scale_emb_act))
    blob += _f64_arr(_inv(w.sw_emb))

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

    blob += _i8_arr(w.w_lm)
    blob += struct.pack("<d", float(w.scale_lm.act))
    blob += _f64_arr(_inv(w.sw_lm))

    xs = ys = None
    if sample_path and sample_path.exists():
        z = np.load(sample_path)
        xs = np.asarray(z["input_ids"], dtype=np.int32)
        ys = np.asarray(z["target_ids"], dtype=np.int32)

    n = 0
    if xs is not None:
        n = min(int(n_samples), xs.shape[0])
    blob += struct.pack("<I", n)
    for i in range(n):
        blob += _i32_arr(xs[i])
        blob += _i32_arr(ys[i])

    out_path.write_bytes(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, {n} samples)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=HOST / "command_lm_weights.npz")
    ap.add_argument("--sample", type=Path, default=HOST / "command_lm_sample.npz")
    ap.add_argument("--out", type=Path, default=Path("command_lm.bin"))
    ap.add_argument("--n-samples", type=int, default=32)
    args = ap.parse_args()
    export(args.weights, args.out, args.sample, args.n_samples)


if __name__ == "__main__":
    main()

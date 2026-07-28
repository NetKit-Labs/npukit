#!/usr/bin/env python3
"""Pack dscnn_mnist_int8.npz (+ samples) for the C++ DS-CNN peer.

Format (little-endian) — see include/npukit/dscnn.hpp.
Weights are stored dequantized float32 (w_i8 / sw).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))

from edge_peer_diag import dscnn_int8_numpy_forward  # noqa: E402

MAGIC = b"NKD1"
VERSION = 1

# Layer order matches dscnn_int8_numpy_forward
LAYERS = (
    # name, act_key, cout, cin, kh, kw, groups (for shape checks)
    ("stem", "in", 16, 1, 3, 3),
    ("b1_dw", "stem", 16, 1, 3, 3),
    ("b1_pw", "b1_dw", 32, 16, 1, 1),
    ("b2_dw", "b1_pw", 32, 1, 3, 3),
    ("b2_pw", "b2_dw", 64, 32, 1, 1),
    ("b3_dw", "b2_pw", 64, 1, 3, 3),
    ("b3_pw", "b3_dw", 64, 64, 1, 1),
)


def export(weights: Path, samples: Path | None, out: Path, n_samples: int) -> None:
    z = np.load(weights)
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<II", VERSION, 28)  # img

    # act scales used as inputs to each layer + pooled→fc
    for key in ("in", "stem", "b1_dw", "b1_pw", "b2_dw", "b2_pw", "b3_dw", "b3_pw"):
        blob += struct.pack("<f", float(z[f"sa_{key}"][0]))

    for name, _act, cout, cin, kh, kw in LAYERS:
        w = z[f"w_{name}"].astype(np.float32) / float(z[f"sw_{name}"][0])
        b = z[f"b_{name}"].astype(np.float32)
        assert w.shape == (cout, cin, kh, kw), (name, w.shape)
        assert b.shape == (cout,)
        blob += w.tobytes(order="C")
        blob += b.tobytes(order="C")

    # FC: store W as [64, 10] = w_fc.T for x @ W + b
    sw = float(z["sw_fc"][0])
    w_fc = z["w_fc"].astype(np.float32) / sw  # [10, 64]
    w_fc_t = np.ascontiguousarray(w_fc.T)  # [64, 10]
    b_fc = z["b_fc"].astype(np.float32)
    blob += w_fc_t.tobytes(order="C")
    blob += b_fc.tobytes(order="C")

    imgs = labels = None
    if samples and samples.exists():
        s = np.load(samples)
        imgs = np.asarray(s["images"], dtype=np.float32)
        labels = np.asarray(s["labels"], dtype=np.int32)
        n_samples = min(n_samples, int(imgs.shape[0]))
    else:
        n_samples = 0

    blob += struct.pack("<I", n_samples)
    for i in range(n_samples):
        logits = dscnn_int8_numpy_forward(imgs[i], z).astype(np.float32).reshape(-1)
        blob += imgs[i].reshape(-1).astype(np.float32).tobytes(order="C")
        blob += struct.pack("<i", int(labels[i]))
        blob += logits.tobytes(order="C")

    out.write_bytes(blob)
    print(f"wrote {out} ({len(blob)} bytes) samples={n_samples}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HOST / "dscnn_mnist_int8.npz")
    ap.add_argument("--samples", type=Path, default=HOST / "mnist_sample.npz")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "dscnn_mnist.bin"
    )
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()
    export(args.weights, args.samples, args.out, args.n_samples)


if __name__ == "__main__":
    main()

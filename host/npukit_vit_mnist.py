#!/usr/bin/env python3
"""MNIST tiny-ViT host path for NpuKit (GEMM + glue, no RTL changes).

Geometry (stay at T=8 for now):
  - Resize digit 28→16
  - Patch size 4 → 16 raw patches, pair-average → T=8 tokens
  - Model dim D=8 (matches 8×8 GEMM tile)

Split:
  - CPU: resize, patchify, pair-pool, position add, mean-pool, class head (10-way; not 8-aligned)
  - FPGA: patch-projection GEMM, 1-layer transformer block (GEMM + glue)

Train weights on the host:
  python3 host/train_vit_mnist.py
  → host/vit_mnist_weights.npz , host/mnist_sample.npz

Usage:
  python3 npukit_vit_mnist.py --ref-only
  python3 npukit_vit_mnist.py --ref-only --weights vit_mnist_weights.npz
  python3 npukit_vit_mnist.py /path/to/npukit.bit --weights vit_mnist_weights.npz
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import npukit_transformer as nt

_HOST_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _HOST_DIR / "vit_mnist_weights.npz"
DEFAULT_SAMPLE = _HOST_DIR / "mnist_sample.npz"

# --- ViT-MNIST geometry (host contract) ---
IMG = 16
PATCH = 4
RAW_T = (IMG // PATCH) * (IMG // PATCH)  # 16 raw patches
VIT_T = 8  # pair-average RAW_T → 8 (Softmax len; see glue MAX_LEN=16 load bug)
VIT_D = 8
PATCH_DIM = PATCH * PATCH  # 16
N_CLASS = 10

assert VIT_T <= nt.MAX_LEN and VIT_T % 8 == 0
assert VIT_D % 8 == 0 and PATCH_DIM % 8 == 0


@dataclass
class VitMnistWeights:
    """Patch embed + 1-layer block + linear head (int8 / Q12)."""

    w_pe: np.ndarray  # [PATCH_DIM, D] int8
    pos: np.ndarray  # [T, D] Q12
    block: nt.TinyBlockWeights
    w_cls: np.ndarray  # [D, N_CLASS] int8

    @staticmethod
    def make(rng: np.random.Generator) -> "VitMnistWeights":
        w_pe = nt.quant_weight_to_i8(rng.normal(0.0, 0.12, size=(PATCH_DIM, VIT_D)))
        pos = nt.to_q12(rng.normal(0.0, 0.05, size=(VIT_T, VIT_D)))
        block = nt.TinyBlockWeights.make(rng, VIT_D)
        w_cls = nt.quant_weight_to_i8(rng.normal(0.0, 0.12, size=(VIT_D, N_CLASS)))
        return VitMnistWeights(w_pe=w_pe, pos=pos, block=block, w_cls=w_cls)

    @staticmethod
    def load(path: str | Path) -> "VitMnistWeights":
        data = np.load(path)
        block = nt.TinyBlockWeights(
            wq=np.asarray(data["wq"], dtype=np.int8),
            wk=np.asarray(data["wk"], dtype=np.int8),
            wv=np.asarray(data["wv"], dtype=np.int8),
            wo=np.asarray(data["wo"], dtype=np.int8),
            w1=np.asarray(data["w1"], dtype=np.int8),
            w2=np.asarray(data["w2"], dtype=np.int8),
            gamma1=np.asarray(data["gamma1"], dtype=np.int32),
            gamma2=np.asarray(data["gamma2"], dtype=np.int32),
        )
        return VitMnistWeights(
            w_pe=np.asarray(data["w_pe"], dtype=np.int8),
            pos=np.asarray(data["pos"], dtype=np.int32),
            block=block,
            w_cls=np.asarray(data["w_cls"], dtype=np.int8),
        )


def resize_nearest(img: np.ndarray, size: int = IMG) -> np.ndarray:
    """Nearest-neighbor resize HxW → size×size (no PIL dependency)."""
    img = np.asarray(img, dtype=np.float64)
    assert img.ndim == 2
    h, w = img.shape
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return img[ys][:, xs]


def patchify(img16: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """[H,W] → [T, P*P] row-major patches."""
    img16 = np.asarray(img16, dtype=np.float64)
    h, w = img16.shape
    assert h == w == IMG and h % patch == 0
    gh = h // patch
    return (
        img16.reshape(gh, patch, gh, patch)
        .transpose(0, 2, 1, 3)
        .reshape(gh * gh, patch * patch)
    )


def synthetic_digit(rng: np.random.Generator, label: int, size: int = 28) -> np.ndarray:
    """Crude MNIST-sized blob so offline runs need no dataset download."""
    img = np.zeros((size, size), dtype=np.float64)
    cy, cx = size // 2, size // 2
    ry = 4 + (label % 5)
    rx = 3 + ((label * 3) % 5)
    yy, xx = np.ogrid[:size, :size]
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    img[mask] = 0.75 + 0.2 * rng.random()
    if label in (1, 7):
        img[:] = 0.0
        img[:, cx - 1 : cx + 2] = 0.9
        if label == 7:
            img[3:6, cx - 4 : cx + 5] = 0.9
    img += 0.05 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


def load_or_synth_batch(
    n: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return images [N,28,28] float and labels [N]. Prefers real MNIST if present."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, N_CLASS, size=n)
    candidates = [
        DEFAULT_SAMPLE,
        Path("mnist_sample.npz"),
        Path("/home/xilinx/jupyter_notebooks/mnist_sample.npz"),
    ]
    for path in candidates:
        try:
            data = np.load(path)
            imgs = data["images"].astype(np.float64)
            labs = data["labels"].astype(int)
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
            if len(labs) > n:
                idx = rng.choice(len(labs), size=n, replace=False)
                imgs, labs = imgs[idx], labs[idx]
            else:
                imgs, labs = imgs[:n], labs[:n]
            print(f"loaded MNIST sample from {path} (n={len(labs)})")
            return imgs, labs
        except (FileNotFoundError, OSError, KeyError):
            pass

    imgs = np.stack([synthetic_digit(rng, int(lbl)) for lbl in labels], axis=0)
    print(
        f"using synthetic digits (n={n}); "
        "run host/train_vit_mnist.py to create mnist_sample.npz"
    )
    return imgs, labels


def pool_patches_to_t8(patches: np.ndarray) -> np.ndarray:
    """Average adjacent raw patches: [16, P] → [8, P] (whole-image coverage)."""
    p = np.asarray(patches, dtype=np.float64)
    assert p.shape[0] == RAW_T
    return 0.5 * (p[0::2] + p[1::2])


def embed_patches_q12(
    patches: np.ndarray,
    w: VitMnistWeights,
    *,
    glue=None,
    mmio=None,
    transport=None,
    use_hw: bool,
) -> np.ndarray:
    """Patch linear (GEMM) + position residual (glue if HW)."""
    x_q12 = nt.to_q12(np.asarray(patches, dtype=np.float64))
    tokens = nt._matmul_q12(x_q12, w.w_pe, mmio=mmio, transport=transport, use_hw=use_hw)
    return nt._residual_rows(glue, tokens, w.pos, use_hw=use_hw)


def classify_tokens_cpu(tokens_q12: np.ndarray, w: VitMnistWeights) -> np.ndarray:
    """Mean-pool + linear head on CPU (N_CLASS=10 is not a multiple of 8)."""
    pooled_q12 = np.rint(tokens_q12.astype(np.float64).mean(axis=0)).astype(np.int32)
    a_i8 = nt.quant_q12_to_i8(pooled_q12.reshape(1, -1), nt.SCALE_ACT)
    return nt.gemm_i8_to_q12(
        a_i8, w.w_cls, a_scale=nt.SCALE_ACT, b_scale=nt.SCALE_W
    ).reshape(-1)


def vit_forward(
    img28: np.ndarray,
    w: VitMnistWeights,
    *,
    glue=None,
    mmio=None,
    transport=None,
    use_hw: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One image → logits Q12 [10] + dump."""
    img16 = resize_nearest(img28, IMG)
    raw = patchify(img16, PATCH)
    patches = pool_patches_to_t8(raw)
    dump: dict[str, np.ndarray] = {"img16": img16, "patches_raw": raw, "patches": patches}

    tokens = embed_patches_q12(
        patches, w, glue=glue, mmio=mmio, transport=transport, use_hw=use_hw
    )
    dump["tokens"] = tokens.copy()

    y, block_dump = nt.transformer_block_1layer(
        tokens,
        w.block,
        glue=glue,
        mmio=mmio,
        transport=transport,
        use_hw=use_hw,
        verbose=verbose,
    )
    dump.update({f"block.{k}": v for k, v in block_dump.items()})
    dump["tokens_out"] = y.copy()

    logits = classify_tokens_cpu(y, w)
    dump["logits"] = logits.copy()
    dump["pred"] = np.array([int(np.argmax(logits))], dtype=np.int32)
    return logits, dump


def run_vit_smoke(
    *,
    bit_path: str | None = None,
    seed: int = 0,
    n: int = 2,
    verbose: bool = False,
    weights_path: str | Path | None | bool = None,
) -> int:
    rng = np.random.default_rng(seed)
    # weights_path=False → force random; None → auto-load default npz if present
    if weights_path is False:
        w = VitMnistWeights.make(rng)
        wsrc = "random-seeded"
    else:
        path = weights_path
        if path is None and DEFAULT_WEIGHTS.exists():
            path = DEFAULT_WEIGHTS
        if path is not None:
            w = VitMnistWeights.load(path)
            wsrc = str(path)
        else:
            w = VitMnistWeights.make(rng)
            wsrc = "random-seeded"
    imgs, labels = load_or_synth_batch(n=n, seed=seed + 1)

    print("=== MNIST tiny-ViT smoke ===")
    print(f"IMG={IMG} PATCH={PATCH} T={VIT_T} D={VIT_D} classes={N_CLASS}")
    print(f"scales ACT/W/P={nt.SCALE_ACT}/{nt.SCALE_W}/{nt.SCALE_P}")
    print(f"weights={wsrc}")

    glue = mmio = transport = None
    if bit_path is not None:
        from npukit_matmul import open_device

        mmio, transport = open_device(bit_path)
        ident = mmio.read(nt.REG_ID)
        ver = mmio.read(nt.REG_VERSION)
        feat = mmio.read(nt.REG_FEATURES)
        print(f"ID=0x{ident:08X} version=0x{ver:08X} features=0x{feat:08X}")
        if ident != nt.ID_MAGIC or ver < nt.VERSION_GLUE or not (feat & nt.FEAT_GLUE):
            print("VIT FAIL: need glue bitstream VERSION>=0x300")
            return 1
        glue = nt.GlueDevice(mmio)

    ok = True
    n_correct_ref = 0
    n_correct_hw = 0
    for i in range(n):
        print(f"\n--- image[{i}] label={int(labels[i])} ---")
        print("--- ref ---")
        logits_ref, dump_ref = vit_forward(imgs[i], w, use_hw=False, verbose=verbose)
        pred_ref = int(dump_ref["pred"][0])
        n_correct_ref += int(pred_ref == int(labels[i]))
        print(f"ref pred={pred_ref} logits_q12[:4]={logits_ref[:4].tolist()}")

        if bit_path is None:
            continue

        print("--- FPGA ---")
        logits_hw, dump_hw = vit_forward(
            imgs[i],
            w,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=True,
            verbose=verbose,
        )
        pred_hw = int(dump_hw["pred"][0])
        n_correct_hw += int(pred_hw == int(labels[i]))
        print(f"hw  pred={pred_hw} logits_q12[:4]={logits_hw[:4].tolist()}")

        for key, tol in (
            ("tokens", 512),
            ("block.y_out", 1024),
            ("logits", 1024),
        ):
            err = int(
                np.max(
                    np.abs(
                        dump_ref[key].astype(np.int64) - dump_hw[key].astype(np.int64)
                    )
                )
            )
            passed = err <= tol
            ok &= passed
            print(f"{key}: {'PASS' if passed else 'FAIL'}  max|err|={err}  tol={tol}")

    print(
        f"\nref accuracy on this batch: {n_correct_ref}/{n} "
        f"({100.0 * n_correct_ref / max(n, 1):.1f}%)"
    )
    if bit_path is not None:
        print(
            f"hw  accuracy on this batch: {n_correct_hw}/{n} "
            f"({100.0 * n_correct_hw / max(n, 1):.1f}%)"
        )

    if bit_path is None:
        print("\nVIT REF-ONLY PASS")
        return 0

    print("\nALL VIT PASS" if ok else "\nVIT FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NpuKit MNIST tiny-ViT host smoke")
    p.add_argument("bit", nargs="?", help="path to npukit.bit (omit with --ref-only)")
    p.add_argument("--ref-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-n", type=int, default=2, help="images to run")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--weights",
        default=None,
        help="path to vit_mnist_weights.npz (default: auto if present)",
    )
    p.add_argument("--random-weights", action="store_true", help="ignore saved weights")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    weights: str | Path | None | bool
    weights = False if args.random_weights else args.weights
    if args.ref_only or not args.bit:
        return run_vit_smoke(
            bit_path=None,
            seed=args.seed,
            n=args.n,
            verbose=args.verbose,
            weights_path=weights,
        )
    return run_vit_smoke(
        bit_path=args.bit,
        seed=args.seed,
        n=args.n,
        verbose=args.verbose,
        weights_path=weights,
    )


if __name__ == "__main__":
    raise SystemExit(main())

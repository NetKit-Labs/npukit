#!/usr/bin/env python3
"""Train a float tiny-ViT matching npukit_vit_mnist geometry, then quantize.

Geometry (same as inference): 28→16, patch 4, pair-average → T=8, D=8, 10 classes.

Writes:
  host/vit_mnist_weights.npz   — int8/Q12 weights for board/ref inference
  host/mnist_sample.npz        — small real MNIST slice for PYNQ without full set
  host/data/mnist/             — cached idx files (gitignored)

Usage (on the Docker host; needs torch):
  python3 host/train_vit_mnist.py
  python3 host/train_vit_mnist.py --epochs 5 --device cpu
"""

from __future__ import annotations

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import npukit_transformer as nt
import npukit_vit_mnist as vit

HOST_DIR = Path(__file__).resolve().parent
DATA_DIR = HOST_DIR / "data" / "mnist"
WEIGHTS_PATH = HOST_DIR / "vit_mnist_weights.npz"
SAMPLE_PATH = HOST_DIR / "mnist_sample.npz"

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte.gz",
}


def _download_mnist() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{MNIST_BASE}/{name}"
        print(f"download {url}")
        urllib.request.urlretrieve(url, dest)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _download_mnist()
    x_tr = _read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_tr = _read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz").astype(np.int64)
    x_te = _read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_te = _read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return x_tr, y_tr, x_te, y_te


def preprocess_batch(imgs28: torch.Tensor) -> torch.Tensor:
    """[B,28,28] → patch tokens float [B,T,P*P] after resize/patch/pool."""
    # nearest resize 28→16
    imgs = imgs28.unsqueeze(1)  # [B,1,28,28]
    imgs16 = F.interpolate(imgs, size=(vit.IMG, vit.IMG), mode="nearest").squeeze(1)
    b, h, w = imgs16.shape
    p = vit.PATCH
    gh = h // p
    patches = (
        imgs16.reshape(b, gh, p, gh, p)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, gh * gh, p * p)
    )
    # pair-average → T=8
    return 0.5 * (patches[:, 0::2, :] + patches[:, 1::2, :])


class TinyViT(nn.Module):
    """Float twin of host/FPGA ViT plumbing (1 layer, T=8, D=8)."""

    def __init__(self) -> None:
        super().__init__()
        d = vit.VIT_D
        t = vit.VIT_T
        self.w_pe = nn.Parameter(torch.randn(vit.PATCH_DIM, d) * 0.12)
        self.pos = nn.Parameter(torch.zeros(t, d))
        self.wq = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wk = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wv = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wo = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w1 = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w2 = nn.Parameter(torch.randn(d, d) * 0.12)
        self.gamma1 = nn.Parameter(torch.ones(d))
        self.gamma2 = nn.Parameter(torch.ones(d))
        self.w_cls = nn.Parameter(torch.randn(d, vit.N_CLASS) * 0.12)

    @staticmethod
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * gamma

    def forward(self, imgs28: torch.Tensor) -> torch.Tensor:
        tok = preprocess_batch(imgs28)  # [B,T,P]
        x = tok @ self.w_pe + self.pos  # [B,T,D]
        # attention
        xn = self.rmsnorm(x, self.gamma1)
        q = xn @ self.wq
        k = xn @ self.wk
        v = xn @ self.wv
        scale = 1.0 / (vit.VIT_D**0.5)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        x = x + (attn @ v) @ self.wo
        # FFN
        xn = self.rmsnorm(x, self.gamma2)
        h = F.gelu(xn @ self.w1)
        x = x + h @ self.w2
        pooled = x.mean(dim=1)
        return pooled @ self.w_cls


def _quant_weight(w: torch.Tensor) -> np.ndarray:
    return nt.quant_weight_to_i8(w.detach().cpu().numpy())


def export_weights(model: TinyViT) -> None:
    block = nt.TinyBlockWeights(
        wq=_quant_weight(model.wq),
        wk=_quant_weight(model.wk),
        wv=_quant_weight(model.wv),
        wo=_quant_weight(model.wo),
        w1=_quant_weight(model.w1),
        w2=_quant_weight(model.w2),
        gamma1=nt.to_q12(model.gamma1.detach().cpu().numpy()),
        gamma2=nt.to_q12(model.gamma2.detach().cpu().numpy()),
    )
    np.savez_compressed(
        WEIGHTS_PATH,
        w_pe=_quant_weight(model.w_pe),
        pos=nt.to_q12(model.pos.detach().cpu().numpy()),
        wq=block.wq,
        wk=block.wk,
        wv=block.wv,
        wo=block.wo,
        w1=block.w1,
        w2=block.w2,
        gamma1=block.gamma1,
        gamma2=block.gamma2,
        w_cls=_quant_weight(model.w_cls),
        meta_t=np.array([vit.VIT_T]),
        meta_d=np.array([vit.VIT_D]),
        scale_act=np.array([nt.SCALE_ACT]),
        scale_w=np.array([nt.SCALE_W]),
    )
    print(f"wrote {WEIGHTS_PATH}")


def save_sample(x_te: np.ndarray, y_te: np.ndarray, n: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_te), size=min(n, len(y_te)), replace=False)
    np.savez_compressed(SAMPLE_PATH, images=x_te[idx], labels=y_te[idx])
    print(f"wrote {SAMPLE_PATH} n={len(idx)}")


@torch.no_grad()
def accuracy(model: TinyViT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb).argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    x_tr, y_tr, x_te, y_te = load_mnist()
    save_sample(x_te, y_te, n=args.sample_n, seed=args.seed)

    if args.subset > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_tr), size=args.subset, replace=False)
        x_tr, y_tr = x_tr[idx], y_tr[idx]

    tr_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    te_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

    torch.manual_seed(args.seed)
    model = TinyViT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epochs(tag: str, n_epochs: int, *, qat: bool) -> None:
        nonlocal opt
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            n = 0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                if qat:
                    # Fake-quant linear weights toward int8 export (STE via round trip).
                    with torch.no_grad():
                        for name in ("w_pe", "wq", "wk", "wv", "wo", "w1", "w2", "w_cls"):
                            p = getattr(model, name)
                            q = torch.clamp(torch.round(p * nt.SCALE_W), -128, 127) / nt.SCALE_W
                            p.copy_(q)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * int(yb.numel())
                n += int(yb.numel())
            te_acc = accuracy(model, te_loader, device)
            print(
                f"{tag} epoch {epoch:02d}  loss={total_loss / max(n, 1):.4f}  "
                f"test_acc={te_acc * 100:.2f}%"
            )

    print(
        f"train tiny-ViT T={vit.VIT_T} D={vit.VIT_D} "
        f"epochs={args.epochs}+{args.qat_epochs} train_n={len(y_tr)} device={device}"
    )
    run_epochs("float", args.epochs, qat=False)
    if args.qat_epochs > 0:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3)
        run_epochs("qat", args.qat_epochs, qat=True)

    export_weights(model)
    # quick quantized ref accuracy on a slice
    w = vit.VitMnistWeights.load(WEIGHTS_PATH)
    n_eval = min(512, len(y_te))
    correct = 0
    for i in range(n_eval):
        logits, dump = vit.vit_forward(x_te[i], w, use_hw=False, verbose=False)
        correct += int(dump["pred"][0] == y_te[i])
    print(f"quantized CPU ref accuracy on {n_eval} test imgs: {100 * correct / n_eval:.2f}%")
    print(f"float test accuracy (full): {100 * accuracy(model, te_loader, device):.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train MNIST tiny-ViT for NpuKit")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--qat-epochs", type=int, default=0, help="optional fake-quant fine-tune epochs")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=0, help="0 = full train set")
    p.add_argument("--sample-n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())

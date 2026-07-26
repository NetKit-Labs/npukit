#!/usr/bin/env python3
"""Train tiny-ViT for NpuKit with scale calibration + STE QAT.

Geometry: native 28×28, patch 7 → T=16, patch vec 49→pad56, D=16,
N_LAYERS host-scheduled transformer blocks, 10 classes.
Uses full MNIST train (60k) plus light shift augmentation.

Pipeline:
  1) Float warm-up
  2) Calibrate scale_act / scale_w / scale_p from activation & weight ranges
  3) QAT with fake-int8 matmuls + Q12 grid snap (STE)
  4) Export int8/Q12 weights + calibrated scales
  5) Report numpy quantized-path accuracy (same as board ref)

Usage:
  python3 host/train_vit_mnist.py
  python3 host/train_vit_mnist.py --epochs 10 --qat-epochs 8
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
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


class FakeQuantSTE(torch.autograd.Function):
    """round(clamp(x*scale))/scale with identity backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, qmin: float, qmax: float):
        y = torch.clamp(torch.round(x * scale), qmin, qmax) / scale
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def fake_quant(x: torch.Tensor, scale: float, qmin: float = -128.0, qmax: float = 127.0) -> torch.Tensor:
    return FakeQuantSTE.apply(x, float(scale), qmin, qmax)


def fake_q12(x: torch.Tensor) -> torch.Tensor:
    """Snap to Q12 grid (matches host to_q12/from_q12)."""
    return FakeQuantSTE.apply(x, float(nt.ONE_Q12), -2_147_483_648.0, 2_147_483_647.0)


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


def augment_batch(imgs28: torch.Tensor, *, max_shift: int = 2) -> torch.Tensor:
    """Cheap MNIST aug: random integer translate (zero pad). Full 60k + shifts ≈ more data."""
    if max_shift <= 0:
        return imgs28
    b, h, w = imgs28.shape
    device = imgs28.device
    out = torch.zeros_like(imgs28)
    dy = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
    dx = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
    for i in range(b):
        y0, x0 = int(dy[i]), int(dx[i])
        src_y0, src_y1 = max(0, -y0), h - max(0, y0)
        dst_y0, dst_y1 = max(0, y0), h - max(0, -y0)
        src_x0, src_x1 = max(0, -x0), w - max(0, x0)
        dst_x0, dst_x1 = max(0, x0), w - max(0, -x0)
        out[i, dst_y0:dst_y1, dst_x0:dst_x1] = imgs28[i, src_y0:src_y1, src_x0:src_x1]
    return out


def preprocess_batch(imgs28: torch.Tensor) -> torch.Tensor:
    """[B,28,28] → padded patch tokens [B,T,PATCH_DIM] (no resize)."""
    assert imgs28.shape[-2:] == (vit.IMG, vit.IMG)
    b = imgs28.shape[0]
    p = vit.PATCH
    gh = vit.IMG // p
    patches = (
        imgs28.reshape(b, gh, p, gh, p)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, gh * gh, p * p)
    )
    if vit.PATCH_DIM == vit.PATCH_DIM_RAW:
        return patches
    out = torch.zeros(b, vit.VIT_T, vit.PATCH_DIM, dtype=patches.dtype, device=patches.device)
    out[..., : vit.PATCH_DIM_RAW] = patches
    return out


class TransformerBlock(nn.Module):
    """One pre-norm attn + FFN block (matches host transformer_block_1layer)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.wq = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wk = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wv = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wo = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w1 = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w2 = nn.Parameter(torch.randn(d, d) * 0.12)
        self.gamma1 = nn.Parameter(torch.ones(d))
        self.gamma2 = nn.Parameter(torch.ones(d))

    @staticmethod
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * gamma


class TinyViT(nn.Module):
    """Float / QAT twin of host ViT plumbing (N_LAYERS host-scheduled blocks)."""

    def __init__(self) -> None:
        super().__init__()
        d = vit.VIT_D
        t = vit.VIT_T
        self.w_pe = nn.Parameter(torch.randn(vit.PATCH_DIM, d) * 0.12)
        self.pos = nn.Parameter(torch.zeros(t, d))
        self.blocks = nn.ModuleList(TransformerBlock(d) for _ in range(vit.N_LAYERS))
        self.w_cls = nn.Parameter(torch.randn(d, vit.N_CLASS) * 0.12)
        self.scale_act = float(nt.SCALE_ACT)
        self.scale_w = float(nt.SCALE_W)
        self.scale_p = float(nt.SCALE_P)

    def _linear(self, x: torch.Tensor, w: torch.Tensor, *, qat: bool) -> torch.Tensor:
        if not qat:
            return x @ w
        xq = fake_quant(x, self.scale_act)
        wq = fake_quant(w, self.scale_w)
        return xq @ wq

    def _block(self, x: torch.Tensor, blk: TransformerBlock, *, qat: bool) -> torch.Tensor:
        xn = TransformerBlock.rmsnorm(x, blk.gamma1)
        if qat:
            xn = fake_q12(xn)
        q = self._linear(xn, blk.wq, qat=qat)
        k = self._linear(xn, blk.wk, qat=qat)
        v = self._linear(xn, blk.wv, qat=qat)
        if qat:
            q, k, v = fake_q12(q), fake_q12(k), fake_q12(v)

        scale = 1.0 / (vit.VIT_D**0.5)
        if qat:
            q_s = fake_quant(q, self.scale_act)
            k_s = fake_quant(k, self.scale_act)
            scores = (q_s @ k_s.transpose(-1, -2)) * scale
            scores = fake_q12(scores)
        else:
            scores = (q @ k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        if qat:
            attn_q = fake_quant(attn, self.scale_p, qmin=0.0, qmax=127.0)
            v_q = fake_quant(v, self.scale_act)
            ctx = fake_q12(attn_q @ v_q)
        else:
            ctx = attn @ v
        x = x + self._linear(ctx, blk.wo, qat=qat)
        if qat:
            x = fake_q12(x)

        xn = TransformerBlock.rmsnorm(x, blk.gamma2)
        if qat:
            xn = fake_q12(xn)
        h = F.gelu(self._linear(xn, blk.w1, qat=qat))
        if qat:
            h = fake_q12(h)
        x = x + self._linear(h, blk.w2, qat=qat)
        if qat:
            x = fake_q12(x)
        return x

    def forward(self, imgs28: torch.Tensor, *, qat: bool = False) -> torch.Tensor:
        tok = preprocess_batch(imgs28)
        if qat:
            tok = fake_q12(tok)
        x = self._linear(tok, self.w_pe, qat=qat) + self.pos
        if qat:
            x = fake_q12(x)
        for blk in self.blocks:
            x = self._block(x, blk, qat=qat)
        pooled = x.mean(dim=1)
        if qat:
            pooled = fake_q12(pooled)
        return self._linear(pooled, self.w_cls, qat=qat)


@torch.no_grad()
def calibrate_scales(
    model: TinyViT,
    loader: DataLoader,
    device: torch.device,
    *,
    batches: int = 16,
    pct: float = 99.9,
) -> tuple[float, float, float]:
    """Set scale_act/w/p from observed ranges so |x|*scale ≈ 127 at pct."""
    model.eval()
    act_abs: list[torch.Tensor] = []
    score_abs: list[torch.Tensor] = []
    attn_abs: list[torch.Tensor] = []

    def hook_collect(module, inp, out):
        pass

    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        tok = preprocess_batch(xb)
        x = tok @ model.w_pe + model.pos
        act_abs.append(tok.detach().abs().reshape(-1))
        act_abs.append(x.detach().abs().reshape(-1))

        for blk in model.blocks:
            xn = TransformerBlock.rmsnorm(x, blk.gamma1)
            q = xn @ blk.wq
            k = xn @ blk.wk
            v = xn @ blk.wv
            act_abs.append(xn.detach().abs().reshape(-1))
            act_abs.append(q.detach().abs().reshape(-1))
            act_abs.append(k.detach().abs().reshape(-1))
            act_abs.append(v.detach().abs().reshape(-1))

            scale = 1.0 / (vit.VIT_D**0.5)
            scores = (q @ k.transpose(-1, -2)) * scale
            score_abs.append(scores.detach().abs().reshape(-1))
            attn = torch.softmax(scores, dim=-1)
            attn_abs.append(attn.detach().reshape(-1))
            ctx = attn @ v
            act_abs.append(ctx.detach().abs().reshape(-1))
            x = x + ctx @ blk.wo
            act_abs.append(x.detach().abs().reshape(-1))

            xn = TransformerBlock.rmsnorm(x, blk.gamma2)
            h = F.gelu(xn @ blk.w1)
            act_abs.append(xn.detach().abs().reshape(-1))
            act_abs.append(h.detach().abs().reshape(-1))
            x = x + h @ blk.w2
            act_abs.append(x.detach().abs().reshape(-1))

        pooled = x.mean(dim=1)
        act_abs.append(pooled.detach().abs().reshape(-1))

        n += 1
        if n >= batches:
            break

    def pct_max(chunks: list[torch.Tensor]) -> float:
        v = torch.cat(chunks).float()
        # torch.quantile has a size cap; subsample large activation dumps
        if v.numel() > 2_000_000:
            idx = torch.randint(0, v.numel(), (2_000_000,), device=v.device)
            v = v.view(-1)[idx]
        return float(torch.quantile(v, pct / 100.0).item())

    a_max = max(pct_max(act_abs), 1e-3)
    w_chunks = [model.w_pe.detach().abs().reshape(-1), model.w_cls.detach().abs().reshape(-1)]
    for blk in model.blocks:
        w_chunks.extend(
            [
                blk.wq.detach().abs().reshape(-1),
                blk.wk.detach().abs().reshape(-1),
                blk.wv.detach().abs().reshape(-1),
                blk.wo.detach().abs().reshape(-1),
                blk.w1.detach().abs().reshape(-1),
                blk.w2.detach().abs().reshape(-1),
            ]
        )
    w_max = max(pct_max(w_chunks), 1e-3)
    p_max = max(pct_max(attn_abs), 1e-3)

    # leave ~5% headroom under 127
    scale_act = min(127.0 / a_max, 1024.0) * 0.95
    scale_w = min(127.0 / w_max, 1024.0) * 0.95
    scale_p = min(127.0 / p_max, 1024.0) * 0.95
    # keep scales in a practical band for int8 GEMM
    scale_act = float(np.clip(scale_act, 8.0, 512.0))
    scale_w = float(np.clip(scale_w, 8.0, 512.0))
    scale_p = float(np.clip(scale_p, 16.0, 512.0))

    model.scale_act = scale_act
    model.scale_w = scale_w
    model.scale_p = scale_p
    print(
        f"calibrated scales: act={scale_act:.2f} (a99.9={a_max:.3f})  "
        f"w={scale_w:.2f} (w99.9={w_max:.3f})  p={scale_p:.2f} (p99.9={p_max:.3f})"
    )
    return scale_act, scale_w, scale_p


def _quant_weight(w: torch.Tensor, scale_w: float) -> np.ndarray:
    return nt.quant_weight_to_i8(w.detach().cpu().numpy(), scale=scale_w)


def export_weights(model: TinyViT) -> None:
    sa, sw, sp = model.scale_act, model.scale_w, model.scale_p
    payload: dict[str, np.ndarray] = {
        "w_pe": _quant_weight(model.w_pe, sw),
        "pos": nt.to_q12(model.pos.detach().cpu().numpy()),
        "w_cls": _quant_weight(model.w_cls, sw),
        "meta_t": np.array([vit.VIT_T]),
        "meta_d": np.array([vit.VIT_D]),
        "meta_layers": np.array([vit.N_LAYERS]),
        "scale_act": np.array([sa], dtype=np.float64),
        "scale_w": np.array([sw], dtype=np.float64),
        "scale_p": np.array([sp], dtype=np.float64),
    }
    for i, blk in enumerate(model.blocks):
        payload[f"wq{i}"] = _quant_weight(blk.wq, sw)
        payload[f"wk{i}"] = _quant_weight(blk.wk, sw)
        payload[f"wv{i}"] = _quant_weight(blk.wv, sw)
        payload[f"wo{i}"] = _quant_weight(blk.wo, sw)
        payload[f"w1{i}"] = _quant_weight(blk.w1, sw)
        payload[f"w2{i}"] = _quant_weight(blk.w2, sw)
        payload[f"gamma1{i}"] = nt.to_q12(blk.gamma1.detach().cpu().numpy())
        payload[f"gamma2{i}"] = nt.to_q12(blk.gamma2.detach().cpu().numpy())
    np.savez_compressed(WEIGHTS_PATH, **payload)
    print(
        f"wrote {WEIGHTS_PATH}  layers={vit.N_LAYERS}  "
        f"scales act/w/p={sa:.2f}/{sw:.2f}/{sp:.2f}"
    )


def save_sample(x_te: np.ndarray, y_te: np.ndarray, n: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_te), size=min(n, len(y_te)), replace=False)
    np.savez_compressed(SAMPLE_PATH, images=x_te[idx], labels=y_te[idx])
    print(f"wrote {SAMPLE_PATH} n={len(idx)}")


@torch.no_grad()
def accuracy(model: TinyViT, loader: DataLoader, device: torch.device, *, qat: bool) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, qat=qat).argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def quant_numpy_accuracy(x_te: np.ndarray, y_te: np.ndarray, n_eval: int) -> float:
    w = vit.VitMnistWeights.load(WEIGHTS_PATH)
    correct = 0
    for i in range(n_eval):
        _, dump = vit.vit_forward(x_te[i], w, use_hw=False, verbose=False)
        correct += int(dump["pred"][0] == y_te[i])
    return correct / max(n_eval, 1)


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
    cal_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)

    torch.manual_seed(args.seed)
    model = TinyViT().to(device)

    def run_epochs(tag: str, n_epochs: int, *, qat: bool, lr: float) -> None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            n = 0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if args.augment:
                    xb = augment_batch(xb, max_shift=args.aug_shift)
                opt.zero_grad(set_to_none=True)
                logits = model(xb, qat=qat)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * int(yb.numel())
                n += int(yb.numel())
            te_acc = accuracy(model, te_loader, device, qat=qat)
            print(
                f"{tag} epoch {epoch:02d}  loss={total_loss / max(n, 1):.4f}  "
                f"test_acc={te_acc * 100:.2f}%  "
                f"scales={model.scale_act:.1f}/{model.scale_w:.1f}/{model.scale_p:.1f}"
            )

    print(
        f"train tiny-ViT T={vit.VIT_T} D={vit.VIT_D} L={vit.N_LAYERS} "
        f"float={args.epochs} qat={args.qat_epochs} train_n={len(y_tr)} "
        f"aug={args.augment}/{args.aug_shift} device={device}"
    )
    run_epochs("float", args.epochs, qat=False, lr=args.lr)
    calibrate_scales(model, cal_loader, device, batches=args.cal_batches)
    if args.qat_epochs > 0:
        run_epochs("qat", args.qat_epochs, qat=True, lr=args.lr * args.qat_lr_mult)
        # Keep scales fixed after the main QAT pass. A second recalibrate on a
        # deeper (2-layer) net shifted ranges without adapting weights and
        # hurt the numpy/FPGA quantized path.

    export_weights(model)
    n_eval = min(args.eval_n, len(y_te))
    qacc = quant_numpy_accuracy(x_te, y_te, n_eval)
    facc = accuracy(model, te_loader, device, qat=False)
    qat_acc = accuracy(model, te_loader, device, qat=True)
    print(f"float test accuracy (full):     {100 * facc:.2f}%")
    print(f"QAT-mode test accuracy (full):  {100 * qat_acc:.2f}%")
    print(f"numpy quantized ref on {n_eval}: {100 * qacc:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train MNIST tiny-ViT for NpuKit (QAT)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--qat-epochs", type=int, default=12)
    p.add_argument("--qat-lr-mult", type=float, default=0.25)
    p.add_argument("--cal-batches", type=int, default=30)
    p.add_argument("--eval-n", type=int, default=2048)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=0, help="0 = full train set (60k)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug-shift", type=int, default=2, help="max ±px translate")
    p.add_argument("--sample-n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())

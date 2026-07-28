#!/usr/bin/env python3
"""Train host-only DS-CNN on MNIST (benchmark vs tiny-ViT).

Not FPGA-mapped. Saves:
  host/dscnn_mnist_weights.pt
  host/dscnn_mnist_int8.npz
  host/dscnn_mnist_metrics.json

Pipeline: float train → BN-fold → calibrate → STE QAT → export int8.

Usage:
  python3 host/train_dscnn_mnist.py
  python3 host/train_dscnn_mnist.py --epochs 6 --qat-epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import dscnn_mnist as dscnn
from train_vit_mnist import augment_batch, load_mnist

HOST_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = HOST_DIR / "dscnn_mnist_weights.pt"
INT8_PATH = HOST_DIR / "dscnn_mnist_int8.npz"
METRICS_PATH = HOST_DIR / "dscnn_mnist_metrics.json"


def train(
    *,
    epochs: int = 6,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    qat_epochs: int = 3,
    qat_lr: float = 3e-4,
) -> dscnn.DscnnMetrics:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dscnn.WEIGHTS_PATH = WEIGHTS_PATH
    dscnn.INT8_PATH = INT8_PATH
    dscnn.METRICS_PATH = METRICS_PATH

    x_tr, y_tr, x_te, y_te = load_mnist()
    tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=False)
    te_loader = DataLoader(te, batch_size=batch, shuffle=False)

    model = dscnn.DSCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(
        f"DS-CNN train dataset=mnist device={device} "
        f"params={dscnn.count_params(model)} epochs={epochs}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb = augment_batch(xb.to(device), max_shift=2)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
            n += int(yb.numel())
        acc = dscnn.evaluate(model, te_loader, device)
        print(f"epoch {ep}/{epochs}  loss={total_loss / max(n, 1):.4f}  test_acc={100 * acc:.2f}%")

    acc_f = dscnn.evaluate(model, te_loader, device)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"saved {WEIGHTS_PATH}")
    print(f"FINAL float test accuracy: {100 * acc_f:.2f}%")

    print("--- host int8 (BN-fold + calibrate + STE QAT) ---")
    _, acc_i = dscnn.build_int8_from_float(
        model,
        tr_loader,
        te_loader,
        device,
        qat_epochs=qat_epochs,
        qat_lr=qat_lr,
    )
    print(f"FINAL int8 test accuracy: {100 * acc_i:.2f}%")

    m = dscnn.DscnnMetrics(
        test_acc_float=float(acc_f),
        test_acc_int8=float(acc_i),
        n_params=dscnn.count_params(model),
        n_test=int(len(y_te)),
        epochs=epochs,
        qat_epochs=qat_epochs,
        notes=(
            "host float + int8 DS-CNN on mnist (BN-fold, per-tensor fake-int8, STE QAT); "
            "not FPGA-mapped; peer vs tiny-ViT deploy-quant"
        ),
    )
    dscnn.save_metrics(m, METRICS_PATH)
    print(f"saved {METRICS_PATH}")
    return m


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train host DS-CNN MNIST reference")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qat-epochs", type=int, default=3)
    p.add_argument("--qat-lr", type=float, default=3e-4)
    args = p.parse_args(argv)
    train(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        qat_epochs=args.qat_epochs,
        qat_lr=args.qat_lr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

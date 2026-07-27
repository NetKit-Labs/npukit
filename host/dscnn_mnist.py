#!/usr/bin/env python3
"""Host MCU-class DS-CNN reference on MNIST (edge peer for tiny-ViT).

Mimics a TinyML depthwise-separable CNN you’d run on a Cortex-M /
TFLite Micro–class MCU. **Not** a ViT CNN stem and **not** on the FPGA.

Compare against `npukit_vit_mnist` as the **MCU/MPU + accelerator** peer
(accuracy + KiB + where compute runs) — not a param-matched bake-off.

Architecture (TinyML-style, 28×28 → 10):
  Conv → (DW+PW)×3 with strides → GAP → Linear

Usage:
  python3 host/train_dscnn_mnist.py
  python3 host/dscnn_mnist.py                 # float + int8 eval
  python3 host/dscnn_mnist.py --skip-int8     # float only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HOST_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = HOST_DIR / "dscnn_mnist_weights.pt"
INT8_PATH = HOST_DIR / "dscnn_mnist_int8.npz"
METRICS_PATH = HOST_DIR / "dscnn_mnist_metrics.json"
VIT_WEIGHTS = HOST_DIR / "vit_mnist_weights.npz"

IMG = 28
N_CLASS = 10
LAYER_NAMES = ("stem", "b1_dw", "b1_pw", "b2_dw", "b2_pw", "b3_dw", "b3_pw", "fc")


class FakeQuantSTE(torch.autograd.Function):
    """round(clamp(x*scale))/scale with identity backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, qmin: float, qmax: float):
        y = torch.clamp(torch.round(x * scale), qmin, qmax) / scale
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def fake_quant(
    x: torch.Tensor, scale: float, *, qmin: float = -128.0, qmax: float = 127.0
) -> torch.Tensor:
    return FakeQuantSTE.apply(x, float(scale), qmin, qmax)


def scale_from_amax(amax: float, *, qmax: float = 127.0) -> float:
    return float(qmax) / max(float(amax), 1e-8)


# ---------------------------------------------------------------------------
# Float model (train)
# ---------------------------------------------------------------------------


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin: int, cout: int, *, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class DSCNN(nn.Module):
    """Small depthwise-separable CNN for MNIST (host float reference)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparable(16, 32, stride=2)  # 14×14
        self.block2 = DepthwiseSeparable(32, 64, stride=2)  # 7×7
        self.block3 = DepthwiseSeparable(64, 64, stride=1)  # 7×7
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, N_CLASS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# BN-folded float + host int8 (fake-int8 / exported int8)
# ---------------------------------------------------------------------------


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return folded weight [Cout,Cin/groups,kH,kW] and bias [Cout]."""
    assert bn.running_mean is not None and bn.running_var is not None
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w = conv.weight * scale.reshape(-1, 1, 1, 1)
    b = beta - mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w.detach().clone(), b.detach().clone()


class FoldedDSCNN(nn.Module):
    """BN-folded DS-CNN with optional per-tensor fake-int8 (host deploy peer)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 16, 3, padding=1)
        self.b1_dw = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16)
        self.b1_pw = nn.Conv2d(16, 32, 1)
        self.b2_dw = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32)
        self.b2_pw = nn.Conv2d(32, 64, 1)
        self.b3_dw = nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64)
        self.b3_pw = nn.Conv2d(64, 64, 1)
        self.fc = nn.Linear(64, N_CLASS)
        # input + post-activation scales (fc uses pooled features as "act")
        self.act_scales: dict[str, float] = {n: 1.0 for n in ("in",) + LAYER_NAMES}
        self.w_scales: dict[str, float] = {n: 1.0 for n in LAYER_NAMES}

    def forward(self, x: torch.Tensor, *, qat: bool = False) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        def run_conv(x_in: torch.Tensor, conv: nn.Conv2d, act_key: str, w_key: str) -> torch.Tensor:
            if not qat:
                y = conv(x_in)
            else:
                xq = fake_quant(x_in, self.act_scales[act_key])
                wq = fake_quant(conv.weight, self.w_scales[w_key])
                y = F.conv2d(
                    xq,
                    wq,
                    None,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                )
                if conv.bias is not None:
                    y = y + conv.bias.reshape(1, -1, 1, 1)
            return F.relu(y, inplace=False)

        x = run_conv(x, self.stem, "in", "stem")
        x = run_conv(x, self.b1_dw, "stem", "b1_dw")
        x = run_conv(x, self.b1_pw, "b1_dw", "b1_pw")
        x = run_conv(x, self.b2_dw, "b1_pw", "b2_dw")
        x = run_conv(x, self.b2_pw, "b2_dw", "b2_pw")
        x = run_conv(x, self.b3_dw, "b2_pw", "b3_dw")
        x = run_conv(x, self.b3_pw, "b3_dw", "b3_pw")
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if not qat:
            return self.fc(x)
        xq = fake_quant(x, self.act_scales["b3_pw"])
        wq = fake_quant(self.fc.weight, self.w_scales["fc"])
        y = F.linear(xq, wq, None)
        if self.fc.bias is not None:
            y = y + self.fc.bias
        return y


@torch.no_grad()
def fold_dscnn(model: DSCNN) -> FoldedDSCNN:
    """Fold BN into convs for deploy-style float / int8."""
    model.eval()
    out = FoldedDSCNN()
    w, b = _fold_conv_bn(model.stem[0], model.stem[1])
    out.stem.weight.copy_(w)
    out.stem.bias.copy_(b)

    for blk, prefix in (
        (model.block1, "b1"),
        (model.block2, "b2"),
        (model.block3, "b3"),
    ):
        dw: nn.Conv2d = getattr(out, f"{prefix}_dw")
        pw: nn.Conv2d = getattr(out, f"{prefix}_pw")
        w, b = _fold_conv_bn(blk.dw, blk.dw_bn)
        dw.weight.copy_(w)
        dw.bias.copy_(b)
        w, b = _fold_conv_bn(blk.pw, blk.pw_bn)
        pw.weight.copy_(w)
        pw.bias.copy_(b)

    out.fc.weight.copy_(model.fc.weight.detach())
    out.fc.bias.copy_(model.fc.bias.detach())
    return out


@torch.no_grad()
def calibrate_folded(
    model: FoldedDSCNN,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 40,
) -> None:
    """Per-tensor symmetric scales from calib amax (inputs + post-ReLU / pooled)."""
    model.eval()
    model.to(device)
    amax: dict[str, float] = {k: 0.0 for k in ("in",) + LAYER_NAMES}

    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        if xb.ndim == 3:
            xb = xb.unsqueeze(1)
        amax["in"] = max(amax["in"], float(xb.abs().max()))
        x = F.relu(model.stem(xb))
        amax["stem"] = max(amax["stem"], float(x.abs().max()))
        x = F.relu(model.b1_dw(x))
        amax["b1_dw"] = max(amax["b1_dw"], float(x.abs().max()))
        x = F.relu(model.b1_pw(x))
        amax["b1_pw"] = max(amax["b1_pw"], float(x.abs().max()))
        x = F.relu(model.b2_dw(x))
        amax["b2_dw"] = max(amax["b2_dw"], float(x.abs().max()))
        x = F.relu(model.b2_pw(x))
        amax["b2_pw"] = max(amax["b2_pw"], float(x.abs().max()))
        x = F.relu(model.b3_dw(x))
        amax["b3_dw"] = max(amax["b3_dw"], float(x.abs().max()))
        x = F.relu(model.b3_pw(x))
        amax["b3_pw"] = max(amax["b3_pw"], float(x.abs().max()))
        # fc act scale = pooled features (same tensor as b3_pw after pool; track pool)
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        amax["fc"] = max(amax["fc"], float(pooled.abs().max()))
        n += 1
        if n >= max_batches:
            break

    # act_scales: scale applied to the *input* of each named layer
    # stem uses "in"; b1_dw uses post-stem; ...; fc uses pooled (= track as b3_pw / fc)
    model.act_scales["in"] = scale_from_amax(amax["in"])
    model.act_scales["stem"] = scale_from_amax(amax["stem"])
    model.act_scales["b1_dw"] = scale_from_amax(amax["b1_dw"])
    model.act_scales["b1_pw"] = scale_from_amax(amax["b1_pw"])
    model.act_scales["b2_dw"] = scale_from_amax(amax["b2_dw"])
    model.act_scales["b2_pw"] = scale_from_amax(amax["b2_pw"])
    model.act_scales["b3_dw"] = scale_from_amax(amax["b3_dw"])
    model.act_scales["b3_pw"] = scale_from_amax(max(amax["b3_pw"], amax["fc"]))
    model.act_scales["fc"] = model.act_scales["b3_pw"]

    for name in LAYER_NAMES:
        mod = getattr(model, name)
        model.w_scales[name] = scale_from_amax(float(mod.weight.detach().abs().max()))


def export_int8(model: FoldedDSCNN, path: Path = INT8_PATH) -> None:
    """Save int8 weights + float biases + scales (host int8 artifact)."""
    payload: dict[str, np.ndarray] = {}
    for name in LAYER_NAMES:
        mod = getattr(model, name)
        sw = model.w_scales[name]
        w_q = torch.clamp(torch.round(mod.weight.detach() * sw), -128, 127).to(torch.int8)
        payload[f"w_{name}"] = w_q.cpu().numpy()
        payload[f"b_{name}"] = mod.bias.detach().cpu().numpy().astype(np.float32)
        payload[f"sw_{name}"] = np.array([sw], dtype=np.float32)
    for key, val in model.act_scales.items():
        payload[f"sa_{key}"] = np.array([val], dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_int8(path: Path = INT8_PATH, device: torch.device | None = None) -> FoldedDSCNN:
    device = device or torch.device("cpu")
    z = np.load(path)
    model = FoldedDSCNN().to(device)
    for name in LAYER_NAMES:
        mod = getattr(model, name)
        sw = float(z[f"sw_{name}"][0])
        w_q = torch.from_numpy(z[f"w_{name}"].astype(np.float32)).to(device)
        mod.weight.data.copy_(w_q / sw)
        mod.bias.data.copy_(torch.from_numpy(z[f"b_{name}"]).to(device))
        model.w_scales[name] = sw
    for key in ("in",) + LAYER_NAMES:
        model.act_scales[key] = float(z[f"sa_{key}"][0])
    model.eval()
    return model


@dataclass
class DscnnMetrics:
    test_acc_float: float
    test_acc_int8: float
    n_params: int
    n_test: int
    epochs: int
    qat_epochs: int = 0
    notes: str = "host float + int8 DS-CNN; not FPGA-mapped"
    # back-compat alias used by older readers
    test_acc: float = field(init=False)

    def __post_init__(self) -> None:
        self.test_acc = self.test_acc_int8 if self.test_acc_int8 >= 0 else self.test_acc_float


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    qat: bool = False,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if isinstance(model, FoldedDSCNN):
            pred = model(xb, qat=qat).argmax(dim=-1)
        else:
            pred = model(xb).argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def load_model(path: Path = WEIGHTS_PATH, device: torch.device | None = None) -> DSCNN:
    device = device or torch.device("cpu")
    model = DSCNN().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def save_metrics(m: DscnnMetrics, path: Path = METRICS_PATH) -> None:
    d = asdict(m)
    path.write_text(json.dumps(d, indent=2) + "\n")


def load_metrics(path: Path = METRICS_PATH) -> dict:
    return json.loads(path.read_text())


def compare_vit_line() -> str:
    if not VIT_WEIGHTS.exists():
        return "ViT weights missing — run train_vit_mnist.py for side-by-side."
    return (
        "ViT (deploy-quantized numpy, full 10k): ~94.28%  "
        f"[weights: {VIT_WEIGHTS.name}]"
    )


def qat_finetune(
    model: FoldedDSCNN,
    tr_loader: DataLoader,
    te_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 3,
    lr: float = 3e-4,
) -> None:
    """Short STE QAT on BN-folded model with fixed calib scales."""
    from train_vit_mnist import augment_batch

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb = augment_batch(xb.to(device), max_shift=2)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, qat=True)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(yb.numel())
            n += int(yb.numel())
        # refresh weight scales from updated floats (acts stay from calib)
        with torch.no_grad():
            for name in LAYER_NAMES:
                mod = getattr(model, name)
                model.w_scales[name] = scale_from_amax(float(mod.weight.detach().abs().max()))
        acc = evaluate(model, te_loader, device, qat=True)
        print(
            f"qat {ep}/{epochs}  loss={total_loss / max(n, 1):.4f}  "
            f"int8_test_acc={100 * acc:.2f}%"
        )


def build_int8_from_float(
    float_model: DSCNN,
    tr_loader: DataLoader,
    te_loader: DataLoader,
    device: torch.device,
    *,
    qat_epochs: int = 3,
    qat_lr: float = 3e-4,
    calib_batches: int = 40,
) -> tuple[FoldedDSCNN, float]:
    folded = fold_dscnn(float_model).to(device)
    calibrate_folded(folded, tr_loader, device, max_batches=calib_batches)
    ptq_acc = evaluate(folded, te_loader, device, qat=True)
    print(f"int8 PTQ (pre-QAT) test_acc={100 * ptq_acc:.2f}%")
    if qat_epochs > 0:
        qat_finetune(
            folded, tr_loader, te_loader, device, epochs=qat_epochs, lr=qat_lr
        )
    export_int8(folded, INT8_PATH)
    print(f"saved {INT8_PATH}")
    int8_acc = evaluate(folded, te_loader, device, qat=True)
    return folded, float(int8_acc)


def run_eval(
    *,
    batch: int = 256,
    skip_int8: bool = False,
    rebuild_int8: bool = False,
    qat_epochs: int = 3,
) -> DscnnMetrics:
    from train_vit_mnist import load_mnist

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr, x_te, y_te = load_mnist()
    tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    te = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True)
    te_loader = DataLoader(te, batch_size=batch, shuffle=False)

    float_model = load_model(WEIGHTS_PATH, device)
    acc_f = evaluate(float_model, te_loader, device)
    n_params = count_params(float_model)

    acc_i = -1.0
    qat_ep = 0
    if not skip_int8:
        if rebuild_int8 or not INT8_PATH.exists():
            _, acc_i = build_int8_from_float(
                float_model,
                tr_loader,
                te_loader,
                device,
                qat_epochs=qat_epochs,
            )
            qat_ep = qat_epochs
        else:
            int8_model = load_int8(INT8_PATH, device)
            acc_i = evaluate(int8_model, te_loader, device, qat=True)
            if METRICS_PATH.exists():
                qat_ep = int(load_metrics().get("qat_epochs", 0))

    print("=== DS-CNN MNIST host reference ===")
    print(f"device={device}  params={n_params}")
    print(f"test accuracy (float, full {len(y_te)}): {100 * acc_f:.2f}%")
    if acc_i >= 0:
        print(f"test accuracy (int8,  full {len(y_te)}): {100 * acc_i:.2f}%")
    print(compare_vit_line())
    print("NOTE: DS-CNN is a separate benchmark model (not a ViT stem, not on FPGA).")
    print("Fair compare: DS-CNN int8 vs ViT deploy-quant (~94.28%).")

    epochs = -1
    if METRICS_PATH.exists():
        epochs = int(load_metrics().get("epochs", -1))

    m = DscnnMetrics(
        test_acc_float=float(acc_f),
        test_acc_int8=float(acc_i),
        n_params=n_params,
        n_test=int(len(y_te)),
        epochs=epochs,
        qat_epochs=qat_ep,
        notes=(
            "host float + int8 DS-CNN (BN-fold, per-tensor fake-int8, STE QAT); "
            "not FPGA-mapped; fair peer vs ViT deploy-quant ~94.28%"
        ),
    )
    save_metrics(m)
    print(f"wrote {METRICS_PATH}")
    return m


def bench_tflite_xnnpack(*, n: int = 64, warmup: int = 4) -> dict:
    """A9 peer latency via TFLite (XNNPACK-backed)."""
    from xnnpack_cnn import DSCNN_TFLITE, load_dscnn_tflite

    sample = HOST_DIR / "mnist_sample.npz"
    if not sample.exists():
        raise FileNotFoundError(sample)
    imgs = np.load(sample)["images"][:n].astype(np.float32)
    labels = np.load(sample)["labels"][:n]
    m = load_dscnn_tflite()
    x0 = imgs[0].reshape(1, 28, 28, 1)
    ms = m.time_ms(x0, warmup=warmup, iters=max(16, warmup))
    correct = 0
    for i in range(len(imgs)):
        y = m(imgs[i].reshape(1, 28, 28, 1)).reshape(-1)
        correct += int(int(y.argmax()) == int(labels[i]))
    out = {
        "backend": m.backend,
        "delegate": m.delegate,
        "ms_per_image": ms,
        "batch_acc": correct / max(len(imgs), 1),
        "n": len(imgs),
        "model": str(DSCNN_TFLITE),
    }
    print("=== DS-CNN TFLite + XNNPACK (A9 peer) ===")
    print(
        f"backend={out['backend']} delegate={out['delegate']}  "
        f"{ms:.2f} ms/img  acc={100 * out['batch_acc']:.1f}% (n={out['n']})"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval host DS-CNN MNIST reference")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument(
        "--rebuild-int8",
        action="store_true",
        help="re-calibrate + QAT from float weights even if int8 npz exists",
    )
    p.add_argument("--qat-epochs", type=int, default=3)
    p.add_argument(
        "--tflite-bench",
        action="store_true",
        help="bench dscnn_mnist.tflite with TFLite/XNNPACK (A9)",
    )
    p.add_argument("--bench-n", type=int, default=64)
    args = p.parse_args(argv)
    if args.tflite_bench:
        bench_tflite_xnnpack(n=args.bench_n)
        return 0
    if not WEIGHTS_PATH.exists():
        print(f"missing {WEIGHTS_PATH} — run: python3 host/train_dscnn_mnist.py")
        return 1
    run_eval(
        batch=args.batch,
        skip_int8=args.skip_int8,
        rebuild_int8=args.rebuild_int8,
        qat_epochs=args.qat_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

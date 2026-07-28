#!/usr/bin/env python3
"""Train tiny-ViT for NpuKit with scale calibration + deploy-faithful STE QAT.

Geometry: native 28×28, richer CPU DS-stem → T=16×D=16, FFN=VIT_MLP,
N_LAYERS host-scheduled transformer blocks (int8 GEMM on FPGA), 10 classes.

Deploy track: per-channel weight scales + A9 float Softmax/RMSNorm/GELU
(GEMM stays int8). No post-QAT scale recal by default.

Pipeline:
  1) Float warm-up
  2) Calibrate per-stage act/p + per-channel weight scales
  3) QAT: fake-int8 + Q12 snap (proxy; float norms/softmax/gelu)
  4) Deploy-faithful fine-tune: CE on board-ref numpy logits, STE grads via proxy
  5) Export int8/Q12 weights + stem + TFLite stem for A9 XNNPACK
  6) Report numpy quantized-path accuracy (full test set by default)

Usage:
  python3 host/train_vit_mnist.py
  python3 host/train_vit_mnist.py --layers 4
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
import vit_ds_stem as stemmod

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


def fake_quant(
    x: torch.Tensor,
    scale: float | torch.Tensor,
    qmin: float = -128.0,
    qmax: float = 127.0,
) -> torch.Tensor:
    """Per-tensor or broadcastable per-channel fake-quant (STE)."""
    if isinstance(scale, torch.Tensor):
        s = scale.to(dtype=x.dtype, device=x.device)
        y = torch.clamp(torch.round(x * s), qmin, qmax) / s
        return y
    return FakeQuantSTE.apply(x, float(scale), qmin, qmax)


def fake_q12(x: torch.Tensor) -> torch.Tensor:
    """Snap to Q12 grid (matches host to_q12/from_q12)."""
    return FakeQuantSTE.apply(x, float(nt.ONE_Q12), -2_147_483_648.0, 2_147_483_647.0)


def _per_channel_scale_linear(w: torch.Tensor, *, lo: float = 8.0, hi: float = 512.0) -> torch.Tensor:
    """Per-output-column scale for linear weight [K, N]."""
    amax = w.detach().abs().amax(dim=0).clamp_min(1e-6)
    return (127.0 / amax * 0.95).clamp(lo, hi)


def _per_channel_scale_conv(w: torch.Tensor, *, lo: float = 8.0, hi: float = 512.0) -> torch.Tensor:
    """Per-output-channel scale for conv weight [Cout, ...]."""
    dims = tuple(range(1, w.ndim))
    amax = w.detach().abs().amax(dim=dims).clamp_min(1e-6)
    return (127.0 / amax * 0.95).clamp(lo, hi)


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


def augment_batch(
    imgs28: torch.Tensor,
    *,
    max_shift: int = 2,
    noise_std: float = 0.0,
    erase_prob: float = 0.0,
    erase_size: int = 4,
) -> torch.Tensor:
    """MNIST aug (no CNN stem / no patch overlap): shift, noise, small erase."""
    out = imgs28
    b, h, w = out.shape
    device = out.device
    if max_shift > 0:
        shifted = torch.zeros_like(out)
        dy = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
        dx = torch.randint(-max_shift, max_shift + 1, (b,), device=device)
        for i in range(b):
            y0, x0 = int(dy[i]), int(dx[i])
            src_y0, src_y1 = max(0, -y0), h - max(0, y0)
            dst_y0, dst_y1 = max(0, y0), h - max(0, -y0)
            src_x0, src_x1 = max(0, -x0), w - max(0, x0)
            dst_x0, dst_x1 = max(0, x0), w - max(0, -x0)
            shifted[i, dst_y0:dst_y1, dst_x0:dst_x1] = out[i, src_y0:src_y1, src_x0:src_x1]
        out = shifted
    if noise_std > 0:
        out = (out + noise_std * torch.randn_like(out)).clamp(0.0, 1.0)
    if erase_prob > 0 and erase_size > 0:
        mask = torch.rand(b, device=device) < erase_prob
        if bool(mask.any()):
            for i in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                y0 = int(torch.randint(0, max(1, h - erase_size + 1), (1,), device=device))
                x0 = int(torch.randint(0, max(1, w - erase_size + 1), (1,), device=device))
                out[i, y0 : y0 + erase_size, x0 : x0 + erase_size] = 0.0
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

    def __init__(self, d: int, mlp_h: int | None = None) -> None:
        super().__init__()
        mh = int(mlp_h if mlp_h is not None else d)
        self.d = d
        self.mlp_h = mh
        self.wq = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wk = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wv = nn.Parameter(torch.randn(d, d) * 0.12)
        self.wo = nn.Parameter(torch.randn(d, d) * 0.12)
        self.w1 = nn.Parameter(torch.randn(d, mh) * 0.12)
        self.w2 = nn.Parameter(torch.randn(mh, d) * 0.12)
        self.gamma1 = nn.Parameter(torch.ones(d))
        self.gamma2 = nn.Parameter(torch.ones(d))
        # Per-output-channel weight scales (filled by calibrate_scales).
        self.sw_wq = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wk = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wv = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_wo = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)
        self.sw_w1 = nn.Parameter(torch.full((mh,), 64.0), requires_grad=False)
        self.sw_w2 = nn.Parameter(torch.full((d,), 64.0), requires_grad=False)

    @staticmethod
    def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x / rms) * gamma


class TinyViT(nn.Module):
    """Float / QAT twin: CPU DS-stem + N_LAYERS FPGA-scheduled blocks."""

    def __init__(self, n_layers: int | None = None) -> None:
        super().__init__()
        d = vit.VIT_D
        t = vit.VIT_T
        mh = vit.VIT_MLP
        self.n_layers = int(n_layers if n_layers is not None else vit.N_LAYERS)
        mid = stemmod.STEM_MID
        self.stem = stemmod.TinyDSStem(d, mid=mid)
        # legacy placeholder so old export keys still exist
        self.w_pe = nn.Parameter(torch.randn(vit.PATCH_DIM, d) * 0.02)
        self.pos = nn.Parameter(torch.zeros(t, d))
        self.blocks = nn.ModuleList(
            TransformerBlock(d, mlp_h=mh) for _ in range(self.n_layers)
        )
        self.w_cls = nn.Parameter(torch.randn(d, vit.N_CLASS) * 0.12)
        self.sw_cls = nn.Parameter(
            torch.full((vit.N_CLASS,), 64.0), requires_grad=False
        )
        self.scale_embed = vit.QuantScales()  # unused with stem; kept for export compat
        self.scale_blocks = [vit.QuantScales() for _ in range(self.n_layers)]
        self.scale_cls = vit.QuantScales()
        # Must match vit_ds_stem.stem_forward_numpy (fake-int8) when qat=True.
        self.stem_act_scales = {
            "in": 127.0,
            "stem": 127.0,
            "dw": 127.0,
            "pw": 127.0,
            "dw2": 127.0,
            "pw2": 127.0,
            "dw3": 127.0,
            "pw3": 127.0,
        }
        self.stem_w_scales: dict[str, torch.Tensor] = {
            "stem": torch.full((mid,), 127.0),
            "dw": torch.full((mid,), 127.0),
            "pw": torch.full((mid,), 127.0),
            "dw2": torch.full((mid,), 127.0),
            "pw2": torch.full((mid,), 127.0),
            "dw3": torch.full((mid,), 127.0),
            "pw3": torch.full((d,), 127.0),
        }

    @property
    def scale_act(self) -> float:
        return self.scale_blocks[0].act

    @property
    def scale_w(self) -> float:
        return self.scale_blocks[0].w

    @property
    def scale_p(self) -> float:
        return self.scale_blocks[0].p

    def _linear(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        *,
        qat: bool,
        scale_act: float,
        scale_w: float | torch.Tensor,
    ) -> torch.Tensor:
        if not qat:
            return x @ w
        xq = fake_quant(x, scale_act)
        # Broadcast per-column scales onto W [K, N].
        if isinstance(scale_w, torch.Tensor):
            wq = fake_quant(w, scale_w.reshape(1, -1))
        else:
            wq = fake_quant(w, scale_w)
        return xq @ wq

    def _stem_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv2d,
        *,
        sa: float,
        sw: float | torch.Tensor,
        qat: bool,
    ) -> torch.Tensor:
        """Conv+ReLU; when qat, match numpy stem_forward_numpy fake-int8."""
        if not qat:
            return F.relu(conv(x), inplace=False)
        xq = fake_quant(x, sa)
        if isinstance(sw, torch.Tensor):
            shape = (sw.numel(),) + (1,) * (conv.weight.ndim - 1)
            wq = fake_quant(conv.weight, sw.reshape(shape))
        else:
            wq = fake_quant(conv.weight, sw)
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

    def _stem_forward(self, imgs28: torch.Tensor, *, qat: bool) -> torch.Tensor:
        """CPU DS-stem → [B,T,D]. QAT path mirrors deploy numpy int8 stem."""
        x = imgs28.unsqueeze(1) if imgs28.ndim == 3 else imgs28
        sa, sw = self.stem_act_scales, self.stem_w_scales
        x = self._stem_conv(
            x, self.stem.stem, sa=sa["in"], sw=sw["stem"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.dw, sa=sa["stem"], sw=sw["dw"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.pw, sa=sa["dw"], sw=sw["pw"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.dw2, sa=sa["pw"], sw=sw["dw2"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.pw2, sa=sa["dw2"], sw=sw["pw2"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.dw3, sa=sa["pw2"], sw=sw["dw3"], qat=qat
        )
        x = self._stem_conv(
            x, self.stem.pw3, sa=sa["dw3"], sw=sw["pw3"], qat=qat
        )
        x = F.pad(x, (0, 1, 0, 1))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x.flatten(2).transpose(1, 2).contiguous()

    def _block(
        self,
        x: torch.Tensor,
        blk: TransformerBlock,
        sc: vit.QuantScales,
        *,
        qat: bool,
    ) -> torch.Tensor:
        # Softmax/RMSNorm/GELU stay float (matches deploy glue_mode='float').
        xn = TransformerBlock.rmsnorm(x, blk.gamma1)
        if qat:
            xn = fake_q12(xn)
        q = self._linear(xn, blk.wq, qat=qat, scale_act=sc.act, scale_w=blk.sw_wq)
        k = self._linear(xn, blk.wk, qat=qat, scale_act=sc.act, scale_w=blk.sw_wk)
        v = self._linear(xn, blk.wv, qat=qat, scale_act=sc.act, scale_w=blk.sw_wv)
        if qat:
            q, k, v = fake_q12(q), fake_q12(k), fake_q12(v)

        scale = 1.0 / (vit.VIT_D**0.5)
        if qat:
            q_s = fake_quant(q, sc.act)
            k_s = fake_quant(k, sc.act)
            scores = fake_q12((q_s @ k_s.transpose(-1, -2)) * scale)
        else:
            scores = (q @ k.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        if qat:
            attn_q = fake_quant(attn, sc.p, qmin=0.0, qmax=127.0)
            v_q = fake_quant(v, sc.act)
            ctx = fake_q12(attn_q @ v_q)
        else:
            ctx = attn @ v
        x = x + self._linear(ctx, blk.wo, qat=qat, scale_act=sc.act, scale_w=blk.sw_wo)
        if qat:
            x = fake_q12(x)

        xn = TransformerBlock.rmsnorm(x, blk.gamma2)
        if qat:
            xn = fake_q12(xn)
        h = F.gelu(self._linear(xn, blk.w1, qat=qat, scale_act=sc.act, scale_w=blk.sw_w1))
        if qat:
            h = fake_q12(h)
        x = x + self._linear(h, blk.w2, qat=qat, scale_act=sc.act, scale_w=blk.sw_w2)
        if qat:
            x = fake_q12(x)
        return x

    def _forward_proxy(self, imgs28: torch.Tensor, *, qat: bool) -> torch.Tensor:
        # CPU DS-stem (QAT↔numpy int8) → tokens; FPGA body = transformer stack.
        x = self._stem_forward(imgs28, qat=qat)
        if qat:
            x = fake_q12(x)
        x = x + self.pos
        if qat:
            x = fake_q12(x)
        for blk, sc in zip(self.blocks, self.scale_blocks):
            x = self._block(x, blk, sc, qat=qat)
        pooled = x.mean(dim=1)
        if qat:
            pooled = fake_q12(pooled)
        return self._linear(
            pooled,
            self.w_cls,
            qat=qat,
            scale_act=self.scale_cls.act,
            scale_w=self.sw_cls,
        )

    def forward(
        self,
        imgs28: torch.Tensor,
        *,
        qat: bool = False,
        deploy_faithful: bool = False,
    ) -> torch.Tensor:
        logits = self._forward_proxy(imgs28, qat=qat)
        if not (qat and deploy_faithful):
            return logits
        # Forward value = board-ref numpy path; grads flow through proxy (STE).
        with torch.no_grad():
            dep = deploy_numpy_logits(self, imgs28)
        return logits + (dep - logits).detach()


def _scale_from_max(a_max: float, *, lo: float, hi: float) -> float:
    return float(np.clip(min(127.0 / max(a_max, 1e-3), 1024.0) * 0.95, lo, hi))


@torch.no_grad()
def calibrate_scales(
    model: TinyViT,
    loader: DataLoader,
    device: torch.device,
    *,
    batches: int = 16,
    pct: float = 99.9,
) -> None:
    """Per-stage scales so |x|*scale ≈ 127 at percentile."""
    model.eval()
    n_layers = len(model.blocks)
    embed_act: list[torch.Tensor] = []
    cls_act: list[torch.Tensor] = []
    block_act: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    block_attn: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    stem_in: list[torch.Tensor] = []
    stem_a_stem: list[torch.Tensor] = []
    stem_a_dw: list[torch.Tensor] = []
    stem_a_pw: list[torch.Tensor] = []
    stem_a_dw2: list[torch.Tensor] = []
    stem_a_pw2: list[torch.Tensor] = []
    stem_a_dw3: list[torch.Tensor] = []
    stem_a_pw3: list[torch.Tensor] = []
    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        x_img = xb.unsqueeze(1) if xb.ndim == 3 else xb
        stem_in.append(x_img.detach().abs().reshape(-1))
        h = F.relu(model.stem.stem(x_img), inplace=False)
        stem_a_stem.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.dw(h), inplace=False)
        stem_a_dw.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.pw(h), inplace=False)
        stem_a_pw.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.dw2(h), inplace=False)
        stem_a_dw2.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.pw2(h), inplace=False)
        stem_a_pw2.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.dw3(h), inplace=False)
        stem_a_dw3.append(h.detach().abs().reshape(-1))
        h = F.relu(model.stem.pw3(h), inplace=False)
        stem_a_pw3.append(h.detach().abs().reshape(-1))
        h = F.pad(h, (0, 1, 0, 1))
        h = F.avg_pool2d(h, kernel_size=2, stride=2)
        tok = h.flatten(2).transpose(1, 2).contiguous()
        x = tok + model.pos
        embed_act.append(tok.detach().abs().reshape(-1))
        embed_act.append(x.detach().abs().reshape(-1))

        for li, blk in enumerate(model.blocks):
            xn = TransformerBlock.rmsnorm(x, blk.gamma1)
            q = xn @ blk.wq
            k = xn @ blk.wk
            v = xn @ blk.wv
            block_act[li].extend(
                [
                    xn.detach().abs().reshape(-1),
                    q.detach().abs().reshape(-1),
                    k.detach().abs().reshape(-1),
                    v.detach().abs().reshape(-1),
                ]
            )
            scale = 1.0 / (vit.VIT_D**0.5)
            scores = (q @ k.transpose(-1, -2)) * scale
            attn = torch.softmax(scores, dim=-1)
            block_attn[li].append(attn.detach().reshape(-1))
            ctx = attn @ v
            block_act[li].append(ctx.detach().abs().reshape(-1))
            x = x + ctx @ blk.wo
            block_act[li].append(x.detach().abs().reshape(-1))

            xn = TransformerBlock.rmsnorm(x, blk.gamma2)
            h = F.gelu(xn @ blk.w1)
            block_act[li].extend(
                [xn.detach().abs().reshape(-1), h.detach().abs().reshape(-1)]
            )
            x = x + h @ blk.w2
            block_act[li].append(x.detach().abs().reshape(-1))

        pooled = x.mean(dim=1)
        cls_act.append(pooled.detach().abs().reshape(-1))

        n += 1
        if n >= batches:
            break

    def pct_max(chunks: list[torch.Tensor]) -> float:
        v = torch.cat(chunks).float()
        if v.numel() > 2_000_000:
            idx = torch.randint(0, v.numel(), (2_000_000,), device=v.device)
            v = v.view(-1)[idx]
        return float(torch.quantile(v, pct / 100.0).item())

    # Stem act/weight scales (must match numpy stem_forward_numpy)
    model.stem_act_scales = {
        "in": _scale_from_max(pct_max(stem_in), lo=8.0, hi=512.0),
        "stem": _scale_from_max(pct_max(stem_a_stem), lo=8.0, hi=512.0),
        "dw": _scale_from_max(pct_max(stem_a_dw), lo=8.0, hi=512.0),
        "pw": _scale_from_max(pct_max(stem_a_pw), lo=8.0, hi=512.0),
        "dw2": _scale_from_max(pct_max(stem_a_dw2), lo=8.0, hi=512.0),
        "pw2": _scale_from_max(pct_max(stem_a_pw2), lo=8.0, hi=512.0),
        "dw3": _scale_from_max(pct_max(stem_a_dw3), lo=8.0, hi=512.0),
        "pw3": _scale_from_max(pct_max(stem_a_pw3), lo=8.0, hi=512.0),
    }
    model.stem_w_scales = {
        "stem": _per_channel_scale_conv(model.stem.stem.weight).cpu(),
        "dw": _per_channel_scale_conv(model.stem.dw.weight).cpu(),
        "pw": _per_channel_scale_conv(model.stem.pw.weight).cpu(),
        "dw2": _per_channel_scale_conv(model.stem.dw2.weight).cpu(),
        "pw2": _per_channel_scale_conv(model.stem.pw2.weight).cpu(),
        "dw3": _per_channel_scale_conv(model.stem.dw3.weight).cpu(),
        "pw3": _per_channel_scale_conv(model.stem.pw3.weight).cpu(),
    }
    print(f"calibrated stem acts: {model.stem_act_scales}")
    print(
        "calibrated stem w (per-ch mean): "
        + " ".join(f"{k}={float(v.mean()):.1f}" for k, v in model.stem_w_scales.items())
    )

    se_a = _scale_from_max(pct_max(embed_act), lo=8.0, hi=512.0)
    se_w = _scale_from_max(
        pct_max([model.w_pe.detach().abs().reshape(-1)]), lo=8.0, hi=512.0
    )
    model.scale_embed = vit.QuantScales(act=se_a, w=se_w)
    print(f"calibrated embed(legacy): act={se_a:.2f} w={se_w:.2f}")

    new_blocks: list[vit.QuantScales] = []
    for li, blk in enumerate(model.blocks):
        a = _scale_from_max(pct_max(block_act[li]), lo=8.0, hi=512.0)
        blk.sw_wq.copy_(_per_channel_scale_linear(blk.wq))
        blk.sw_wk.copy_(_per_channel_scale_linear(blk.wk))
        blk.sw_wv.copy_(_per_channel_scale_linear(blk.wv))
        blk.sw_wo.copy_(_per_channel_scale_linear(blk.wo))
        blk.sw_w1.copy_(_per_channel_scale_linear(blk.w1))
        blk.sw_w2.copy_(_per_channel_scale_linear(blk.w2))
        w_mean = float(
            torch.cat(
                [
                    blk.sw_wq.reshape(-1),
                    blk.sw_wk.reshape(-1),
                    blk.sw_wv.reshape(-1),
                    blk.sw_wo.reshape(-1),
                    blk.sw_w1.reshape(-1),
                    blk.sw_w2.reshape(-1),
                ]
            )
            .mean()
            .item()
        )
        p = _scale_from_max(pct_max(block_attn[li]), lo=16.0, hi=512.0)
        new_blocks.append(vit.QuantScales(act=a, w=w_mean, p=p))
        print(
            f"calibrated block{li}: act={a:.2f} w_mean={w_mean:.2f} p={p:.2f} "
            f"(per-channel weights)"
        )
    model.scale_blocks = new_blocks

    sc_a = _scale_from_max(pct_max(cls_act), lo=8.0, hi=512.0)
    model.sw_cls.copy_(_per_channel_scale_linear(model.w_cls))
    sc_w = float(model.sw_cls.mean().item())
    model.scale_cls = vit.QuantScales(act=sc_a, w=sc_w)
    print(f"calibrated cls: act={sc_a:.2f} w_mean={sc_w:.2f} (per-channel)")


def _quant_weight(w: torch.Tensor, scale_w: float | torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(scale_w, torch.Tensor):
        s = scale_w.detach().cpu().numpy()
    else:
        s = scale_w
    return nt.quant_weight_to_i8(w.detach().cpu().numpy(), scale=s)


def _stem_w_scales_np(model: TinyViT) -> dict[str, np.ndarray]:
    return {k: np.asarray(v.detach().cpu().numpy(), dtype=np.float64) for k, v in model.stem_w_scales.items()}


def model_to_weights(model: TinyViT) -> vit.VitMnistWeights:
    """Snapshot float params → int8/Q12 VitMnistWeights (board-ref layout)."""
    blocks = []
    for blk, sc in zip(model.blocks, model.scale_blocks):
        blocks.append(
            nt.TinyBlockWeights(
                wq=_quant_weight(blk.wq, blk.sw_wq),
                wk=_quant_weight(blk.wk, blk.sw_wk),
                wv=_quant_weight(blk.wv, blk.sw_wv),
                wo=_quant_weight(blk.wo, blk.sw_wo),
                w1=_quant_weight(blk.w1, blk.sw_w1),
                w2=_quant_weight(blk.w2, blk.sw_w2),
                gamma1=nt.to_q12(blk.gamma1.detach().cpu().numpy()),
                gamma2=nt.to_q12(blk.gamma2.detach().cpu().numpy()),
                sw_wq=blk.sw_wq.detach().cpu().numpy().astype(np.float64),
                sw_wk=blk.sw_wk.detach().cpu().numpy().astype(np.float64),
                sw_wv=blk.sw_wv.detach().cpu().numpy().astype(np.float64),
                sw_wo=blk.sw_wo.detach().cpu().numpy().astype(np.float64),
                sw_w1=blk.sw_w1.detach().cpu().numpy().astype(np.float64),
                sw_w2=blk.sw_w2.detach().cpu().numpy().astype(np.float64),
            )
        )
    stem = stemmod.StemInt8.from_module(
        model.stem,
        act_scales=model.stem_act_scales,
        w_scales=_stem_w_scales_np(model),
    )
    sw_cls = model.sw_cls.detach().cpu().numpy().astype(np.float64)
    scale_cls = vit.QuantScales(act=model.scale_cls.act, w=sw_cls, p=model.scale_cls.p)
    return vit.VitMnistWeights(
        w_pe=_quant_weight(model.w_pe, model.scale_embed.w),
        pos=nt.to_q12(model.pos.detach().cpu().numpy()),
        blocks=tuple(blocks),
        w_cls=_quant_weight(model.w_cls, model.sw_cls),
        scale_embed=model.scale_embed,
        scale_blocks=tuple(model.scale_blocks),
        scale_cls=scale_cls,
        stem=stem,
    )


@torch.no_grad()
def deploy_numpy_logits(model: TinyViT, imgs28: torch.Tensor) -> torch.Tensor:
    """Board-ref numpy forward for a batch → float logits [B,10]."""
    w = model_to_weights(model)
    imgs = imgs28.detach().cpu().numpy().astype(np.float64)
    outs = []
    for i in range(imgs.shape[0]):
        logits_q12, _ = vit.vit_forward(
            imgs[i], w, use_hw=False, verbose=False, use_tflite_stem=False
        )
        outs.append(nt.from_q12(logits_q12))
    arr = np.stack(outs, axis=0).astype(np.float32)
    return torch.from_numpy(arr).to(device=imgs28.device, dtype=imgs28.dtype)


def load_weights_into_model(model: TinyViT, path: Path, device: torch.device) -> None:
    """Load exported int8/Q12 npz back into float TinyViT (for continued deploy-FT)."""
    w = vit.VitMnistWeights.load(path)
    if len(w.blocks) != len(model.blocks):
        raise ValueError(
            f"{path} has {len(w.blocks)} layers; model has {len(model.blocks)}"
        )
    def _deq(wi8: np.ndarray, sw) -> torch.Tensor:
        s = np.asarray(sw, dtype=np.float32)
        if s.ndim == 0 or s.size == 1:
            return torch.from_numpy(wi8.astype(np.float32) / float(s.reshape(-1)[0]))
        return torch.from_numpy(wi8.astype(np.float32) / s.reshape(1, -1))

    if w.stem is not None:
        s = w.stem

        def _deq_conv(wi8: np.ndarray, sw: np.ndarray) -> torch.Tensor:
            ss = np.asarray(sw, dtype=np.float32).reshape(-1, *([1] * (wi8.ndim - 1)))
            return torch.from_numpy(wi8.astype(np.float32) / ss)

        model.stem.stem.weight.data.copy_(_deq_conv(s.w_stem, s.sw_stem))
        model.stem.stem.bias.data.copy_(torch.from_numpy(s.b_stem))
        model.stem.dw.weight.data.copy_(_deq_conv(s.w_dw, s.sw_dw))
        model.stem.dw.bias.data.copy_(torch.from_numpy(s.b_dw))
        model.stem.pw.weight.data.copy_(_deq_conv(s.w_pw, s.sw_pw))
        model.stem.pw.bias.data.copy_(torch.from_numpy(s.b_pw))
        model.stem.dw2.weight.data.copy_(_deq_conv(s.w_dw2, s.sw_dw2))
        model.stem.dw2.bias.data.copy_(torch.from_numpy(s.b_dw2))
        model.stem.pw2.weight.data.copy_(_deq_conv(s.w_pw2, s.sw_pw2))
        model.stem.pw2.bias.data.copy_(torch.from_numpy(s.b_pw2))
        model.stem.dw3.weight.data.copy_(_deq_conv(s.w_dw3, s.sw_dw3))
        model.stem.dw3.bias.data.copy_(torch.from_numpy(s.b_dw3))
        model.stem.pw3.weight.data.copy_(_deq_conv(s.w_pw3, s.sw_pw3))
        model.stem.pw3.bias.data.copy_(torch.from_numpy(s.b_pw3))
        model.stem_act_scales = {
            "in": s.sa_in,
            "stem": s.sa_stem,
            "dw": s.sa_dw,
            "pw": s.sa_pw,
            "dw2": s.sa_dw2,
            "pw2": s.sa_pw2,
            "dw3": s.sa_dw3,
            "pw3": s.sa_pw3,
        }
        model.stem_w_scales = {
            "stem": torch.from_numpy(np.asarray(s.sw_stem, dtype=np.float32)),
            "dw": torch.from_numpy(np.asarray(s.sw_dw, dtype=np.float32)),
            "pw": torch.from_numpy(np.asarray(s.sw_pw, dtype=np.float32)),
            "dw2": torch.from_numpy(np.asarray(s.sw_dw2, dtype=np.float32)),
            "pw2": torch.from_numpy(np.asarray(s.sw_pw2, dtype=np.float32)),
            "dw3": torch.from_numpy(np.asarray(s.sw_dw3, dtype=np.float32)),
            "pw3": torch.from_numpy(np.asarray(s.sw_pw3, dtype=np.float32)),
        }
    model.w_pe.data.copy_(
        torch.from_numpy((w.w_pe.astype(np.float32) / float(np.asarray(w.scale_embed.w).reshape(-1)[0])))
    )
    model.pos.data.copy_(
        torch.from_numpy(nt.from_q12(w.pos).astype(np.float32))
    )
    sw_cls = np.asarray(w.scale_cls.w, dtype=np.float32).reshape(-1)
    if sw_cls.size == 1:
        sw_cls = np.full(vit.N_CLASS, float(sw_cls[0]), dtype=np.float32)
    model.w_cls.data.copy_(
        torch.from_numpy(w.w_cls.astype(np.float32) / sw_cls.reshape(1, -1))
    )
    model.sw_cls.copy_(torch.from_numpy(sw_cls.astype(np.float32)))
    model.scale_embed = w.scale_embed
    model.scale_cls = w.scale_cls
    model.scale_blocks = list(w.scale_blocks)
    for blk, bw, sc in zip(model.blocks, w.blocks, w.scale_blocks):
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            wi8 = getattr(bw, name)
            sw = getattr(bw, f"sw_{name}")
            if sw is None:
                sw = np.asarray(sc.w, dtype=np.float64)
            getattr(blk, name).data.copy_(_deq(wi8, sw))
            getattr(blk, f"sw_{name}").copy_(
                torch.from_numpy(np.asarray(sw, dtype=np.float32).reshape(-1))
            )
        blk.gamma1.data.copy_(
            torch.from_numpy(nt.from_q12(bw.gamma1).astype(np.float32))
        )
        blk.gamma2.data.copy_(
            torch.from_numpy(nt.from_q12(bw.gamma2).astype(np.float32))
        )
    model.to(device)

    def _wmean(s) -> float:
        return float(np.asarray(s.w).reshape(-1).mean())

    print(
        f"loaded {path} → float TinyViT+DS-stem  layers={len(model.blocks)}  "
        + " ".join(
            f"L{i}={s.act:.1f}/{_wmean(s):.1f}/{s.p:.1f}"
            for i, s in enumerate(model.scale_blocks)
        )
        + f"  cls={model.scale_cls.act:.1f}/{float(model.sw_cls.mean()):.1f}"
    )


def export_weights(model: TinyViT) -> None:
    sw_cls = model.sw_cls.detach().cpu().numpy().astype(np.float64)
    payload: dict[str, np.ndarray] = {
        "w_pe": _quant_weight(model.w_pe, model.scale_embed.w),
        "pos": nt.to_q12(model.pos.detach().cpu().numpy()),
        "w_cls": _quant_weight(model.w_cls, model.sw_cls),
        "meta_t": np.array([vit.VIT_T]),
        "meta_d": np.array([vit.VIT_D]),
        "meta_mlp": np.array([vit.VIT_MLP]),
        "meta_layers": np.array([len(model.blocks)]),
        "meta_stem": np.array([1]),
        "meta_glue_mode": np.array(["float"]),
        # legacy aliases = block0 (older loaders)
        "scale_act": np.array([model.scale_blocks[0].act], dtype=np.float64),
        "scale_w": np.array([model.scale_blocks[0].w], dtype=np.float64),
        "scale_p": np.array([model.scale_blocks[0].p], dtype=np.float64),
        "scale_embed_act": np.array([model.scale_embed.act], dtype=np.float64),
        "scale_embed_w": np.array([float(np.asarray(model.scale_embed.w).reshape(-1)[0])], dtype=np.float64),
        "scale_cls_act": np.array([model.scale_cls.act], dtype=np.float64),
        "scale_cls_w": np.array([float(sw_cls.mean())], dtype=np.float64),
        "scale_cls_w_ch": sw_cls,
    }
    stem = stemmod.StemInt8.from_module(
        model.stem,
        act_scales=model.stem_act_scales,
        w_scales=_stem_w_scales_np(model),
    )
    payload.update(stem.to_dict())
    for i, (blk, sc) in enumerate(zip(model.blocks, model.scale_blocks)):
        payload[f"wq{i}"] = _quant_weight(blk.wq, blk.sw_wq)
        payload[f"wk{i}"] = _quant_weight(blk.wk, blk.sw_wk)
        payload[f"wv{i}"] = _quant_weight(blk.wv, blk.sw_wv)
        payload[f"wo{i}"] = _quant_weight(blk.wo, blk.sw_wo)
        payload[f"w1{i}"] = _quant_weight(blk.w1, blk.sw_w1)
        payload[f"w2{i}"] = _quant_weight(blk.w2, blk.sw_w2)
        payload[f"gamma1{i}"] = nt.to_q12(blk.gamma1.detach().cpu().numpy())
        payload[f"gamma2{i}"] = nt.to_q12(blk.gamma2.detach().cpu().numpy())
        payload[f"scale_block{i}_act"] = np.array([sc.act], dtype=np.float64)
        payload[f"scale_block{i}_w"] = np.array([sc.w], dtype=np.float64)
        payload[f"scale_block{i}_p"] = np.array([sc.p], dtype=np.float64)
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            payload[f"scale_block{i}_{name}_w"] = (
                getattr(blk, f"sw_{name}").detach().cpu().numpy().astype(np.float64)
            )
    np.savez_compressed(WEIGHTS_PATH, **payload)
    print(
        f"wrote {WEIGHTS_PATH}  stem=1 layers={len(model.blocks)} mlp={vit.VIT_MLP}  "
        + " ".join(
            f"L{i}={s.act:.1f}/{float(np.asarray(s.w).mean()):.1f}/{s.p:.1f}"
            for i, s in enumerate(model.scale_blocks)
        )
        + f"  cls={model.scale_cls.act:.1f}/{float(sw_cls.mean()):.1f}"
    )
    # TFLite stem for A9 XNNPACK
    try:
        from export_tflite_cnn import export_stem_tflite

        export_stem_tflite(model)
    except Exception as exc:
        print(f"warn: stem TFLite export skipped ({exc})")


def save_sample(x_te: np.ndarray, y_te: np.ndarray, n: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_te), size=min(n, len(y_te)), replace=False)
    np.savez_compressed(SAMPLE_PATH, images=x_te[idx], labels=y_te[idx])
    print(f"wrote {SAMPLE_PATH} n={len(idx)}")


@torch.no_grad()
def accuracy(
    model: TinyViT,
    loader: DataLoader,
    device: torch.device,
    *,
    qat: bool,
    deploy_faithful: bool = False,
) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, qat=qat, deploy_faithful=deploy_faithful).argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def quant_numpy_accuracy(
    x_te: np.ndarray,
    y_te: np.ndarray,
    n_eval: int,
    *,
    log_every: int = 1000,
) -> float:
    """Board-ref path accuracy on first n_eval test images (n_eval<=0 → all)."""
    w = vit.VitMnistWeights.load(WEIGHTS_PATH)
    n_eval = len(y_te) if n_eval <= 0 else min(n_eval, len(y_te))
    correct = 0
    for i in range(n_eval):
        _, dump = vit.vit_forward(
            x_te[i], w, use_hw=False, verbose=False, use_tflite_stem=False
        )
        correct += int(dump["pred"][0] == y_te[i])
        if log_every and (i + 1) % log_every == 0:
            print(
                f"  numpy eval {i + 1}/{n_eval}  "
                f"running_acc={100.0 * correct / (i + 1):.2f}%"
            )
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
    n_layers = int(args.layers)
    vit.N_LAYERS = n_layers  # keep module contract in sync for prints/meta
    model = TinyViT(n_layers=n_layers).to(device)
    if args.init_weights:
        load_weights_into_model(model, Path(args.init_weights), device)

    def run_epochs(
        tag: str,
        n_epochs: int,
        *,
        qat: bool,
        lr: float,
        deploy_faithful: bool = False,
        eval_deploy: bool = False,
        label_smoothing: float = 0.0,
        recal_every: int = 0,
    ) -> None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=args.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            n = 0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if args.augment:
                    xb = augment_batch(
                        xb,
                        max_shift=args.aug_shift,
                        noise_std=args.aug_noise,
                        erase_prob=args.aug_erase_prob,
                        erase_size=args.aug_erase_size,
                    )
                opt.zero_grad(set_to_none=True)
                logits = model(xb, qat=qat, deploy_faithful=deploy_faithful)
                loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                total_loss += float(loss.item()) * int(yb.numel())
                n += int(yb.numel())
            sched.step()
            if qat and recal_every > 0 and epoch % recal_every == 0 and epoch < n_epochs:
                print(f"{tag} epoch {epoch:02d}: re-calibrate scales")
                calibrate_scales(model, cal_loader, device, batches=args.cal_batches)
            te_acc = accuracy(
                model,
                te_loader,
                device,
                qat=qat,
                deploy_faithful=eval_deploy,
            )
            print(
                f"{tag} epoch {epoch:02d}  loss={total_loss / max(n, 1):.4f}  "
                f"test_acc={te_acc * 100:.2f}%  lr={sched.get_last_lr()[0]:.2e}  "
                f"scales L0={model.scale_blocks[0].act:.1f}/"
                f"{model.scale_blocks[0].w:.1f}/{model.scale_blocks[0].p:.1f}"
            )

    n_stem = sum(p.numel() for p in model.stem.parameters())
    n_body = sum(p.numel() for p in model.parameters()) - n_stem
    print(
        f"train tiny-ViT+DS-stem T={vit.VIT_T} D={vit.VIT_D} mlp={vit.VIT_MLP} "
        f"L={n_layers} glue={vit.DEFAULT_GLUE_MODE} "
        f"stem_params={n_stem} body_params={n_body} total={n_stem + n_body} "
        f"float={args.epochs} qat={args.qat_epochs} deploy_ft={args.deploy_epochs} "
        f"init={args.init_weights or '-'} "
        f"train_n={len(y_tr)} aug={args.augment}/shift{args.aug_shift}/"
        f"noise{args.aug_noise}/erase{args.aug_erase_prob} device={device}"
    )
    if args.epochs > 0:
        run_epochs(
            "float",
            args.epochs,
            qat=False,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
        )
    if args.epochs > 0 or not args.init_weights:
        calibrate_scales(model, cal_loader, device, batches=args.cal_batches)
    if args.qat_epochs > 0:
        run_epochs(
            "qat",
            args.qat_epochs,
            qat=True,
            lr=args.lr * args.qat_lr_mult,
            label_smoothing=args.label_smoothing,
            recal_every=args.qat_recal_every,
        )
    # Snapshot after QAT (stem now fake-int8 in proxy — deploy should already be close).
    export_weights(model)
    probe_n = min(2000, len(y_te)) if args.eval_n <= 0 else min(args.eval_n, len(y_te))
    print(f"post-QAT numpy deploy probe on {probe_n} images...")
    qacc_probe = quant_numpy_accuracy(x_te, y_te, probe_n)
    qat_proxy = accuracy(model, te_loader, device, qat=True)
    print(
        f"post-QAT proxy={100 * qat_proxy:.2f}%  numpy_deploy={100 * qacc_probe:.2f}%  "
        f"gap={100 * (qat_proxy - qacc_probe):+.2f} pp"
    )

    do_deploy = args.deploy_epochs > 0
    if do_deploy and qacc_probe >= args.deploy_skip_threshold:
        print(
            f"skipping deploy-FT (numpy_deploy {100 * qacc_probe:.2f}% "
            f">= threshold {100 * args.deploy_skip_threshold:.2f}%)"
        )
        do_deploy = False
    if do_deploy:
        if args.recalibrate_after_qat:
            # Recalibrating after stem-QAT retunes the int8 grid under trained
            # weights and usually widens proxy↔numpy gap — keep off unless asked.
            print("re-calibrate scales (pre-deploy) — may widen stem quant gap")
            calibrate_scales(model, cal_loader, device, batches=args.cal_batches)
        # Fine-tune so CE sees board-ref numpy logits (STE grads via proxy QAT).
        if args.qat_batch > 0 and args.qat_batch != args.batch:
            tr_loader = DataLoader(tr_ds, batch_size=args.qat_batch, shuffle=True)
        run_epochs(
            "deploy",
            args.deploy_epochs,
            qat=True,
            lr=args.lr * args.qat_lr_mult * args.deploy_lr_mult,
            deploy_faithful=True,
            eval_deploy=False,
            label_smoothing=0.0,
        )
        export_weights(model)
    n_eval = len(y_te) if args.eval_n <= 0 else min(args.eval_n, len(y_te))
    print(f"numpy quantized eval on {n_eval} test images (board-ref path)...")
    qacc = quant_numpy_accuracy(x_te, y_te, n_eval if args.eval_n > 0 else 0)
    facc = accuracy(model, te_loader, device, qat=False)
    qat_acc = accuracy(model, te_loader, device, qat=True)
    print(f"float test accuracy (full):     {100 * facc:.2f}%")
    print(f"QAT-mode test accuracy (full):  {100 * qat_acc:.2f}%")
    print(f"numpy quantized ref on {n_eval}: {100 * qacc:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train MNIST tiny-ViT for NpuKit (QAT)")
    p.add_argument(
        "--layers",
        type=int,
        default=4,
        help="transformer blocks (2–4); stem is separate on CPU/XNNPACK",
    )
    p.add_argument("--epochs", type=int, default=45, help="float warm-up epochs (0=skip)")
    p.add_argument("--qat-epochs", type=int, default=20, help="0=skip")
    p.add_argument("--qat-lr-mult", type=float, default=0.25)
    p.add_argument("--cal-batches", type=int, default=30)
    p.add_argument(
        "--init-weights",
        default="",
        help="load vit_mnist_weights.npz and continue (dequant to float)",
    )
    p.add_argument(
        "--eval-n",
        type=int,
        default=0,
        help="numpy eval images; 0 = full test set (10000)",
    )
    p.add_argument(
        "--qat-batch",
        type=int,
        default=32,
        help="batch size during deploy-faithful fine-tune (0 = use --batch)",
    )
    p.add_argument(
        "--deploy-epochs",
        type=int,
        default=12,
        help="extra epochs with board-ref numpy logits STE",
    )
    p.add_argument(
        "--deploy-skip-threshold",
        type=float,
        default=0.96,
        help="skip deploy-FT when post-QAT numpy deploy acc >= this",
    )
    p.add_argument(
        "--deploy-lr-mult",
        type=float,
        default=0.4,
        help="LR multiplier on top of qat-lr during deploy fine-tune",
    )
    p.add_argument(
        "--recalibrate-after-qat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="re-run per-stage scale calib before deploy-FT (usually hurts stem-QAT)",
    )
    p.add_argument(
        "--qat-recal-every",
        type=int,
        default=0,
        help="re-calibrate scales every N QAT epochs (0 = off)",
    )
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="CE label smoothing for float/QAT (deploy uses 0)",
    )
    p.add_argument("--subset", type=int, default=0, help="0 = full train set (60k)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug-shift", type=int, default=2, help="max ±px translate")
    p.add_argument("--aug-noise", type=float, default=0.05, help="gaussian noise std")
    p.add_argument("--aug-erase-prob", type=float, default=0.15)
    p.add_argument("--aug-erase-size", type=int, default=4)
    p.add_argument("--sample-n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())

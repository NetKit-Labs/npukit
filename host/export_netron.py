#!/usr/bin/env python3
"""Export Netron-viewable graphs for the two edge peers.

Opens in https://netron.app (drag-and-drop) or the Netron desktop app.

Writes:
  host/netron/dscnn_mnist_peer.onnx     — MCU-class DS-CNN benchmark (CPU-only)
  host/netron/dscnn_mnist_peer.tflite   — copy of TFLite peer (also Netron-ok)
  host/netron/npukit_vit_system.onnx    — full NpuKit path: CPU DS-stem + FPGA
                                         transformer body + CPU head

Node name prefixes encode the deploy split:
  CPU/...   host / TFLite+XNNPACK (A9): DS-stem, pos, Softmax/RMSNorm/GELU, head
  FPGA/...  NpuKit int8 GEMM (time-multiplexed on PL); HW glue optional

Usage:
  python3 host/export_netron.py
  python3 host/export_netron.py --layers 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import dscnn_mnist as dscnn
import npukit_vit_mnist as vit
import train_vit_mnist as tv
import vit_ds_stem as stemmod

HOST = Path(__file__).resolve().parent
OUT = HOST / "netron"


class _AttnBlock(nn.Module):
    """Explicit Linear-based block so Netron shows matmuls clearly."""

    def __init__(self, d: int, mlp_h: int | None = None) -> None:
        super().__init__()
        mh = int(mlp_h if mlp_h is not None else d)
        self.rmsnorm1_gamma = nn.Parameter(torch.ones(d))
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rmsnorm2_gamma = nn.Parameter(torch.ones(d))
        self.w1 = nn.Linear(d, mh, bias=False)
        self.w2 = nn.Linear(mh, d, bias=False)

    def _rms(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
        return (x / rms) * g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self._rms(x, self.rmsnorm1_gamma)
        q, k, v = self.wq(xn), self.wk(xn), self.wv(xn)
        scale = 1.0 / (x.shape[-1] ** 0.5)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        x = x + self.wo(attn @ v)
        xn = self._rms(x, self.rmsnorm2_gamma)
        x = x + self.w2(F.gelu(self.w1(xn)))
        return x


class NpuKitViTSystem(nn.Module):
    """Full deploy graph for Netron: CPU stem → FPGA blocks → CPU head."""

    def __init__(self, n_layers: int | None = None) -> None:
        super().__init__()
        d = vit.VIT_D
        n_layers = int(n_layers if n_layers is not None else vit.N_LAYERS)
        # --- CPU / A9 (TFLite+XNNPACK intended) ---
        self.CPU_ds_stem = stemmod.TinyDSStem(d)
        self.CPU_pos = nn.Parameter(torch.zeros(1, vit.VIT_T, d))
        # --- FPGA (int8 GEMM; Softmax/RMSNorm/GELU on A9 float in deploy) ---
        self.FPGA_transformer = nn.ModuleList(
            [_AttnBlock(d, mlp_h=vit.VIT_MLP) for _ in range(n_layers)]
        )
        # --- CPU head ---
        self.CPU_classifier = nn.Linear(d, vit.N_CLASS, bias=False)

    def forward(self, image_nchw: torch.Tensor) -> torch.Tensor:
        # image: [B,1,28,28]
        tokens = self.CPU_ds_stem(
            image_nchw.squeeze(1) if image_nchw.ndim == 4 else image_nchw
        )
        x = tokens + self.CPU_pos
        for blk in self.FPGA_transformer:
            x = blk(x)
        pooled = x.mean(dim=1)  # CPU mean-pool
        return self.CPU_classifier(pooled)


def _load_vit_into_netron(model: NpuKitViTSystem, src: tv.TinyViT) -> None:
    model.CPU_ds_stem.load_state_dict(src.stem.state_dict())
    model.CPU_pos.data.copy_(src.pos.detach().unsqueeze(0))
    for dst, blk in zip(model.FPGA_transformer, src.blocks):
        dst.rmsnorm1_gamma.data.copy_(blk.gamma1.detach())
        dst.rmsnorm2_gamma.data.copy_(blk.gamma2.detach())
        dst.wq.weight.data.copy_(blk.wq.detach().T)
        dst.wk.weight.data.copy_(blk.wk.detach().T)
        dst.wv.weight.data.copy_(blk.wv.detach().T)
        dst.wo.weight.data.copy_(blk.wo.detach().T)
        dst.w1.weight.data.copy_(blk.w1.detach().T)
        dst.w2.weight.data.copy_(blk.w2.detach().T)
    model.CPU_classifier.weight.data.copy_(src.w_cls.detach().T)


def export_dscnn_onnx(path: Path) -> None:
    device = torch.device("cpu")
    if dscnn.WEIGHTS_PATH.exists():
        m = dscnn.load_model(dscnn.WEIGHTS_PATH, device)
    else:
        m = dscnn.DSCNN().to(device)
    m.eval()
    # Wrap so top-level name is readable in Netron
    class MCU_DS_CNN_Peer(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.CPU_dscnn = inner

        def forward(self, image_nchw: torch.Tensor) -> torch.Tensor:
            return self.CPU_dscnn(image_nchw)

    wrap = MCU_DS_CNN_Peer(m).eval()
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        wrap,
        dummy,
        path,
        input_names=["image_28x28"],
        output_names=["logits_10"],
        opset_version=17,
        dynamo=False,
    )
    _annotate_onnx(
        path,
        doc=(
            "NpuKit edge peer: MCU-class DS-CNN on MNIST (CPU / TFLite+XNNPACK). "
            "Not FPGA-mapped. Compare vs npukit_vit_system.onnx."
        ),
    )
    print(f"wrote {path}")


def export_vit_system_onnx(path: Path, *, n_layers: int) -> None:
    net = NpuKitViTSystem(n_layers=n_layers).eval()
    src = tv.TinyViT(n_layers=n_layers).eval()
    # Prefer in-memory trained weights if npz has matching stem+layers
    if tv.WEIGHTS_PATH.exists():
        try:
            w = vit.VitMnistWeights.load(tv.WEIGHTS_PATH)
            if w.stem is not None and len(w.blocks) == n_layers:
                tv.load_weights_into_model(src, tv.WEIGHTS_PATH, torch.device("cpu"))
                print(f"loaded weights from {tv.WEIGHTS_PATH}")
            else:
                print(
                    f"note: {tv.WEIGHTS_PATH} stem={w.stem is not None} "
                    f"layers={len(w.blocks)} (want {n_layers}); exporting arch structure"
                )
        except Exception as exc:
            print(f"note: could not load vit weights ({exc}); exporting arch structure")
    _load_vit_into_netron(net, src)
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        net,
        dummy,
        path,
        input_names=["image_28x28"],
        output_names=["logits_10"],
        opset_version=17,
        dynamo=False,
    )
    _annotate_onnx(
        path,
        doc=(
            "NpuKit full MNIST tiny-ViT deploy graph. "
            "CPU_ds_stem + CPU_pos + CPU_classifier run on the A9 "
            "(TFLite/XNNPACK intended for the stem). "
            "FPGA_transformer_* blocks are host-scheduled on the "
            "8×8 int8 systolic GEMM + transformer glue (MAX_LEN=16)."
        ),
    )
    print(f"wrote {path}  (L={n_layers})")


def _annotate_onnx(path: Path, *, doc: str) -> None:
    import onnx

    m = onnx.load(str(path))
    m.doc_string = doc
    m.graph.name = path.stem
    # Helpful graph doc for Netron details pane
    m.graph.doc_string = doc
    onnx.save(m, str(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export Netron ONNX graphs")
    p.add_argument("--layers", type=int, default=int(vit.N_LAYERS))
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    export_dscnn_onnx(OUT / "dscnn_mnist_peer.onnx")
    tflite_src = HOST / "dscnn_mnist.tflite"
    if tflite_src.exists():
        shutil.copy2(tflite_src, OUT / "dscnn_mnist_peer.tflite")
        print(f"wrote {OUT / 'dscnn_mnist_peer.tflite'}")
    export_vit_system_onnx(OUT / "npukit_vit_system.onnx", n_layers=args.layers)

    (OUT / "README.txt").write_text(
        "Open these in https://netron.app (Open Model… / drag-and-drop).\n\n"
        "dscnn_mnist_peer.onnx|.tflite  — MCU DS-CNN benchmark (CPU-only)\n"
        "npukit_vit_system.onnx         — CPU DS-stem + FPGA transformer + CPU head\n\n"
        "In the ViT graph, node prefixes CPU_* vs FPGA_* show the deploy split.\n"
    )
    print(f"\nOpen in Netron: https://netron.app")
    print(f"  {OUT / 'dscnn_mnist_peer.onnx'}")
    print(f"  {OUT / 'npukit_vit_system.onnx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

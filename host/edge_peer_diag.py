#!/usr/bin/env python3
"""Edge-peer diagnostics: float vs deploy ViT, param counts, A9 latency.

Item-1 check (is the gap a deploy bug?):
  - float TinyViT (dequantized from vit_mnist_weights.npz)
  - proxy-QAT TinyViT
  - board-ref numpy deploy-quant path

A9 bench (no torch required on board):
  - DS-CNN int8 on A9 CPU (numpy)
  - tiny-ViT on FPGA (host schedule + PL)

Usage (host / Docker — needs torch):
  python3 host/edge_peer_diag.py
  python3 host/edge_peer_diag.py --eval-n 2000

Usage (on PYNQ, sudo + XRT/venv — numpy only):
  python3 edge_peer_diag.py --skip-acc --bench \\
      --bit /home/xilinx/jupyter_notebooks/npukit.bit --bench-n 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST))

import npukit_vit_mnist as vit  # noqa: E402

DSCNN_INT8 = HOST / "dscnn_mnist_int8.npz"
DSCNN_METRICS = HOST / "dscnn_mnist_metrics.json"
DSCNN_WEIGHTS = HOST / "dscnn_mnist_weights.pt"


def _count_vit_deploy(w: vit.VitMnistWeights) -> dict:
    i8 = [w.w_pe, w.w_cls]
    q12 = [w.pos]
    for b in w.blocks:
        i8 += [b.wq, b.wk, b.wv, b.wo, b.w1, b.w2]
        q12 += [b.gamma1, b.gamma2]
    n_i8 = int(sum(a.size for a in i8))
    n_q12 = int(sum(a.size for a in q12))
    bytes_i8 = int(sum(a.nbytes for a in i8))
    bytes_q12 = int(sum(a.nbytes for a in q12))
    return {
        "n_params_i8": n_i8,
        "n_params_q12": n_q12,
        "n_params_total": n_i8 + n_q12,
        "bytes_i8": bytes_i8,
        "bytes_q12": bytes_q12,
        "kib_deploy": (bytes_i8 + bytes_q12) / 1024.0,
        "kib_i8_only": bytes_i8 / 1024.0,
    }


def _count_dscnn_int8(path: Path = DSCNN_INT8) -> dict:
    z = np.load(path)
    n_i8 = bytes_i8 = 0
    n_bias = bytes_bias = 0
    for k in z.files:
        a = z[k]
        if a.dtype == np.int8:
            n_i8 += int(a.size)
            bytes_i8 += int(a.nbytes)
        elif k.startswith("b_"):
            n_bias += int(a.size)
            bytes_bias += int(a.nbytes)
    return {
        "n_params_i8": n_i8,
        "n_params_bias": n_bias,
        "n_params_total": n_i8 + n_bias,
        "bytes_i8": bytes_i8,
        "bytes_bias": bytes_bias,
        "kib_deploy": (bytes_i8 + bytes_bias) / 1024.0,
        "kib_i8_only": bytes_i8 / 1024.0,
    }


def _fq(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.round(x * scale), -128, 127).astype(np.float32) / scale


def _conv2d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> np.ndarray:
    """Vectorized NCHW conv for DS-CNN bench (float32)."""
    n, cin, _, _ = x.shape
    cout, cg, kh, kw = w.shape
    assert cin == cg * groups
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h_out = (x.shape[2] - kh) // stride + 1
    w_out = (x.shape[3] - kw) // stride + 1
    # as_strided windows: [N, Cin, Hout, Wout, Kh, Kw]
    s0, s1, s2, s3 = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n, cin, h_out, w_out, kh, kw),
        strides=(s0, s1, s2 * stride, s3 * stride, s2, s3),
        writeable=False,
    )
    y = np.empty((n, cout, h_out, w_out), dtype=np.float32)
    cout_g = cout // groups
    for g in range(groups):
        # [N, Cg, Hout, Wout, Kh, Kw]
        xg = windows[:, g * cg : (g + 1) * cg]
        wg = w[g * cout_g : (g + 1) * cout_g]  # [Cout_g, Cg, Kh, Kw]
        # tensordot over Cin/Cg, Kh, Kw → [N, Hout, Wout, Cout_g]
        yg = np.tensordot(xg, wg, axes=([1, 4, 5], [1, 2, 3]))
        y[:, g * cout_g : (g + 1) * cout_g] = np.moveaxis(yg, -1, 1) + b[
            g * cout_g : (g + 1) * cout_g
        ].reshape(1, -1, 1, 1)
    return y


def dscnn_int8_numpy_forward(img28: np.ndarray, z) -> np.ndarray:
    """One MNIST image → logits using exported int8 npz (fake-int8 style)."""
    x = np.asarray(img28, dtype=np.float32).reshape(1, 1, 28, 28)

    def layer(x_in, name, act_key, *, stride=1, padding=0, groups=1):
        sa = float(z[f"sa_{act_key}"][0])
        sw = float(z[f"sw_{name}"][0])
        w_q = z[f"w_{name}"].astype(np.float32)
        b = z[f"b_{name}"].astype(np.float32)
        w = w_q / sw
        xq = _fq(x_in, sa)
        y = _conv2d(xq, w, b, stride=stride, padding=padding, groups=groups)
        return np.maximum(y, 0.0)

    x = layer(x, "stem", "in", stride=1, padding=1, groups=1)
    x = layer(x, "b1_dw", "stem", stride=2, padding=1, groups=16)
    x = layer(x, "b1_pw", "b1_dw", stride=1, padding=0, groups=1)
    x = layer(x, "b2_dw", "b1_pw", stride=2, padding=1, groups=32)
    x = layer(x, "b2_pw", "b2_dw", stride=1, padding=0, groups=1)
    x = layer(x, "b3_dw", "b2_pw", stride=1, padding=1, groups=64)
    x = layer(x, "b3_pw", "b3_dw", stride=1, padding=0, groups=1)
    pooled = x.mean(axis=(2, 3))  # adaptive avg pool 1x1
    sa = float(z["sa_b3_pw"][0])
    sw = float(z["sw_fc"][0])
    w = z["w_fc"].astype(np.float32) / sw
    b = z["b_fc"].astype(np.float32)
    xq = _fq(pooled, sa)
    return xq @ w.T + b


def _time_batches(fn, n_warmup: int, n_iters: int) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / max(n_iters, 1)


def run_accuracy(*, eval_n: int) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import dscnn_mnist as dscnn
    import train_vit_mnist as tv

    device = torch.device("cpu")
    _x_tr, _y_tr, x_te, y_te = tv.load_mnist()
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
        batch_size=256,
        shuffle=False,
    )

    model = tv.TinyViT().to(device)
    tv.load_weights_into_model(model, tv.WEIGHTS_PATH, device)
    n_float = sum(p.numel() for p in model.parameters() if p.requires_grad)

    facc = tv.accuracy(model, te_loader, device, qat=False)
    qat_acc = tv.accuracy(model, te_loader, device, qat=True)
    n_np = len(y_te) if eval_n <= 0 else min(eval_n, len(y_te))
    print(f"numpy deploy-quant eval on {n_np} images...")
    qacc = tv.quant_numpy_accuracy(x_te, y_te, n_np if eval_n > 0 else 0)

    dscnn_m = dscnn.load_metrics() if DSCNN_METRICS.exists() else {}
    float_m = dscnn.load_model(DSCNN_WEIGHTS, device)
    n_dscnn = dscnn.count_params(float_m)
    te_loader_cnn = DataLoader(
        TensorDataset(
            torch.from_numpy(x_te.astype(np.float32)),
            torch.from_numpy(y_te.astype(np.int64)),
        ),
        batch_size=256,
        shuffle=False,
    )
    acc_f = dscnn.evaluate(float_m, te_loader_cnn, device, qat=False)
    acc_i = -1.0
    if DSCNN_INT8.exists():
        int8_m = dscnn.load_int8(DSCNN_INT8, device)
        acc_i = dscnn.evaluate(int8_m, te_loader_cnn, device, qat=True)

    w = vit.VitMnistWeights.load(vit.DEFAULT_WEIGHTS)
    vit_sz = _count_vit_deploy(w)
    cnn_sz = _count_dscnn_int8()

    return {
        "vit": {
            "float_dequant_acc": facc,
            "proxy_qat_acc": qat_acc,
            "deploy_numpy_acc": qacc,
            "deploy_numpy_n": n_np,
            "n_params_float": int(n_float),
            **vit_sz,
            "geometry": "T=16 D=16 L=2 patch=7 no-CNN-stem",
            "note": (
                "float_dequant = dequantized deploy npz (not a separate float teacher ckpt); "
                "deploy-FT tunes the numpy/FPGA path, so float can be << deploy"
            ),
        },
        "dscnn": {
            "float_acc": acc_f,
            "int8_acc": acc_i,
            "metrics_json": dscnn_m,
            "n_params_float": int(n_dscnn),
            **cnn_sz,
        },
    }


def run_bench(*, bit_path: str, bench_n: int, warmup: int) -> dict:
    """A9 timings: DS-CNN CPU int8 (numpy) + ViT FPGA."""
    import npukit_transformer as nt
    from npukit_matmul import open_device

    sample = np.load(vit.DEFAULT_SAMPLE)
    imgs = sample["images"][:bench_n].astype(np.float64)
    labels = sample["labels"][:bench_n]

    if not DSCNN_INT8.exists():
        raise FileNotFoundError(f"missing {DSCNN_INT8}")
    z = np.load(DSCNN_INT8)

    def cnn_one():
        dscnn_int8_numpy_forward(imgs[0], z)

    t_cnn_1_ms = 1000.0 * _time_batches(cnn_one, n_warmup=warmup, n_iters=16)

    t0 = time.perf_counter()
    for i in range(bench_n):
        dscnn_int8_numpy_forward(imgs[i], z)
    t_cnn_batch_ms = 1000.0 * (time.perf_counter() - t0) / bench_n

    w = vit.VitMnistWeights.load(vit.DEFAULT_WEIGHTS)

    def vit_cpu_one():
        vit.vit_forward(imgs[0], w, use_hw=False, verbose=False)

    t_vit_cpu_ms = 1000.0 * _time_batches(vit_cpu_one, n_warmup=2, n_iters=8)

    mmio, transport = open_device(bit_path)
    glue = nt.GlueDevice(mmio)

    def vit_hw_one():
        vit.vit_forward(
            imgs[0],
            w,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=True,
            verbose=False,
        )

    vit_hw_one()
    t_vit_hw_ms = 1000.0 * _time_batches(
        vit_hw_one, n_warmup=max(1, warmup // 2), n_iters=8
    )

    t0 = time.perf_counter()
    n_ok = 0
    for i in range(bench_n):
        _, d = vit.vit_forward(
            imgs[i],
            w,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=True,
            verbose=False,
        )
        n_ok += int(d["pred"][0] == int(labels[i]))
    t_vit_hw_batch_ms = 1000.0 * (time.perf_counter() - t0) / bench_n

    return {
        "bench_n": bench_n,
        "dscnn_int8_cpu_ms_per_image_single": t_cnn_1_ms,
        "dscnn_int8_cpu_ms_per_image_over_batch": t_cnn_batch_ms,
        "vit_deploy_numpy_cpu_ms_per_image": t_vit_cpu_ms,
        "vit_fpga_ms_per_image_single": t_vit_hw_ms,
        "vit_fpga_ms_per_image_over_batch": t_vit_hw_batch_ms,
        "vit_fpga_batch_label_acc": n_ok / bench_n,
        "note": (
            "A9 host; DS-CNN = numpy int8 on CPU (no torch); "
            "ViT FPGA includes host schedule + DMA/MMIO + glue"
        ),
    }


def _print_report(acc: dict | None, bench: dict | None) -> None:
    if acc:
        v, c = acc["vit"], acc["dscnn"]
        print("\n=== Accuracy (item 1: float vs deploy) ===")
        print(f"ViT float (dequant from npz, full 10k):  {100 * v['float_dequant_acc']:.2f}%")
        print(f"ViT proxy-QAT (full 10k):                {100 * v['proxy_qat_acc']:.2f}%")
        print(
            f"ViT deploy-quant numpy ({v['deploy_numpy_n']}):   "
            f"{100 * v['deploy_numpy_acc']:.2f}%"
        )
        gap = v["float_dequant_acc"] - v["deploy_numpy_acc"]
        print(f"float − deploy gap:                      {100 * gap:+.2f} pp")
        if gap > 0.02:
            print("→ Float teacher >> deploy: chase quant/scales/numerics.")
        elif gap < -0.02:
            print(
                "→ Deploy ≥ float(dequant): not a quant cliff; "
                "weights tuned for numpy/FPGA (capacity/front-end limited)."
            )
        else:
            print("→ Float ≈ deploy: capacity/front-end limited.")
        print(f"DS-CNN float (full 10k):                 {100 * c['float_acc']:.2f}%")
        if c["int8_acc"] >= 0:
            print(f"DS-CNN int8 (full 10k):                  {100 * c['int8_acc']:.2f}%")

        print("\n=== Parameters / weight footprint ===")
        print(
            f"DS-CNN: float params={c['n_params_float']:,}  "
            f"int8 elems={c['n_params_i8']:,}  "
            f"deploy≈{c['kib_deploy']:.2f} KiB (i8≈{c['kib_i8_only']:.2f} KiB)"
        )
        print(
            f"ViT:    float params={v['n_params_float']:,}  "
            f"i8 elems={v['n_params_i8']:,} + Q12={v['n_params_q12']:,}  "
            f"deploy≈{v['kib_deploy']:.2f} KiB (i8≈{v['kib_i8_only']:.2f} KiB)  "
            f"[{v['geometry']}]"
        )

    if bench:
        print("\n=== A9 inference latency ===")
        print(f"batch images: {bench['bench_n']}")
        print(
            f"DS-CNN int8 CPU  single-image:  {bench['dscnn_int8_cpu_ms_per_image_single']:.2f} ms/img"
        )
        print(
            f"DS-CNN int8 CPU  over batch:    {bench['dscnn_int8_cpu_ms_per_image_over_batch']:.2f} ms/img"
        )
        print(
            f"ViT deploy numpy CPU:           {bench['vit_deploy_numpy_cpu_ms_per_image']:.2f} ms/img"
        )
        print(
            f"ViT FPGA (single, steady):     {bench['vit_fpga_ms_per_image_single']:.2f} ms/img"
        )
        print(
            f"ViT FPGA (avg over batch):     {bench['vit_fpga_ms_per_image_over_batch']:.2f} ms/img"
        )
        print(
            f"ViT FPGA batch label acc:      {100 * bench['vit_fpga_batch_label_acc']:.1f}%"
        )
        print(f"note: {bench['note']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NpuKit edge-peer diagnostics")
    p.add_argument("--eval-n", type=int, default=0, help="numpy eval images; 0=full 10k")
    p.add_argument("--skip-acc", action="store_true")
    p.add_argument("--bench", action="store_true", help="A9 latency (DS-CNN CPU + ViT FPGA)")
    p.add_argument("--bit", default="")
    p.add_argument("--bench-n", type=int, default=64)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)

    acc = None if args.skip_acc else run_accuracy(eval_n=args.eval_n)
    bench = None
    if args.bench:
        bit = args.bit or "/home/xilinx/jupyter_notebooks/npukit.bit"
        bench = run_bench(bit_path=bit, bench_n=args.bench_n, warmup=args.warmup)
    _print_report(acc, bench)

    payload = {"accuracy": acc, "bench": bench}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

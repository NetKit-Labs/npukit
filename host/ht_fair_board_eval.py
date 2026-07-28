#!/usr/bin/env python3
"""Board/host eval: fair Hybrid Transformer int8 body (last-token head).

Usage:
  python3 host/ht_fair_board_eval.py
  python3 host/ht_fair_board_eval.py /path/to/npukit.bit
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import npukit_transformer as nt

HOST_DIR = Path(__file__).resolve().parent
DEPLOY = HOST_DIR / "speech_peers_fair_ht_deploy.npz"
OUT_JSON = HOST_DIR / "speech_peers_fair_board.json"


def load_blocks(z) -> list[nt.TinyBlockWeights]:
    n_layers = int(z["n_layers"])
    out = []
    for li in range(n_layers):
        out.append(
            nt.TinyBlockWeights(
                wq=z[f"l{li}_wq"],
                wk=z[f"l{li}_wk"],
                wv=z[f"l{li}_wv"],
                wo=z[f"l{li}_wo"],
                w1=z[f"l{li}_w1"],
                w2=z[f"l{li}_w2"],
                gamma1=z[f"l{li}_gamma1"],
                gamma2=z[f"l{li}_gamma2"],
                sw_wq=z[f"l{li}_sw_wq"],
                sw_wk=z[f"l{li}_sw_wk"],
                sw_wv=z[f"l{li}_sw_wv"],
                sw_wo=z[f"l{li}_sw_wo"],
                sw_w1=z[f"l{li}_sw_w1"],
                sw_w2=z[f"l{li}_sw_w2"],
            )
        )
    return out


def forward_one(
    tok_f32: np.ndarray,
    blocks: list[nt.TinyBlockWeights],
    w_cmd_i8: np.ndarray,
    sw_w_cmd: np.ndarray,
    *,
    mmio=None,
    transport=None,
    use_hw_gemm: bool = False,
) -> np.ndarray:
    x = nt.to_q12(tok_f32)
    for blk in blocks:
        x, _ = nt.transformer_block_1layer(
            x,
            blk,
            mmio=mmio,
            transport=transport,
            use_hw=False,
            use_hw_gemm=use_hw_gemm,
            glue_mode="float",
            scale_act=64.0,
            scale_p=127.0,
            causal=False,
            verbose=False,
        )
    # Fair Hybrid Transformer: last-token head (right-aligned speech).
    pooled = np.asarray(x[-1], dtype=np.int32)
    # N_CMD=16 is 8-aligned; keep head on CPU for a clean body vs head split.
    logits = nt._matmul_q12(
        pooled.reshape(1, -1),
        w_cmd_i8,
        mmio=None,
        transport=None,
        use_hw=False,
        scale_act=64.0,
        scale_w=sw_w_cmd,
    ).reshape(-1)
    return logits


def bench(fn, n: int = 10) -> float:
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1e3 / n


def eval_pack(*, use_hw_gemm: bool, mmio=None, transport=None, max_n: int = 32) -> tuple[float, float]:
    z = np.load(DEPLOY, allow_pickle=True)
    tokens = z["tokens"]
    labels = z["labels"]
    blocks = load_blocks(z)
    w_cmd_i8 = z["w_cmd_i8"]
    sw = z["sw_w_cmd"]
    n = min(len(labels), max_n)
    ok = 0
    for i in range(n):
        if (i + 1) % 8 == 0:
            print(f"  … {i+1}/{n}", flush=True)
        logits = forward_one(
            tokens[i],
            blocks,
            w_cmd_i8,
            sw,
            mmio=mmio,
            transport=transport,
            use_hw_gemm=use_hw_gemm,
        )
        ok += int(int(np.argmax(logits)) == int(labels[i]))
    acc = ok / max(n, 1)
    ms = bench(
        lambda: forward_one(
            tokens[0],
            blocks,
            w_cmd_i8,
            sw,
            mmio=mmio,
            transport=transport,
            use_hw_gemm=use_hw_gemm,
        ),
        n=5 if use_hw_gemm else 8,
    )
    return acc, ms


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bit = argv[0] if argv else None
    if not DEPLOY.is_file():
        print(f"missing {DEPLOY}; run host/export_fair_ht_deploy.py")
        return 1

    z = np.load(DEPLOY, allow_pickle=True)
    torch_acc = float(z["torch_acc"]) if "torch_acc" in z.files else float("nan")
    print(f"deploy T={z['t']} D={z['d']} L={z['n_layers']} n={len(z['labels'])}")
    print(f"host Torch ref (export subset): {100*torch_acc:.1f}%")

    print("=== Hybrid Transformer int8 body (CPU tiled) ===", flush=True)
    acc_cpu, ms_cpu = eval_pack(use_hw_gemm=False, max_n=64)
    print(f"  acc={100*acc_cpu:.1f}%  ms={ms_cpu:.2f}", flush=True)

    acc_fpga = ms_fpga = None
    if bit:
        print(f"=== Hybrid Transformer int8 body (FPGA) {bit} ===", flush=True)
        from npukit_matmul import open_device

        mmio, transport = open_device(bit)
        # Smaller n — T=128×L=6 over AXI is slow.
        acc_fpga, ms_fpga = eval_pack(
            use_hw_gemm=True, mmio=mmio, transport=transport, max_n=16
        )
        print(f"  acc={100*acc_fpga:.1f}%  ms={ms_fpga:.2f}", flush=True)
        print(f"  vs CPU int8: {ms_cpu/max(ms_fpga,1e-9):.2f}×", flush=True)
    else:
        print("pass bitstream to enable FPGA", flush=True)

    payload = {
        "model": "Hybrid Transformer (fair order-only)",
        "geometry": f"T={int(z['t'])} D={int(z['d'])} L={int(z['n_layers'])}",
        "head": "last-token",
        "torch_acc_export_subset": torch_acc,
        "cpu_i8_acc": acc_cpu,
        "cpu_i8_ms": ms_cpu,
        "fpga_i8_acc": acc_fpga,
        "fpga_i8_ms": ms_fpga,
        "bit": bit,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

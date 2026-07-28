#!/usr/bin/env python3
"""Board/host peer-C int8 GEMM: CPU tiled vs FPGA (numpy only, no Torch).

Usage:
  python3 host/speech_peers_fpga_eval.py                  # CPU only
  python3 host/speech_peers_fpga_eval.py /path/to/npukit.bit
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import npukit_transformer as nt

HOST_DIR = Path(__file__).resolve().parent
DEPLOY = HOST_DIR / "speech_peers_c_deploy.npz"
METRICS = HOST_DIR / "speech_peers_metrics.json"


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
    pooled = np.rint(x.astype(np.float64).mean(axis=0)).astype(np.int32)
    # N_CMD=22 is not 8-aligned → always CPU for the intent head (body may be FPGA).
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


def bench(fn, n: int = 20) -> float:
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1e3 / n


def eval_pack(
    *,
    use_hw_gemm: bool,
    mmio=None,
    transport=None,
    max_n: int | None = None,
) -> tuple[float, float]:
    z = np.load(DEPLOY)
    tokens = z["tokens"]
    labels = z["labels"]
    blocks = load_blocks(z)
    w_cmd_i8 = z["w_cmd_i8"]
    sw = z["sw_w_cmd"]
    n = len(labels) if max_n is None else min(len(labels), max_n)
    ok = 0
    for i in range(n):
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
        n=10 if use_hw_gemm else 20,
    )
    return acc, ms


def patch_metrics(acc_cpu: float, ms_cpu: float, acc_fpga: float | None, ms_fpga: float | None, bit: str | None) -> None:
    if not METRICS.is_file():
        return
    d = json.loads(METRICS.read_text(encoding="utf-8"))
    peers = {p["name"]: p for p in d["peers"]}
    if "C_npukit_cpu_i8" in peers:
        peers["C_npukit_cpu_i8"]["accuracy"] = acc_cpu
        peers["C_npukit_cpu_i8"]["ms_per_phrase"] = ms_cpu
        peers["C_npukit_cpu_i8"]["notes"] = "deploy-pack int8 tiled GEMM (board/host numpy)"
    if acc_fpga is not None and "C_npukit_fpga_i8" in peers:
        peers["C_npukit_fpga_i8"]["accuracy"] = acc_fpga
        peers["C_npukit_fpga_i8"]["ms_per_phrase"] = ms_fpga if ms_fpga is not None else -1.0
        peers["C_npukit_fpga_i8"]["notes"] = f"PL GEMM via {bit}"
    d["peers"] = list(peers.values())
    d["bit"] = bit
    METRICS.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(f"updated {METRICS}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bit = argv[0] if argv else None
    if not DEPLOY.is_file():
        print(f"missing {DEPLOY}; run host/export_speech_peer_c_deploy.py first")
        return 1

    print("=== C NpuKit CPU int8 (deploy pack) ===")
    acc_cpu, ms_cpu = eval_pack(use_hw_gemm=False)
    print(f"  acc={100*acc_cpu:.1f}%  ms={ms_cpu:.2f}")

    acc_fpga = ms_fpga = None
    if bit:
        print(f"=== C NpuKit FPGA int8 ({bit}) ===")
        from npukit_matmul import open_device

        mmio, transport = open_device(bit)
        acc_fpga, ms_fpga = eval_pack(use_hw_gemm=True, mmio=mmio, transport=transport)
        print(f"  acc={100*acc_fpga:.1f}%  ms={ms_fpga:.2f}")
        print(f"  speedup vs CPU int8: {ms_cpu / max(ms_fpga, 1e-9):.2f}×")
    else:
        print("pass bitstream path to enable FPGA")

    patch_metrics(acc_cpu, ms_cpu, acc_fpga, ms_fpga, bit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

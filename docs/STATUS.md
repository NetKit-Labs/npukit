# NpuKit status

Checkpoint after MNIST tiny-ViT **T=16 × D=16 × L=2** + per-stage quant scales (2026-07-26).

## Where we are

**Hardware MVP is complete.** The Zynq-7020 bitstream (`VERSION 0x300`) provides:

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz |
| Host (A9) | Orchestration, per-stage quant/scales, RoPE / masks / reshape |

Fabric clock: **PS FCLK0 @ 100 MHz**. Latest rebuild closed timing (**WNS ≈ +0.83 ns** post-route). Glue Softmax length **16** works on this bit (`len_r` holds `MAX_LEN`).

## Board results (saved notebooks)

| Notebook | What it covers | Result |
|----------|----------------|--------|
| `host/npukit_matmul.ipynb` | GEMM unit suite: classic 8×8 + tiled 16×16 / 32×32 | **12/12 PASS** |
| `host/npukit_transformer.ipynb` | Glue unit ops + GEMM sanity on same bit | **ALL BOARD PASS** |
| `host/npukit_transformer_e2e.ipynb` | Synthetic 1-layer block, T=8×D=8, fixed scales | **ALL E2E PASS** (ref + board) |
| `host/npukit_vit_mnist.ipynb` | MNIST tiny-ViT T=16×D=16×L=2, per-stage QAT | **ALL VIT PASS** |

CLI on the board (same bit):

```bash
python npukit_matmul.py /home/xilinx/jupyter_notebooks/npukit.bit 32 32 32
python npukit_transformer.py /home/xilinx/jupyter_notebooks/npukit.bit
python npukit_transformer.py /home/xilinx/jupyter_notebooks/npukit.bit --e2e
python npukit_vit_mnist.py /home/xilinx/jupyter_notebooks/npukit.bit -n 64
```

## MNIST tiny-ViT (current)

Geometry (no resize):

- Native **28×28**, patch **7** → **T=16** tokens
- Patch vector **49** zero-padded to **56** (GEMM 8-alignment)
- Model dim **D=16** (glue `MAX_LEN`; GEMM host-tiles 8×8)
- **2 transformer layers** host-scheduled on the same GEMM + glue (time-multiplexed)
- **Per-stage quant scales**: embed / block0 / block1 / cls (`act`, `w`, `p` where applicable)
- Class head on CPU (`N_CLASS=10` not 8-aligned)

Train: `host/train_vit_mnist.py` — full MNIST 60k + ± aug → float warm-up → **per-stage calibration** → STE QAT → `vit_mnist_weights.npz`.

| Metric | Result |
|--------|--------|
| Float test (full) | ~**95.1%** |
| QAT-mode test (full) | ~**95.0%** |
| Numpy quantized ref (2048) | ~**80%** (trails QAT; Softmax/RMSNorm + Q12 depth) |
| Board sample n=64 | ref **56/64 (87.5%)**, hw **55/64 (85.9%)** |
| ref↔hw pred agree | **63/64 (98.4%)** |
| Numeric check | **ALL VIT PASS** (tight L0 abs tol; L1+ abs drift reported; ≥90% pred agree) |

“ALL VIT PASS” means FPGA tracks the quantized host path under the smoke gates — **not** 100% classification accuracy.

Example calibrated scales (order-of-magnitude; see npz): embed act/w ≈ 72/336, L0 ≈ 40/216/315, L1 ≈ 13/139/160, cls ≈ 16/130 — L0 vs L1 act differs a lot, which is why per-stage scales matter.

## Geometry notes

- GEMM tile is always **8×8**; larger matmuls are host-tiled.
- Glue vector length: unit tests **4**, e2e **8**, ViT Softmax / row ops **16**, hardware cap **16**.
- Softmax length tracks sequence **T**; residual/GELU/RMSNorm length track model dim **D** (both ≤ `MAX_LEN`).
- Multi-head attention does **not** require larger `MAX_LEN`; add **per-head scales** when heads are introduced.

## Next path

1. **Per-head scales** when multi-head attention is added
2. Optional CNN stem / longer QAT if chasing higher MNIST label accuracy
3. **Defer:** PE-grid growth, ping-pong, depthwise, RTL wider than D=16

Related docs: [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

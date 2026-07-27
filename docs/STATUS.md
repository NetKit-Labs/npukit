# NpuKit status

Checkpoint after board re-smoke + host DS-CNN peer (T=16 × D=16 × L=2, per-stage scales) (2026-07-27).

## Where we are

**Hardware MVP is complete.** The Zynq-7020 bitstream (`VERSION 0x300`) provides:

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz |
| Host (A9) | Orchestration, per-stage quant/scales, deploy-faithful train, RoPE / masks / reshape |

Fabric clock: **PS FCLK0 @ 100 MHz**. Latest rebuild closed timing (**WNS ≈ +0.83 ns** post-route). Glue Softmax length **16** works on this bit (`len_r` holds `MAX_LEN`).

## Board results (saved notebooks)

| Notebook | What it covers | Result |
|----------|----------------|--------|
| `host/npukit_matmul.ipynb` | GEMM unit suite: classic 8×8 + tiled 16×16 / 32×32 | **12/12 PASS** |
| `host/npukit_transformer.ipynb` | Glue unit ops + GEMM sanity on same bit | **ALL BOARD PASS** |
| `host/npukit_transformer_e2e.ipynb` | Synthetic 1-layer block, T=8×D=8, fixed scales | **ALL E2E PASS** (ref + board) |
| `host/npukit_vit_mnist.ipynb` | MNIST tiny-ViT T=16×D=16×L=2, deploy-faithful QAT | **ALL VIT PASS** |
| `host/dscnn_mnist.ipynb` | Host-only DS-CNN MNIST reference (not on FPGA) | float **98.00%** / int8 **98.39%** |
| `host/npukit_board_smoke.ipynb` | One-shot matmul → glue → e2e → ViT | **ALL SMOKE PASS** |

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

Train: `host/train_vit_mnist.py` — float warm-up → per-stage calibration → proxy STE QAT → **deploy-faithful fine-tune** (CE on board-ref numpy logits; STE grads via proxy) → `vit_mnist_weights.npz`.

| Metric | Result |
|--------|--------|
| **Numpy quantized ref (full 10k test)** | **94.28%** |
| Board sample n=64 labels | ref **60/64 (93.8%)**, hw **61/64 (95.3%)** |
| ref↔hw pred agree | **63/64 (98.4%)** |
| Numeric check | **ALL VIT PASS** (tight L0; ≥90% pred agree) |
| Float / proxy-QAT test | not the deploy metric (weights tuned for numpy/FPGA path) |

“ALL VIT PASS” means FPGA tracks the quantized host path under the smoke gates — **not** 100% classification accuracy.

Example calibrated scales (order-of-magnitude; see npz): embed act/w ≈ 72/336, L0 ≈ 40/216/315, L1 ≈ 13/139/160, cls ≈ 16/130 — L0 vs L1 act differs a lot, which is why per-stage scales matter.

## Edge comparison intent (MCU vs MCU+accelerator)

Two **deployment-shaped** peers on the same MNIST task — not a param-matched bake-off:

| Role | Model | What it mimics | Headline |
|------|--------|----------------|----------|
| **MCU-class CNN** | Host DS-CNN (int8) | TinyML DW/PW CNN on Cortex-M / TFLite Micro–class MCU | **98.39%** / ~**8.3 KiB** |
| **MCU/MPU + accelerator ViT** | Tiny-ViT on NpuKit | Micro transformer scheduled on GEMM + glue (T=16×D=16×L=2) | **94.28%** / ~**4.3 KiB** |

Do **not** force equal layer counts or equal params. Compare **task accuracy + weight footprint + where compute runs** (CNN = host MCU-shaped; ViT = FPGA path). Full bring-up: `host/npukit_board_smoke.py` / `.ipynb`.

## DS-CNN host reference (MCU-class peer)

Host-only TinyML-style DS-CNN — **not** a ViT CNN stem and **not** mapped to the FPGA. Int8 path is the headline number (MCU deploy shape).

Pipeline: float train → BN-fold → per-tensor calibrate → STE QAT → `dscnn_mnist_int8.npz`.

| Item | Value |
|------|--------|
| Train / eval | `host/train_dscnn_mnist.py`, `host/dscnn_mnist.py`, `host/dscnn_mnist.ipynb` |
| Weights / metrics | `dscnn_mnist_weights.pt`, `dscnn_mnist_int8.npz`, `dscnn_mnist_metrics.json` |
| Params / int8 weights | **~9.0k** / **~8.3 KiB** |
| Float test (full 10k) | **98.00%** |
| **Int8 test (full 10k)** | **98.39%** (MCU-shaped headline) |
| ViT deploy-quant (full 10k) | **94.28%** (~4.3 KiB; FPGA path) |

Headline compare: **MCU DS-CNN int8** vs **MCU+NpuKit ViT deploy-quant** (accuracy + KiB + compute locus).

## Geometry notes

- GEMM tile is always **8×8**; larger matmuls are host-tiled.
- Glue vector length: unit tests **4**, e2e **8**, ViT Softmax / row ops **16**, hardware cap **16**.
- Softmax length tracks sequence **T**; residual/GELU/RMSNorm length track model dim **D** (both ≤ `MAX_LEN`).
- Multi-head attention does **not** require larger `MAX_LEN`; add **per-head scales** when heads are introduced.

## Next path

1. Keep both peers **deployment-shaped** (MCU DS-CNN vs MCU+accel ViT); optional longer ViT deploy-FT for FPGA labels
2. **Per-head scales** when multi-head attention is added
3. **Defer:** PE-grid growth, ping-pong, depthwise / CNN on FPGA, RTL wider than D=16

Related docs: [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

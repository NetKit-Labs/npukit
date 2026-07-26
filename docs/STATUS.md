# NpuKit status

Checkpoint after T=16 ViT + glue `len==MAX_LEN` bitstream (2026-07-26).

## Where we are

**Hardware MVP is complete.** The Zynq-7020 bitstream (`VERSION 0x300`) provides:

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz |
| Host (A9) | Orchestration, quant/scales, RoPE / masks / reshape |

Fabric clock: **PS FCLK0 @ 100 MHz**. Latest rebuild closed timing (**WNS ≈ +0.83 ns** post-route). Glue Softmax length **16** works on this bit (`len_r` holds `MAX_LEN`).

## Board results (saved notebooks)

| Notebook | What it covers | Result |
|----------|----------------|--------|
| `host/npukit_matmul.ipynb` | GEMM unit suite: classic 8×8 + tiled 16×16 / 32×32 | **12/12 PASS** |
| `host/npukit_transformer.ipynb` | Glue unit ops + GEMM sanity on same bit | **ALL BOARD PASS** |
| `host/npukit_transformer_e2e.ipynb` | Synthetic 1-layer block, T=8×D=8, fixed scales | **ALL E2E PASS** (ref + board) |
| `host/npukit_vit_mnist.ipynb` | MNIST tiny-ViT T=16×D=8, trained QAT weights | **ALL VIT PASS** |

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
- Model dim **D=16** (glue max; GEMM host-tiles); **2 layers** host-scheduled on same GEMM+glue
- Class head on CPU (`N_CLASS=10`)

Train path: full MNIST 60k + shift aug → float warm-up → scale calibration → STE QAT → export scales in `vit_mnist_weights.npz`.

| Metric | Result |
|--------|--------|
| Float test (full) | ~**95.2%** |
| QAT-mode test (full) | ~**94.9%** |
| Numpy quantized ref (2048) | ~**79%** (deeper int8 path; QAT fake-quant stays high) |
| Board sample n=64 | ref **56/64**, hw **52/64**; ref↔hw pred agree **60/64 (93.8%)** |
| Numeric check | **ALL VIT PASS** (tight L0; L2 abs drift reported; ≥90% pred agree) |

“ALL VIT PASS” means FPGA matches the quantized ref within tolerance — **not** 100% classification accuracy.

## Geometry notes

- GEMM tile is always **8×8**; larger matmuls are host-tiled.
- Glue vector length: unit tests **4**, e2e **8**, ViT Softmax **16**, hardware cap **16**.
- Multi-head attention does **not** require larger `MAX_LEN`.

## Next path

1. Longer train / more QAT if chasing higher MNIST accuracy
2. Optional full test-set accuracy via board (slow)
3. **Defer:** PE-grid growth, ping-pong, depthwise, wider D

Related docs: [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

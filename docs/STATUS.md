# NpuKit status

Checkpoint after board re-smoke of deploy-track ViT (richer DS-stem, L=4, MLP=32, per-channel weights, A9 float Softmax/RMSNorm/GELU) (2026-07-27).

## Where we are

**Hardware MVP is complete.** The Zynq-7020 bitstream (`VERSION 0x300`) provides:

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz (optional path) |
| Host (A9) | Orchestration, per-channel quant/scales, DS-stem, float Softmax/RMSNorm/GELU |

Fabric clock: **PS FCLK0 @ 100 MHz**. Latest rebuild closed timing (**WNS ≈ +0.83 ns** post-route).

## Board results (2026-07-27 re-smoke)

Captured in [`host/npukit_board_smoke.ipynb`](../host/npukit_board_smoke.ipynb) on `192.168.0.215` with current `vit_mnist_weights.npz`.

| Notebook / suite | What it covers | Result |
|------------------|----------------|--------|
| `host/npukit_board_smoke.ipynb` | matmul → glue → e2e → ViT | **ALL SMOKE PASS** (4/4) |
| matmul | classic 8×8 + tiled 16/32 | **12/12 PASS** |
| glue | unit ops + GEMM tile | **ALL BOARD PASS** |
| e2e | synthetic 1-layer T=8×D=8 | **ALL E2E PASS** |
| vit | DS-stem + T=16×D=16×MLP32×L=4, `glue=float`, int8 GEMM HW | **ALL VIT PASS** |
| `host/dscnn_mnist.ipynb` | Host-only DS-CNN peer (not on FPGA) | float **98.00%** / int8 **98.39%** |

ViT board sample **n=64**:

| Metric | Result |
|--------|--------|
| ref accuracy | **62/64 (96.9%)** |
| hw accuracy | **62/64 (96.9%)** |
| ref↔hw pred agree | **64/64 (100%)** |
| tensor max\|err\| | **0** (HW GEMM matches numpy GEMM with float norms) |

CLI:

```bash
python npukit_board_smoke.py /home/xilinx/jupyter_notebooks/npukit.bit --vit-n 64
python npukit_vit_mnist.py /home/xilinx/jupyter_notebooks/npukit.bit -n 64
```

## MNIST tiny-ViT (current)

Geometry (no resize):

- Native **28×28** → **CPU richer DS-stem** (stem + 2× DW/PW) → **T=16×D=16** tokens
- Model dim **D=16**, FFN hidden **MLP=32**, **L=4** transformer blocks
- **int8 GEMM** on FPGA; **A9 float32** Softmax / RMSNorm / GELU (`glue_mode=float`)
- **Per-channel weight scales** on stem + GEMM mats; per-stage act / attn-p scales
- Class head on CPU (`N_CLASS=10` not 8-aligned)
- Smoke uses **numpy int8 stem** (TFLite on A9 can SIGILL on `invoke`)

| Metric | Result |
|--------|--------|
| **Numpy quantized ref (full 10k test)** | **97.40%** |
| Float test (full 10k) | **96.90%** |
| Proxy QAT test (full 10k) | **97.38%** |
| Proxy ↔ numpy gap (post-QAT) | **~0.6 pp** |
| Learnable params (active path) | **~9.8k** (stem 1.0k + blocks 8.3k + pos/cls); ~10.7k incl. legacy `w_pe` |
| Weight footprint (int8 + Q12) | **~11.5 KiB** |
| Board sample n=64 | ref/hw **96.9%**, agree **100%**, **ALL VIT PASS** |

## Edge comparison

| Role | Model | Headline |
|------|--------|----------|
| **MCU-class CNN** | Host DS-CNN (int8) | **98.39%** / ~**8.3 KiB** / **~9.0k** params |
| **MCU/MPU + accelerator ViT** | Tiny-ViT on NpuKit | **97.40%** / ~**11.5 KiB** / **~9.8k** active params |

Animated summary: [`viz/out/edge_peers.gif`](../viz/out/edge_peers.gif). Five-minute board path: [`host/board_bringup_5min.sh`](../host/board_bringup_5min.sh).

## What is complete

- GEMM + glue RTL/bit @ 100 MHz, board unit + e2e PASS
- Deploy-faithful tiny-ViT: CPU DS-stem + FPGA int8 GEMM + A9 float norms
- Per-channel quant, stem-matched QAT, numpy deploy **97.40%**
- Full board smoke with new weights: **ALL SMOKE PASS**, ViT ref↔hw **100%** agree
- Host DS-CNN peer, Netron graphs, edge-peers GIF, 5-min bring-up script

## Still optionally valuable

1. Close the last **~1 pp** vs DS-CNN (slightly richer stem / longer float — not required for the story)
2. Optional `d_ff=4×` experiment (larger KiB; uncertain gain)
3. Depthwise / CNN path in RTL — only if profiling shows stem/DW dominate wall-clock
4. PE-grid growth, ping-pong, HW glue for ViT deploy — deferred (float-on-A9 is winning for accuracy)
5. Multi-head attention + per-head scales

## Geometry notes

- GEMM tile is always **8×8**; larger matmuls are host-tiled (`MLP=32` is fine).
- HW glue Softmax/RMSNorm/GELU stay capped at `MAX_LEN=16`; ViT deploy uses **float** norms so FFN hidden may exceed 16.
- Multi-head attention does **not** require larger `MAX_LEN`.

Related docs: [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

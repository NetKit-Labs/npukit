# NpuKit status

Checkpoint after PYNQ-Z2 board re-verification (2026-07-26).

## Where we are

**Hardware MVP is complete.** The Zynq-7020 bitstream (`VERSION 0x300`) provides:

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz |
| Host (A9) | Orchestration, quant/scales, RoPE / masks / reshape |

Fabric clock: **PS FCLK0 @ 100 MHz**. Latest rebuild closed timing (**WNS ≈ +0.69 ns**). Glue `MAX_LEN=16` is timing headroom (current smoke peaks at length **8**); keep 16 unless utilization forces a shrink.

## Board results (saved notebooks)

All three host notebooks talk to the **real PL** (not Icarus-only). Offline/ref cells exist inside them for golden checks; board cells load `npukit.bit`.

| Notebook | What it covers | Result |
|----------|----------------|--------|
| `host/npukit_matmul.ipynb` | GEMM unit suite: classic 8×8 + tiled 16×16 / 32×32 | **12/12 PASS** |
| `host/npukit_transformer.ipynb` | Glue unit ops + GEMM sanity on same bit | **ALL BOARD PASS** |
| `host/npukit_transformer_e2e.ipynb` | Synthetic 1-layer block, T=8×D=8, fixed scales | **ALL E2E PASS** (ref + board) |

CLI equivalents on the board (same bit):

```bash
python npukit_matmul.py /home/xilinx/jupyter_notebooks/npukit.bit 32 32 32
python npukit_transformer.py /home/xilinx/jupyter_notebooks/npukit.bit
python npukit_transformer.py /home/xilinx/jupyter_notebooks/npukit.bit --e2e
```

Icarus sims under `sim/` remain host-side RTL checks; they are separate from these notebooks.

## Geometry note

- GEMM tile is always **8×8**; larger matmuls (e.g. 32×32) are **host-tiled**, not a bigger array.
- Glue vector length used today: unit tests **4**, e2e **8**, hardware cap **16**.
- Multi-head attention does **not** require larger `MAX_LEN`; Softmax length tracks sequence **T**, not head count.

## Next path

1. **MNIST tiny-ViT (host)** — real patches/weights, quant/dequant with fixed scales, drive existing GEMM + glue. RoPE / masks / reshape stay on the A9.
2. Grow sequence/dim via **host tiling** first; only raise glue `MAX_LEN` if a single vector must exceed 16.
3. **Defer** PE-grid growth, ping-pong SRAM, depthwise engines until the model path works.

Related docs: [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

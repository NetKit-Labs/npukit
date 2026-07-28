# Weight-stationary + A ping-pong (v3.1)

`VERSION 0x301` / `FEATURES` bits:

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | GEMM | 8×8 int8 array |
| 1 | GLUE | Softmax / RMSNorm / residual / GELU |
| 2 | WS | A-only / B-only AXIS loads (`LOAD_CFG`) |
| 3 | PP | Dual-A banks; shadow fill while GEMM busy |

## Why

Legacy kicks always streamed **A∥B** (32×`uint32`) and only accepted AXIS when idle. That re-fetched weights every tile and serialized DMA behind compute — fine for bring-up, bad for latency on the A9.

## Host contract

`LOAD_CFG` @ `0x028`:

| Value | AXIS payload |
|------:|--------------|
| 0 | A then B (32 words) — legacy default |
| 1 | A only (16 words) |
| 2 | B only (16 words) |

Weight-stationary matmul (C++ `Device::matmul_i8` when `FEAT_WS`):

1. For each output tile `(i0,j0)` and each K-tile `k0`: load **B** (`LOAD_B`), load **A** (`LOAD_A`) unless already in the shadow bank.
2. `CTRL` start (clear on first K-step).
3. If `FEAT_PP` and another K-tile remains: start next **A** DMA into the shadow bank **while** GEMM runs; wait for both GEMM done and MM2S idle.

Idle A fills update the active read bank; busy A-only fills update the inactive bank and set `STATUS[5]` (`shadow_a_ready`). `START` consumes the shadow (swap banks).

## C++ driver

- Prefers **XRT/zocl CMA** BOs (two TX + one RX) on PYNQ 3.x; legacy `/dev/xlnk` if present; else MMIO.
- Old bitstreams (`VERSION < 0x301`, no `FEAT_WS`) keep the A∥B path automatically.

## Host policy (C++)

Default matmul path stays **legacy A∥B** (fastest on this 8×8 @ 100 MHz).

Opt into WS+PP with:

```bash
NPUKIT_WS_PP=1 sudo ./npukit_vit --weights vit_mnist.bin
```

(Requires `K > 8` so a shadow A prefetch exists.) Measured on tiny-ViT: forced WS+PP is parity-clean but **slower** than A∥B because MM2S setup dominates ~22-cycle compute — see board note.

## Board note (tiny-ViT)

WS+PP RTL/driver is in (`VERSION 0x301`, `FEATURES` includes `WS|PP`). For this MNIST model, keep the default A∥B path (~9–10 ms e2e). Next wins: layer-resident `W`, larger PE / `K`, or bigger demo model.

## What this does *not* do yet

- B is still single-buffered (cannot prefetch B during compute).
- Weights are not stored for a whole layer in BRAM — only the current 8×8 B tile is held.
- Full layer weight-stationary (preload all `W` once) is a follow-on.

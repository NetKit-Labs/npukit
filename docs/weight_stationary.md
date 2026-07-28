# Weight-stationary + layer-resident weights

`VERSION 0x302` / `FEATURES` bits:

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | GEMM | 8×8 int8 array |
| 1 | GLUE | Softmax / RMSNorm / residual / GELU |
| 2 | WS | A-only / B-only AXIS loads (`LOAD_CFG`) |
| 3 | PP | Dual-A banks; shadow fill while GEMM busy |
| 4 | WMEM | Layer-resident weight bank (full `K×N` in BRAM) |

## Layer-resident weights (`FEAT_WMEM`)

On-chip `w_mem` is a **RAMB18** word bank (≤1024 int8, e.g. 32×32). Host contract:

1. `W_SHAPE` @ `0x02C`: `[15:0]=K`, `[31:16]=N`
2. `LOAD_CFG=3` (`LOAD_W`), DMA row-major `K×N` int8 once
3. For each output tile / K-tile: set `TILE_KJ` @ `0x030`, DMA **A-only**, `CTRL` with `USE_WMEM` (`bit3`) + `START`

RTL copies the selected 8×8 from `w_mem` → `b_mem` (sync BRAM read + BFILL) before the systolic feed.

### Host policy (board-measured)

On tiny-ViT @ 100 MHz, WMEM is **functionally correct** (`max|err|=0`) but **not faster** than A∥B (BFILL + split DMA overhead). Default stays legacy A∥B. Opt in:

```bash
NPUKIT_WMEM=1 sudo ./npukit_vit --weights vit_mnist.bin
```

| Path | Result |
|------|--------|
| ViT e2e WMEM | ~10.0 ms/img |
| ViT e2e A∥B (`NPUKIT_WMEM` unset) | **~9.8 ms/img** |
| 320× GEMM 8×8 WMEM | 8.8 ms |
| 320× GEMM 8×8 A∥B | **6.4 ms** |

Saved run: [`results/wmem_20260728T133256Z/`](../results/wmem_20260728T133256Z/).

## Tile-level WS + A ping-pong (opt-in)

`LOAD_CFG` @ `0x028`:

| Value | AXIS payload |
|------:|--------------|
| 0 | A then B (32 words) — legacy default |
| 1 | A only (16 words) |
| 2 | B only (16 words) |
| 3 | Full weight bank (`K*N` bytes, padded to words) |

```bash
NPUKIT_WS_PP=1 sudo ./npukit_vit --weights vit_mnist.bin
```

## Notes

- Timing for the WMEM bitstream was **WNS ≈ −1.57 ns** at 100 MHz in the smoke build (still functionally clean on board). Closing timing is follow-on.
- Old bitstreams (`VERSION < 0x302`) keep A∥B automatically.

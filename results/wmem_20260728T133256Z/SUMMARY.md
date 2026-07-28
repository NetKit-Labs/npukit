# WMEM board smoke wmem_20260728T133256Z

## Bitstream
- VERSION `0x302`, FEATURES `0x1F` (GEMM|GLUE|WS|PP|WMEM)
- `w_mem` inferred as **RAMB18** (256×32)
- Timing: **WNS = -1.574 ns** at 100 MHz (not closed; functional OK)

## Functional
- GEMM 8×8 PASS
- ViT e2e: agree_argmax **8/8**, max|err|=**0** (WMEM and legacy)

## Latency (PYNQ-Z2, XRT CMA)
| Path | Result |
|------|--------|
| 320× GEMM 8×8 WMEM | 8.79 ms (0.03 ms/kick) |
| 320× GEMM 8×8 legacy A∥B (`NPUKIT_WMEM=0`) | **6.42 ms** (0.02 ms/kick) |
| ViT e2e WMEM | 9.97 ms/img |
| ViT e2e legacy A∥B | **9.83 ms/img** |

WMEM is correct but not faster on this tiny-ViT (BFILL + A-only DMA overhead dominates). Host default should stay A∥B; opt in with `NPUKIT_WMEM=1`.

# Hardware Verification and Performance Report — NPUKit

**Target:** Zynq-7020 programmable logic + Arm Cortex-A9 host  
**Evidence review:** 2026-10-25; persisted FPGA bring-up and benchmark records (**no new board run**)  
**Scope:** 8×8 INT8→INT32 systolic GEMM, DMA/AXI-Lite, transformer glue, optional weight memory, software/FPGA comparison.

## 1. Verification setup

Board implementation contains an **8×8 INT8 systolic array**, AXI-Lite control and AXI DMA data transport, plus transformer operations. The board's fabric clock is configured at **100 MHz**. The optional layer-resident weight memory (WMEM) uses an inferred **RAMB18**, while the default host path still transfers A and B conventionally. See [README](../README.md), [status](STATUS.md), and [weight-stationary design](weight_stationary.md).

## 2. Recorded results, including non-passing timing

| Evidence / requirement | Measured or reported result | Assessment |
| --- | --- | --- |
| Board smoke for matmul, glue, end-to-end, tiny-ViT | **ALL SMOKE PASS**, from maintained status | On-board functional check reported; [status](STATUS.md) |
| WMEM GEMM **8×8** | **PASS** | [Saved WMEM smoke summary](../results/wmem_20260728T133256Z/SUMMARY.md) |
| WMEM vs legacy tiny-ViT parity | Argmax agree **8/8**, maximum absolute error **0** | Narrow recorded test, not exhaustive numerical verification |
| 320 × GEMM 8×8 WMEM | **8.79 ms** total (reported **0.03 ms/kick**) | Slower than legacy transfer path on this test |
| 320 × GEMM 8×8 legacy A+B | **6.42 ms** total (reported **0.02 ms/kick**) | Retained faster default |
| tiny-ViT end-to-end WMEM vs legacy | **9.97 vs 9.83 ms/image**, respectively | No speedup from WMEM on measured workload |
| FPGA static-timing report | **WNS = −1.574 ns** at nominal **100 MHz** | **TIMING NOT MET** despite functional smoke pass |
| Placed WMEM design resources | **9,333 LUT, 9,865 FF, RAMB18=1** (WMEM) | [Timing/resource artifact](../results/wmem_20260728T133256Z/timing_summary.txt) |
| Fair speech transformer body on Zynq board, precomputed tokens | Arm CPU INT8 GEMM **~1.97 s/phrase** vs FPGA offload **~30.94 s/phrase** | FPGA **~15.7× slower** for this workload, not a product-level acceleration claim |
| MNIST tiny-ViT verification | Reported ~**98.0%** numpy deployment accuracy, and **100%** board prediction agreement at **n=64** | Digit-model **bring-up sanity check**, not final audio-product accuracy |
| GSC fair-order-only classification | **96.5%** host result for hybrid transformer | **Host evaluation**, not full live-audio FPGA deployment |

WMEM and timing details: [WMEM snapshot](../results/wmem_20260728T133256Z/SUMMARY.md), [timing text](../results/wmem_20260728T133256Z/timing_summary.txt). Speech-board comparison: [status](STATUS.md), [speech peers](speech_peers.md), [fair board metrics](../host/speech_peers_fair_board.json).

**Critical engineering distinction:** the hardware design can execute the recorded smoke sequence without proving that static timing has closed. Negative WNS means the 100 MHz constraint was **not satisfied** by the saved WMEM implementation; the functional pass cannot supersede this result.

## 3. Verification environment and CI boundary

CI uses Icarus Verilog RTL simulations (PE, systolic array, glue, AXI-Lite) plus host NumPy references and a C++ HOST=1 parity build. GitHub Actions **does not** synthesize/place in Vivado, produce a bitstream, run the physical PYNQ target, download Speech Commands, or retrain models. See [CI workflow](../.github/workflows/ci.yml).

```sh
./scripts/ci_sim.sh
./scripts/ci_host.sh
# Requires PYNQ board and compiled bitstream:
bash host/board_bringup_5min.sh
```

These are reproduction commands, **not newly executed tests**. A future hardware report should preserve the bitstream/hash, XDC constraints, utilization/timing exports, software/driver revisions and full board run transcript.

## 4. Disposition and improvement priorities

**Functional bring-up recorded; performance constraints not universally met.** Preserve the conventional A+B path as measured default for this tiny workload. Primary unresolved items: close 100 MHz timing for WMEM; reduce DMA/weight-load overhead; prove numerical correctness across larger tensor/shape suites; and implement the **not-yet-complete live-audio board application**. The host Google Speech Commands accuracy is not board speech-system accuracy.

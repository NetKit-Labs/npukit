# Transformer glue on NpuKit

The systolic array is an **int8 GEMM tile** (int8×int8 → **int32** C). A tiny
transformer also needs epilogue ops on every block. Those live in
**`rtl/npukit_glue.sv`**, not in `pe.sv` / `systolic_array.sv`.

Bring-up history: [`glue_bringup.md`](glue_bringup.md).  
Full FPGA vs CPU map (tables + dataflow): [`transformer_split.md`](transformer_split.md).

## Split of work (summary)

| On FPGA (this glue + GEMM) | On A9 (host glue) |
|----------------------------|-------------------|
| Tile GEMM (Q/K/V, \(QK^{T}\), \(AV\), FFN) | RoPE |
| Residual add | Attention mask build |
| Softmax (rows ≤ 16) | Reshape / transpose / pack |
| RMSNorm | KV cache layout |
| GELU | Vectors longer than `MAX_LEN` |

See [`transformer_split.md`](transformer_split.md) for the per-step table.

## Fixed-point / width contract

**Host and AXI-Lite are 32-bit only** (Cortex-A9 / future 32-bit MCU driver).
Glue banks are int32; there is no int64 MMIO.

| Bank / field | Format |
|--------------|--------|
| `GLUE_X`, `GLUE_Y`, `GLUE_OUT` (most ops) | int32 **Q12** (`real = val / 4096`) |
| `GLUE_GAMMA` | int32 **Q12** (`1.0 → 4096`) |
| Softmax `GLUE_OUT` | int32 **Q16** probs (sum ≈ 65536) |
| `GLUE_PARAM` | RMSNorm ε in Q12 |

Must match `host/npukit_transformer.py`.

GEMM stays **int8 in / int32 out**. Glue is wider on purpose (post-GEMM
activations): dequant → Q12 glue → requant to int8 for the next tile.

Inside the FPGA, multiply products are often held in **64-bit wires for one or
two cycles** then shifted back to int32 (same reason PE MAC is wider than
int8). That is fabric arithmetic, not a 64-bit CPU. Details in
[`glue_bringup.md`](glue_bringup.md#bit-widths-int8-fabric-vs-int32-host-vs-64-bit-rtl).

### RMSNorm note

Hardware truncates to Q12 between the two multiplies
\((x\cdot\mathrm{inv})\gg 12\) then \((\cdot\gamma)\gg 12\). The Python
`ref_rmsnorm` mirrors that.

### Softmax / GELU LUTs

256-entry distributed ROMs; index is `(x + bias) >> 7` saturated to **255**
(so the max Softmax logit does not wrap `256 → 0`).

## Register map (additions in v0x300)

| Offset | Name | |
|--------|------|--|
| `0x004` | VERSION | `0x00000300` |
| `0x008` | STATUS | `[4]=glue_done`, `[0]=busy` (GEMM\|glue) |
| `0x014` | FEATURES | `[0]=GEMM`, `[1]=GLUE` |
| `0x018` | GLUE_CTRL | `[0]=start`, `[7:4]=opcode` |
| `0x01C` | GLUE_LEN | `1..16` |
| `0x020` | GLUE_PARAM | e.g. RMSNorm ε |
| `0x024` | GLUE_COUNT | increments each completed glue op |
| `0x500` | GLUE_X | 16 × int32 |
| `0x600` | GLUE_Y | 16 × int32 |
| `0x700` | GLUE_OUT | 16 × int32 |
| `0x800` | GLUE_GAMMA | 16 × int32 |

### Opcodes

| Code | Op | Result |
|------|----|--------|
| `0x1` | RESIDUAL | `OUT = X + Y` |
| `0x2` | GELU | LUT approx on \([-4,4]\) Q12 |
| `0x3` | RMSNORM | `OUT = (X / rms(X)) * GAMMA` (power-of-two `LEN` for mean) |
| `0x4` | SOFTMAX | row softmax → Q16 |

## Host sequence (one glue op)

1. Write `GLUE_LEN`, `GLUE_PARAM` if needed  
2. Fill `GLUE_X` (and `Y` / `GAMMA`)  
3. `GLUE_CTRL = (opcode<<4) | 1`  
4. Wait for **`GLUE_COUNT`** to advance (preferred; `STATUS[4]` is sticky)  
5. Read `GLUE_OUT`

Board-verified on PYNQ-Z2 @ 100 MHz (WNS ≥ 0): residual / GELU / RMSNorm /
Softmax + GEMM tile via `python3 host/npukit_transformer.py <bit>`.

## Rebuild / sim

After RTL changes, force a clean PL resynth (see bring-up doc), then:

```bash
../scripts/build_bitstream.sh npukit

iverilog -g2012 -o sim/npukit_glue_tb.vvp rtl/npukit_glue.sv sim/npukit_glue_tb.sv
vvp sim/npukit_glue_tb.vvp
```

## Tiny transformer dataflow (conceptual)

```
x ──GEMM(Wq/Wk/Wv)──► Q,K,V
Q,K ──(RoPE on A9)──► Q',K'
Q',K' ──GEMM──► scores ──Softmax(glue)──► P
P,V ──GEMM──► attn ──Residual+RMSNorm(glue)──► h
h ──GEMM──► ──GELU(glue)──► ──GEMM──► ──Residual+RMSNorm──► y
```

# Transformer work split: FPGA vs CPU

What runs on the **PYNQ-Z2 PL** (NpuKit fabric) versus the **Zynq A9 / host CPU**
for a tiny transformer built on this kit.

Related: [`transformer_glue.md`](transformer_glue.md) (opcodes / fixed-point),
[`glue_bringup.md`](glue_bringup.md) (how glue closed timing).

---

## One-page map

```text
                    ┌─────────────────────────────────────────┐
                    │         CPU (A9 / future 32-bit MCU)      │
                    │  orchestration, buffers, scales, RoPE,    │
                    │  masks, reshape/pack, quant / dequant     │
                    └───────────────┬─────────────────────────┘
                                    │ AXI-Lite (control, glue banks)
                                    │ AXI DMA  (A/B/C int8/int32 tiles)
                    ┌───────────────▼─────────────────────────┐
                    │              FPGA (PL)                    │
                    │  8×8 int8 GEMM  +  glue (len ≤ 16)       │
                    │  residual · GELU · RMSNorm · Softmax      │
                    └─────────────────────────────────────────┘
```

---

## Per-op table

| Step in a transformer block | Where | Notes |
|-----------------------------|-------|--------|
| Token / activation buffers | **CPU** | Own memory, tiling plan |
| Quantize float/Q12 → int8 for GEMM | **CPU** | Scales chosen on host |
| Dequantize int32 C → Q12 (or float) | **CPU** | After each GEMM tile accumulate |
| \(W_Q, W_K, W_V, W_O\), FFN matmuls | **FPGA GEMM** | 8×8 int8 tiles via DMA/MMIO |
| RoPE on Q/K | **CPU** | “Weird” geometry; not in glue |
| Attention mask build / apply | **CPU** | Causal / padding masks |
| Reshape, transpose, pack/unpack | **CPU** | Layout for tiles and heads |
| \(QK^{T}\) scores (tiled GEMM) | **FPGA GEMM** | Then often dequant on CPU before Softmax |
| Softmax over attention row | **FPGA glue** | Opcode `0x4`, len ≤ 16, OUT Q16 |
| \(PV\) (attn · V) | **FPGA GEMM** | |
| Residual add | **FPGA glue** | Opcode `0x1`, Q12 |
| RMSNorm | **FPGA glue** | Opcode `0x3`, Q12; power-of-two len |
| FFN Linear | **FPGA GEMM** | |
| GELU | **FPGA glue** | Opcode `0x2`, Q12 LUT |
| FFN Linear out + residual + RMSNorm | **FPGA GEMM + glue** | Same as above |
| Softmax / GELU / RMSNorm for **len > 16** | **CPU** | Glue `MAX_LEN=16` today |
| Final logits / argmax / sampling | **CPU** | |
| KV cache addressing / policy | **CPU** | |

---

## Dataflow (one block, conceptual)

```text
x (CPU buffer)
  │
  ├─ quant ──► GEMM Wq/Wk/Wv (FPGA) ──► dequant (CPU)
  │                 │
  │                 ▼
  │            RoPE (CPU) on Q,K
  │                 │
  │                 ▼
  │            GEMM QKᵀ (FPGA) ──► scores
  │                 │
  │            mask (CPU) optional
  │                 │
  │            Softmax (FPGA glue) ──► P
  │                 │
  │            GEMM P·V (FPGA) ──► attn
  │                 │
  │            GEMM Wo (FPGA)
  │                 │
  ├────────── Residual (FPGA glue) ◄── x
  │                 │
  │            RMSNorm (FPGA glue)
  │                 │
  │            GEMM W1 (FPGA) ──► GELU (FPGA glue) ──► GEMM W2 (FPGA)
  │                 │
  └────────── Residual + RMSNorm (FPGA glue) ──► y
```

Anything drawn on a **CPU** arrow above stays off the systolic array and off
`npukit_glue` (unless later moved).

---

## Why this split

| Prefer FPGA | Prefer CPU |
|-------------|------------|
| Regular dense matmul tiles | Irregular index math (RoPE, gather) |
| Short vector epilogues reused every block | Control flow, batching, I/O |
| Fixed-point ops that fit `MAX_LEN` | Long sequences, masks, cache policy |
| Bound latency for hot loops | Flexibility while the model still moves |

Glue is intentionally **not** a full transformer accelerator — it is the
common epilogue next to the int8 GEMM so the A9 is not soft-maxing / GELU-ing
every tiny row in Python if we do not want it to.

---

## Host entry points

| Role | File |
|------|------|
| GEMM tiles | `host/npukit_matmul.py` |
| Glue ops + ref model | `host/npukit_transformer.py` |
| Interactive GEMM | `host/npukit_matmul.ipynb` |

CPU-only helpers already sketched in the transformer host: RoPE, attention
reference in float, quant/dequant helpers — not a full production runtime yet.

---

## Limits that force CPU fallback

- Glue vector length **> 16** → run that op on CPU (or tile the vector in
  software and issue multiple glue kicks).
- Heads / sequence dims that do not pack into 8×8 GEMM tiles → CPU tiling
  around GEMM (same as matmul host today).
- Any new op not in the opcode table (LayerNorm, SiLU, etc.) → CPU until RTL
  grows.

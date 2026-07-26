# Transformer glue bring-up notes

Record of how `rtl/npukit_glue.sv` (VERSION `0x300`) got from “RTL sketched” to
**board PASS @ 100 MHz** on PYNQ-Z2, and why some datapaths use wide
intermediates even though the NPU is int8 and the host is a 32-bit A9.

Companion docs: [`transformer_glue.md`](transformer_glue.md) (contract / regmap),
[`tiling.md`](tiling.md) (GEMM tiling).

---

## Bit widths: int8 fabric vs int32 host vs “64-bit” RTL

### What the product contract is

| Layer | Width | Role |
|-------|-------|------|
| Systolic PE MAC | **int8 × int8 → int32** acc | Tile GEMM (same as before glue) |
| AXI-Lite / MMIO / Python driver | **32-bit** registers and words | A9 is a 32-bit Cortex-A9; host never needs int64 |
| Glue banks `X/Y/OUT/GAMMA` | **int32** Q12 (Softmax OUT **Q16**) | Fixed-point activations on the wire |

There is **no 64-bit MCU ABI** and no 64-bit AXI-Lite beat. The PS driver reads
and writes `uint32` / `int32` only (`host/npukit_transformer.py`,
`host/npukit_matmul.py`).

### Why RTL still has `[63:0]` wires

Those are **internal multiply/accumulate widenings inside the FPGA**, not a
host datatype:

1. **Product of two int32 values is up to 64 bits.**  
   Example: RMSNorm does \(x \cdot x\) then sums over the vector. Two Q12
   int32 values multiply to ~Q24 in a 64-bit product; summing \(N\) of them
   needs headroom. Same idea as PE MAC: int8×int8 needs more than 8 bits in
   the accumulator — we just chose int32 there because \(K\) tiles stay small.

2. **DSP48E1 is a hardware multiplier.** Vivado maps `a * b` to one or more
   DSP slices. Declaring `logic signed [63:0] mul64` is “keep the full product
   for one cycle, then `>>> Q12` back to int32.” It is not “run a 64-bit CPU.”

3. **The Softmax / inv-RMS divider really is a 64-bit machine in the fabric.**  
   That is what burned timing early on — not a misunderstanding of “MCU is
   32-bit.”

### The 64-bit divider (what you saw struggle)

We need Softmax probs in Q16:

\[
q_i = \frac{e_i \cdot 2^{16}}{\sum_j e_j}
\]

So the **mathematical dividend** is \((e_i \ll 16)\), which for \(e_i\) near
\(2^{16}\) already wants ~32+ bits, and the classic MSB-first restoring
divider is implemented as:

- `div_numer[63:0]` ← `{ dividend_32, 32'b0 }` (left-aligned)
- `div_rem[63:0]`, trial subtract each cycle
- 32 steps → **32-bit** `div_quot` written back to `OUT`

Same engine is reused for RMSNorm `inv = (4096*4096) / sqrt(...)`.

So yes: **64-bit shift-register divider state** in RTL. That is wider than the
PE’s int32 accumulator, and it is why early builds missed 100 MHz until the
divider was multi-cycle (and every combo `/` was removed from the hot path).

What it is *not*:

- Not an AXI-Lite 64-bit beat
- Not an A9 / MCU `uint64_t` ABI — host still only writes/reads **int32**
  banks and kicks the op; the wide math stays inside PL for ~32 cycles

Could we avoid 64-bit divider state? Yes, later options:

- Smaller fixed-point (e.g. Q8 probs) so dividend fits in 32 bits with a
  32-wide restoring div
- Reciprocal LUT + multiply instead of divide
- Keep Softmax on the A9 for tiny \(N\) (was the original PLAN default)

Bring-up chose “correct Q16 Softmax in fabric” first, then paid for it with
pipelining. Later cleanup: the divider is **two 32-bit registers** (`div_num` +
`div_rem`) instead of one 64-bit shift register — same math, tidier fabric.

### What we deliberately did *not* do

- Softmax / GELU / RMSNorm are **not** int8 on the glue path today. After GEMM
  you typically dequant to a wider activation (here Q12 int32) for norm /
  nonlinearity, then requant to int8 for the next GEMM tile. That matches
  common TinyML / edge-transformer practice.
- We truncated RMSNorm to **two pipelined int32×int32 multiplies** with a Q12
  result between them (instead of one combinational 64×32→96 path) so timing
  could close — numerical contract is documented in
  `ref_rmsnorm()` in the host.

### Could glue be narrower overall?

Yes: Q8/int16 activations, 32-wide divider, reciprocal LUT, or Softmax on the
CPU. The bring-up choice was **int32 Q12 banks + int32 MMIO** for a simple
32-bit driver, accepting **wide internal** mul/div engines in PL.

---

## What shipped (end state)

| Item | Value |
|------|--------|
| VERSION | `0x00000300` |
| FEATURES | bit0 GEMM, bit1 GLUE |
| Ops | residual, GELU, RMSNorm, Softmax |
| `MAX_LEN` | **16** (was 64; cut for mux timing). Glue `len==MAX_LEN` load fix is in the current bit; Softmax length 16 used by ViT. |
| Clock | PS FCLK0 **100 MHz**, WNS ≈ **+0.83 ns** (latest rebuild) |
| Board | residual / GELU / RMSNorm / Softmax + GEMM tile **PASS**; e2e T=8×D=8 **ALL E2E PASS**; ViT T=16×D=16×L=2 **ALL VIT PASS** |
| Host | `host/npukit_transformer.py`, `host/npukit_transformer_e2e.ipynb`, `host/npukit_vit_mnist.py` |
| Status | [`STATUS.md`](STATUS.md) |
| Sync | prefer `GLUE_COUNT` (`0x024`), not sticky `STATUS[4]` alone |

Approx placed util with glue (Z7020): ~**18–19% LUT**, ~**33% DSP** (GEMM 64
DSP + glue DSP muls) — still comfortable headroom.

---

## Iteration log (what broke, what we changed)

Rough chronological order of the bring-up loops.

### 1. First glue RTL + MMIO

- Added `npukit_glue.sv`, wired into `npukit_axil.sv`, VERSION `0x300`,
  banks at `0x500`–`0x8FF`, host driver + `docs/transformer_glue.md`.
- Sim TB for residual + softmax smoke.

### 2. Board / lab hazards (not RTL bugs)

- Wrong IP briefly (`.119` vs board `.215`).
- Board looked dead: **SD card slipped out** — only power LED; reseat fixed it.
- `/dev/fpga0` needs **sudo** + XRT/venv on PYNQ.
- Docker Vivado: WebTalk/`libudev` crash → stub `libudev.so.1`.
- Stale OOC synth: must delete `system_pl_logic_0_synth_1` (and impl) or RTL
  changes never make the bit.
- Host `mmio.read` as signed int32 overflowed IDs — use uint32 view.

### 3. Functional sticky-done / sync

- `glue_done` sticky made “wait for done” race on short ops.
- Added **`GLUE_COUNT`** and host wait on count advance (+ short sleep
  fallback).

### 4. Timing: the long fight (−130 ns → +1 ns)

Critical theme: **do not put dividers, 256-entry LUT address math, dual DSP
cascades, or huge register-file mux + ALU in one 10 ns cycle.**

| Step | Change | Approx WNS |
|------|--------|------------|
| Early glue | Combo Softmax divide, fat muxing | ~−130 ns |
| Multi-cycle restoring divider + isqrt | Softmax / inv-rms not combo | still bad (~−16 ns) |
| `MAX_LEN` 64→**16** | Smaller `x_mem` mux trees | still ~−15 ns |
| Remove combo `/` in LUT index (`*255/32768`) | use `>>7` | improved |
| Pipeline GELU / Softmax (load → index → LUT) | break idx→ROM→sum_exp | ~−7 ns |
| Extra load stages (register `x_r` before ALU) | | ~−6 ns |
| Pipeline isqrt (mid / sq / update) | DSP×DSP compare was one cycle | ~−5 ns |
| Pipeline residual + RMS apply | | ~−3 ns |
| Narrow RMS to int32×int32 + Q12 truncate between muls | kill 64×32 DSP cascade into `out_mem` | ~−2 ns |
| Split `x*x` and `acc +=` | | ~−0.9 ns |
| Register mul operands (`mul_a`/`mul_b`) + saturate LUT index | | **met (~+0.8…+1.1 ns)** |

Softmax functional bug found **after** timing closed:

- At the max logit, \(t=0\): `(0 + 32768) >> 7 == 256`, truncated to 8 bits →
  **0**, so the winner read `exp_lut[0]` (≈e⁻⁸) instead of the top bin.
- Fix: saturate index to 255. Host `ref_softmax` updated to the same LUT
  addressing. Board: **softmax PASS**.

GEMM “FAIL” in an early suite print was a **test harness bug** (`ok` mixed
glue+gemm); GEMM itself was fine.

### 5. Clean rebuild discipline

Always force PL resynth after glue RTL edits:

```bash
rm -rf vivado/npukit.runs/system_pl_logic_0_synth_1 \
       vivado/npukit.runs/synth_1 \
       vivado/npukit.runs/impl_1 \
       output/npukit.bit
../scripts/build_bitstream.sh npukit   # or in-repo scripts/
```

Confirm: timing summary says constraints met; then SCP bit + host to the board
and run with sudo + XRT.

---

## Design rules we kept (for the next person)

1. **Host ABI = 32-bit.** Never require int64 on the A9 / future MCU driver.
2. **Widen only inside mul/acc/div engines**, then truncate back to int32 Qx.
3. **One heavy op per cycle** at 100 MHz: mul *or* add *or* LUT, not mux+div+ROM.
4. **`MAX_LEN=16`** until BRAMs + registered read ports replace register files.
5. Prefer **`GLUE_COUNT`** for completion; treat sticky DONE as secondary.
6. Delete OOC synth runs when changing `npukit_glue` / `npukit_axil`.

---

## Open follow-ups

- End-to-end tiny transformer (quantize → GEMM tiles → glue → requant).
- Optional: BRAM banks + larger `MAX_LEN` without blowing timing.
- Optional: narrower activation format if int8 epilogue proves accurate enough.
- Drop local `host/_glue_diag.py` if still lying around untracked.

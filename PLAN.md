# FPGA NPU kickoff (8×8 int8 systolic)

## Goal

Start a **small NPU** on PYNQ-Z2: an **8×8 int8 systolic array**, built and verified the same way as blinker (SV modules + Icarus TBs on the host; Vivado `.bit`; load via Overlay / `Bitstream.download()`).

## Assumed MVP (default)

1. **`pe` module** — int8 × int8 → accumulate (int32), with clear / enable; DSP48-preferred MAC
2. **`systolic_array`** — 8×8 PE grid, **output-stationary** (A west→east, B north→south)
3. **Testbenches** for `pe`, the array, and the AXI-Lite host path
4. **Project skeleton** with AXI-Lite + AXI DMA (PS `M_AXI_GP0` + `S_AXI_HP0`)
5. **Host** — tiled matmul via `host/npukit_matmul.py` / `.ipynb` (NumPy check)

**Next / later:** ping-pong / larger on-chip tiles, real NN models, quantization toolchain.

## Project layout

- `rtl/pe.sv`, `rtl/systolic_array.sv`, `rtl/npukit_axil.sv`, `rtl/npukit_top.sv`, `rtl/npukit_pl.v`
- `sim/pe_tb.sv`, `sim/systolic_array_tb.sv`, `sim/npukit_axil_tb.sv`
- `host/npukit_matmul.py`, `host/npukit_matmul.ipynb`
- `scripts/create_project.tcl`, `scripts/build_bitstream.tcl`, `scripts/pynq_bitstream.tcl`

```mermaid
flowchart LR
  subgraph done [Done]
    PE[pe.sv]
    SA[systolic_array.sv]
    AXIL[npukit_axil BRAM AXIS]
    DMA[AXI DMA HP0]
    HOST[host tiled matmul]
    PE --> SA --> AXIL --> DMA --> HOST
  end
  subgraph later [Later]
    PP[ping-pong / larger tiles]
    Q[quant / models]
    HOST --> PP --> Q
  end
```

## Keep from blinker lessons

- SystemVerilog for design; `.v` wrapper only for BD
- Load with Overlay (DMA for A/B/C tiles) plus AXI-Lite MMIO for CTRL/STATUS (base `0x43C00000`); keep both paths
- Host: `npukit_matmul.py` is the source of truth; `.ipynb` is the interactive wrapper (classic 8×8 + tiled suites)
- Keep PS + board preset in the Vivado BD
- Module TBs before board bring-up

## Status

MVP through DMA + host tiling is **done** (sim PASS, board PASS with Overlay/`axi_dma_0`). Rebuild with `../scripts/build_bitstream.sh npukit` (or in-repo `scripts/`) after RTL/BD changes.

### Z7020 size note (8×8 + DMA measured; 16×16 not built)

Placed util for the current DMA bitstream: **~13% LUT, ~9% FF, 64 DSP (29%), 2 BRAM tiles**.

Those **2 BRAM tiles** are Vivado’s packing of the three small logical memories (`a_mem`, `b_mem`, `c_mem` — one 8×8 tile each). They are **not** ping-pong buffers and **not** two concurrent matrix slots. Larger MxKxN products are tiled in software over the single A/B/C tile buffers.

Scaling the PE grid \(8→16\) is still \(4×\) array area (~256 DSPs needed for 1:1 MAC mapping; the chip has 220), so prefer **stay 8×8 + tile + DMA**, optionally add ping-pong later to overlap load and compute.

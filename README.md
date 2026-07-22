# NpuKit

FPGA NPU kit for the [PYNQ-Z2](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html) (Xilinx Zynq-7020): a small, teachable **8×8 int8 output-stationary systolic array**, verified in simulation first, then loaded as a bitstream onto the board.

Part of [NetKit Labs](https://github.com/NetKit-Labs) — companion direction to [netkit](https://github.com/NetKit-Labs/netkit) (embedded NN inference on MCU/MPU), focused here on **custom FPGA acceleration**.

## Where we are (MVP milestone 1)

| Done | Item |
|:---:|---|
| yes | `pe` — int8 × int8 → int32 MAC, with clear / enable and A/B forward |
| yes | `systolic_array` — 8×8 PE grid (A west→east, B north→south, C stationary) |
| yes | Icarus testbenches: `pe_tb` and `systolic_array_tb` (known matmul) |
| yes | PYNQ-Z2 project skeleton: PS board preset + `npukit_pl` BD wrapper |
| yes | Board face: LD0 ~1 Hz heartbeat, BTN0 = reset hold |
| later | AXI-Lite host registers / DMA |
| later | Real model bring-up / quantization flow |

**First win achieved:** correct **8×8 int8 matrix multiply in simulation**.

## Hierarchy

```
npukit_pl.v              Vivado BD Verilog wrapper
└── npukit_top.sv        LED heartbeat + array instance
    └── systolic_array.sv
        └── pe.sv × 64
```

## Simulate (host)

```bash
cd npukit

# PE unit tests
iverilog -g2012 -o sim/pe_tb.vvp rtl/pe.sv sim/pe_tb.sv
vvp sim/pe_tb.vvp

# 8×8 matmul check
iverilog -g2012 -o sim/array_tb.vvp rtl/pe.sv rtl/systolic_array.sv sim/systolic_array_tb.sv
vvp sim/array_tb.vvp
```

Expect `pe_tb: ALL PASS` and `systolic_array_tb: ALL PASS (8x8 int8 matmul)`.

`c_out` is flat row-major: index = `row * N + col`.

## Build / load (board)

Requires Vivado (and shared helpers in sibling `../scripts/pynq_bitstream.tcl` when using this monorepo layout):

```bash
# Create Vivado project (once)
vivado -mode batch -source scripts/create_project.tcl

# Build bitstream
vivado -mode batch -source scripts/build_bitstream.tcl
# → output/npukit.bit

# On the PYNQ (after scp):
from pynq import Bitstream
Bitstream("/home/xilinx/jupyter_notebooks/npukit.bit").download()
```

No Python Overlay / `.hwh` required for this MVP — use `Bitstream.download()`.

## Layout

```
npukit/
  rtl/           pe.sv, systolic_array.sv, npukit_top.sv, npukit_pl.v
  sim/           pe_tb.sv, systolic_array_tb.sv
  constraints/   pynq_z2.xdc
  scripts/       create_project.tcl, build_bitstream.tcl
  output/        bitstream artifacts (gitignored)
```

## Design notes

- **Output-stationary:** each PE keeps its accumulator; activations and weights stream past.
- **SystemVerilog** for design RTL; thin **Verilog** `npukit_pl` wrapper for Vivado Block Design module-reference.
- Zynq **PS** stays in the BD with the PYNQ-Z2 board preset so Ethernet/DDR remain correct under Linux.

## License

MIT (see `LICENSE`).

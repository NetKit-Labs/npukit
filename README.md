# NpuKit

FPGA NPU kit for the [PYNQ-Z2](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html) (Xilinx Zynq-7020): an **8×8 int8 output-stationary systolic array** with AXI4-Lite control and AXI DMA data movement from the Zynq PS.

Part of [NetKit Labs](https://github.com/NetKit-Labs) — companion direction to [netkit](https://github.com/NetKit-Labs/netkit) (embedded NN inference on MCU/MPU), focused here on **custom FPGA acceleration**.

## Where we are

| Done | Item |
|:---:|---|
| yes | `pe` — int8 × int8 → int32 MAC, with clear / enable and A/B forward |
| yes | `systolic_array` — 8×8 PE grid (A west→east, B north→south, C stationary) |
| yes | Icarus testbenches: `pe_tb`, `systolic_array_tb`, `npukit_axil_tb` |
| yes | BRAM-backed A/B/C tile storage, DSP-preferred PE MACs |
| yes | AXI4-Lite control + AXI-Stream tile ports; Vivado BD adds AXI DMA via PS HP0 |
| yes | Board face: LD0 heartbeat, LD1 busy, LD2 done; BTN0 = reset hold |
| yes | Python host on PYNQ: `npukit_matmul.py` + notebook wrapper (classic 8×8 + tiled suites) |
| yes | Board bring-up verified on PYNQ-Z2 (DMA + AXI-Lite control, **12/12 PASS**) |
| yes | Keep both paths: DMA for matrix tiles, MMIO/AXI-Lite for control (+ fallback) |
| yes | Saved notebook run: classic 8×8 + tiled 16×16 / 32×32 dumps in `npukit_matmul.ipynb` |
| yes | **Hardware MVP complete** — next work is mostly runtime / models |
| later | Quantization, tiny end-to-end NN on host+GEMM; optional HW polish |

## Hierarchy

```
npukit_pl.v                 BD wrapper (AXI4-Lite + AXIS + btn/led)
└── npukit_top.sv           heartbeat + reset combine
    └── npukit_axil.sv      regs, BRAM A/B/C, AXIS, sequencer
        └── systolic_array.sv
            └── pe.sv × 64     (DSP48E1 MAC)
```

Placed utilization (Zynq-7020, DMA bitstream): **~13% LUT, ~9% FF, 64 DSP (29%), 2 BRAM tiles**  
(~87% LUT, ~91% FF, ~71% DSP, ~99% BRAM still free; DSPs are the main limit if the array grows).

## AXI-Lite register map

Base address (BD default target): **`0x43C0_0000`**

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x000` | ID | R | `0x4E50554B` (`NPUK`) |
| `0x004` | VERSION | R | `0x00000200` |
| `0x008` | STATUS | R | `[0]` busy, `[1]` done, `[2]` AXIS RX done, `[3]` AXIS TX done |
| `0x00C` | CTRL | W | `[0]` start, `[1]` clear, `[2]` arm C AXIS TX (all write-1 pulses) |
| `0x010` | N | R | `8` |
| `0x100`–`0x13F` | A | R/W | 64× int8 packed as 16× `uint32` (LE), row-major |
| `0x200`–`0x23F` | B | R/W | same packing |
| `0x400`–`0x4FF` | C | R | 64× int32, row-major |

AXIS packets are 16 packed `uint32` A words then 16 B words, with `TLAST` on word 32; C returns 64 `int32` words after `CTRL.tx_arm`. For a K-tiled result, issue `CTRL=3` for the first tile and `CTRL=1` thereafter; `start` alone preserves the PE accumulators.

## Simulate (host)

```bash
cd npukit

iverilog -g2012 -o sim/pe_tb.vvp rtl/pe.sv sim/pe_tb.sv && vvp sim/pe_tb.vvp

iverilog -g2012 -o sim/array_tb.vvp rtl/pe.sv rtl/systolic_array.sv sim/systolic_array_tb.sv
vvp sim/array_tb.vvp

iverilog -g2012 -o sim/npukit_axil_tb.vvp \
  rtl/pe.sv rtl/systolic_array.sv rtl/npukit_axil.sv sim/npukit_axil_tb.sv
vvp sim/npukit_axil_tb.vvp
```

Expect `ALL PASS` from each TB.

## Build / load (board)

```bash
# From fpga/ monorepo (sources Vivado if needed):
../scripts/build_bitstream.sh npukit
# → output/npukit.bit and output/npukit.hwh
```

Or manually:

```bash
vivado -mode batch -source scripts/create_project.tcl   # once / after RTL port changes
vivado -mode batch -source scripts/build_bitstream.tcl
```

Copy to the PYNQ and run the host check:

```bash
scp output/npukit.bit output/npukit.hwh host/npukit_matmul.py host/npukit_matmul.ipynb \
  xilinx@<pynq>:jupyter_notebooks/
```

**CLI** (use the PYNQ venv; `/dev/fpga0` may require `sudo`):

```bash
source /etc/profile.d/xrt_setup.sh
source /usr/local/share/pynq-venv/bin/activate
python /home/xilinx/jupyter_notebooks/npukit_matmul.py \
  /home/xilinx/jupyter_notebooks/npukit.bit 32 32 32
```

**Jupyter:** open `npukit_matmul.ipynb` and run all cells. The notebook is an interactive wrapper around `npukit_matmul.py` (classic 8×8 suite + tiled MxKxN). Matrix data is always built in Python (e.g. `classic_8x8_cases()` / `tiled_cases()`), never loaded from the `.bit`. A checked-in board run on PYNQ-Z2 (DMA transport) shows **12/12 PASS**, including tiled **16×16** and **32×32×32**, with verbose A/B/C dumps left in the cell outputs.

**PS ↔ PL paths (keep both):**

| Path | Protocol / IP | Use |
|------|----------------|-----|
| MMIO | AXI4-Lite @ `0x43C0_0000` | CTRL / STATUS / ID; also A/B/C fallback if DMA unavailable |
| DMA | `axi_dma_0` @ `0x4040_0000` → AXIS | Bulk A/B tile in, C tile out |

With the matching `.hwh`, the host prefers Overlay + DMA for matrices and still uses AXI-Lite for control. Without `.hwh`, it falls back to AXI-Lite MMIO for everything. Keep `.bit` and `.hwh` side by side. The PL config is lost on power cycle; re-run the host to reload the `.bit`.

## Tiling math

How host tiling and the inner dimension $K$ work (with a 16×16 worked example): **[`docs/tiling.md`](docs/tiling.md)**.

The PYNQ host/`ipynb` print **A**, **B**, **C_npu**, **C_ref**, and a tiling plan for every case (pass `--quiet` on the CLI to suppress). After running the notebook, save it so those dumps stay in the file.

## Visualize

```bash
python3 viz/systolic_anim.py          # Pillow GIF → viz/out/systolic_8x8.gif
# or Manim (see viz/render_linkedin.sh)
```

## Layout

```
npukit/
  rtl/           pe, systolic_array, npukit_axil, npukit_top, npukit_pl
  sim/           pe_tb, systolic_array_tb, npukit_axil_tb
  host/          npukit_matmul.py, npukit_matmul.ipynb
  docs/          tiling.md (math + worked example)
  viz/           animation scripts
  constraints/   pynq_z2.xdc (btn/led; clock from PS FCLK0)
  scripts/       create_project.tcl, build_bitstream.tcl, pynq_bitstream.tcl
  output/        bitstream artifacts (gitignored)
```

## Design notes

- **Output-stationary:** each PE keeps its accumulator; A/B stream past.
- PL fabric clock is **PS FCLK0 (100 MHz)**; AXI-Lite on **M_AXI_GP0**; DMA data on **S_AXI_HP0**.
- Host software runs on the **Zynq A9 / PYNQ Linux** (not on a laptop). It tiles MxKxN as 8×8 blocks and accumulates over K in the PE registers.
- **MMIO ≠ AXI-Lite:** AXI-Lite is the bus protocol; MMIO is the CPU mapping of those registers. We keep MMIO (control + fallback) and DMA (matrix payloads) together.
- **A/B/C memories:** three logical tile buffers (`a_mem` / `b_mem` / `c_mem`) — one 8×8 A tile, one 8×8 B tile, one 8×8 C result. Vivado may report **2 BRAM tiles** for packing; that is a utilization detail, not ping-pong or two matrix slots.
- **NN coverage:** an int8 GEMM is enough for MLP / FC and 1×1 conv; standard conv can use `im2col`→GEMM later. No dedicated 3×3 / depthwise engine for now.
- **Host vs PL:** bias, ReLU, pooling, reshape, quant/scales, softmax/argmax, and orchestration stay on the **runtime (A9)** first. Optional later PL: fused bias+ReLU, ping-pong buffers — only if profiling warrants it.
- **Sim:** `npukit_axil_tb` covers AXI-Lite (+ accumulate). Do not Icarus-sim the Xilinx DMA IP; optional later: AXIS stimulus in that TB.
- BD helper: `scripts/pynq_bitstream.tcl` (`use_axi_lite=1`, `use_axi_dma=1`).

## License

MIT (see `LICENSE`).

# NpuKit C++ driver (PYNQ-Z2)

Userspace driver with **no Python in the GEMM kick loop**:

- AXI-Lite NPU @ `0x43C00000`
- AXI DMA @ `0x40400000` with **pooled CMA** TX/RX (two TX BOs for A ping-pong)
  - **XRT/zocl BO** on modern PYNQ 3.x (no `/dev/xlnk`)
  - legacy `libcma` + `/dev/xlnk` when present
  - MMIO fallback if neither works
- **Layer-resident weights** in bitstream `VERSION ≥ 0x302` (`FEAT_WMEM`): load `W` once, stream A-only. Opt in: `NPUKIT_WMEM=1` (default stays A∥B — faster on tiny-ViT). Tile WS+PP: `NPUKIT_WS_PP=1` — see [`docs/weight_stationary.md`](../../docs/weight_stationary.md)
- CPU DS-stem + float Softmax/RMSNorm/GELU helpers
- Fabric glue MMIO for residual / Softmax / RMSNorm (`len ≤ 16`); GELU stays float when `mlp > 16`


## Build on the board

```bash
# Once per boot: load the bitstream (Python is fine for this)
sudo bash -lc 'source /usr/local/share/pynq-venv/bin/activate; \
  python3 -c "from pynq import Overlay; Overlay(\"/home/xilinx/jupyter_notebooks/npukit.bit\")"'

cd host/cpp
python3 export_vit_bin.py --out vit_mnist.bin   # from vit_mnist_weights.npz
make -j2
sudo ./npukit_bench           # DMA / MMIO microbench
sudo ./npukit_vit --weights vit_mnist.bin
sudo ./npukit_vit --weights vit_mnist.bin --mmio
sudo ./npukit_vit --weights vit_mnist.bin --hybrid   # HW Softmax/RMSNorm + float GELU
```

Needs root for `/dev/mem` + CMA.

## Host parity check (no FPGA)

```bash
cd host/cpp
python3 export_vit_bin.py --n-samples 8
python3 export_dscnn_bin.py --n-samples 8
make clean && make HOST=1 npukit_vit npukit_dscnn
./npukit_vit --cpu --weights vit_mnist.bin
./npukit_dscnn --weights dscnn_mnist.bin
```

Compares C++ logits to Python reference logits embedded in the `.bin`.

## Fair peer latency (same C++ runtime)

On the board, both peers are C++ CPU/FPGA drivers (not Python):

```bash
sudo ./npukit_vit --weights vit_mnist.bin --mmio
./npukit_dscnn --weights dscnn_mnist.bin
```


## Layout

| Path | Role |
|------|------|
| `include/npukit/regs.hpp` | Register map |
| `include/npukit/device.hpp` + `src/device.cpp` | MMIO + DMA + tiled GEMM + glue |
| `include/npukit/cpu_ops.hpp` | Q12 + float Softmax/RMSNorm/GELU |
| `include/npukit/stem.hpp` + `src/stem.cpp` | Tiny DS-stem (C, no XNNPACK) |
| `include/npukit/vit.hpp` + `src/vit.cpp` | Weight load + ViT e2e forward |
| `export_vit_bin.py` | Pack `vit_mnist_weights.npz` → `vit_mnist.bin` |
| `src/bench_main.cpp` | GEMM/glue/stem microbench |
| `src/vit_main.cpp` | Full ViT classify + latency |

## Board result (MMIO fallback)

On the current PYNQ image `/dev/xlnk` is missing, so CMA DMA falls back to **MMIO tiles**.
Even so, the C kick loop dominates Python:

| Path | ~time |
|------|--------|
| Python ViT FPGA e2e (pool + fast stem) | **~614 ms/img** |
| Python DS-CNN int8 (numpy) | **~219 ms/img** |
| C++ ViT e2e FPGA GEMM + float glue (**XRT CMA**, `VERSION 0x302` A∥B) | **~9.8 ms/img** |
| Same with `NPUKIT_WMEM=1` | ~10.0 ms/img (correct, not faster here) |
| C++ DS-CNN int8 peer (A9) | **~9.3 ms/img** |
| C++ ViT e2e CPU GEMM on A9 | **~6.6 ms/img** |
| 320 × GEMM 8×8 kicks A∥B / WMEM | **~6.4 / ~8.8 ms** |

Fair peer race is C++ vs C++. `VERSION 0x302` advertises `WMEM`; host default stays A∥B. See [`results/wmem_20260728T133256Z/`](../../results/wmem_20260728T133256Z/).

Parity: ViT `max\|err\|=0` vs Python float-glue; DS-CNN float err ~1e-6 vs Python int8 numpy.

CMA is via **XRT/zocl** (`DMA=xrt-cma`).

## Notes

- This is **not** ping-pong yet — one TX + one RX CMA buffer when DMA is available.
- Default glue is A9 float Softmax/RMSNorm/GELU (matches Python deploy). `--hybrid` uses fabric Softmax/RMSNorm/residual when `len≤16`.
- Keep Python notebooks for bring-up; use `npukit_vit` for e2e latency.


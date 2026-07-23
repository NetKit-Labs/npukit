#!/usr/bin/env python3
"""PYNQ host for tiled int8 matmul using NpuKit's AXI DMA or AXI-Lite path.

The accelerator is an 8x8 tile.  Larger MxK times KxN products are reduced
over K in the array accumulators: the first K tile writes CTRL=CLEAR|START,
and later K tiles write CTRL=START only.

See docs/tiling.md for the math.

Usage:
  python3 npukit_matmul.py [/path/to/npukit.bit] [M K N]
  python3 npukit_matmul.py ... --quiet   # PASS/FAIL lines only
"""

from __future__ import annotations

import sys
import time
from typing import Protocol

import numpy as np

try:
    from pynq import Bitstream, MMIO, Overlay, allocate
except ImportError as exc:  # allow syntax checks off-board
    raise SystemExit("This script must run on PYNQ with pynq installed") from exc

N = 8
BASE = 0x43C0_0000
SPAN = 0x1000

REG_ID = 0x000
REG_VERSION = 0x004
REG_STATUS = 0x008
REG_CTRL = 0x00C
REG_N = 0x010
OFF_A = 0x100
OFF_B = 0x200
OFF_C = 0x400

CTRL_START = 0x1
CTRL_CLEAR = 0x2
CTRL_TX_ARM = 0x4
ID_MAGIC = 0x4E50554B

# Default: show matrices and tiling. Pass --quiet for summary lines only.
VERBOSE = True


def pack_i8(mat: np.ndarray) -> np.ndarray:
    """Row-major 8x8 int8 tile as 16 little-endian uint32 words."""
    flat = np.ascontiguousarray(mat, dtype=np.int8).reshape(-1)
    return np.frombuffer(flat.tobytes(), dtype="<u4").copy()


def unpack_i32(words: np.ndarray) -> np.ndarray:
    return np.asarray(words, dtype=np.uint32).view(np.int32).copy().reshape(N, N)


class TileTransport(Protocol):
    def load(self, a: np.ndarray, b: np.ndarray) -> None: ...
    def read(self) -> np.ndarray: ...


class MmioTransport:
    """Portable fallback for bitstreams without an HWH/DMA instance."""

    def __init__(self, mmio: MMIO) -> None:
        self.mmio = mmio

    def load(self, a: np.ndarray, b: np.ndarray) -> None:
        for offset, words in ((OFF_A, pack_i8(a)), (OFF_B, pack_i8(b))):
            for i, word in enumerate(words):
                self.mmio.write(offset + 4 * i, int(word))

    def read(self) -> np.ndarray:
        words = np.array(
            [self.mmio.read(OFF_C + 4 * i) & 0xFFFFFFFF for i in range(N * N)],
            dtype=np.uint32,
        )
        return unpack_i32(words)


class DmaTransport:
    """AXI DMA simple-mode transport; one S_AXIS packet per A/B tile pair."""

    def __init__(self, dma: object, mmio: MMIO) -> None:
        self.dma = dma
        self.mmio = mmio

    def load(self, a: np.ndarray, b: np.ndarray) -> None:
        packet = allocate(shape=(2 * N * N // 4,), dtype=np.uint32)
        packet[: N * N // 4] = pack_i8(a)
        packet[N * N // 4 :] = pack_i8(b)
        packet.flush()
        self.dma.sendchannel.transfer(packet)
        self.dma.sendchannel.wait()
        del packet

    def read(self) -> np.ndarray:
        packet = allocate(shape=(N * N,), dtype=np.uint32)
        self.dma.recvchannel.transfer(packet)
        self.mmio.write(REG_CTRL, CTRL_TX_ARM)
        self.dma.recvchannel.wait()
        packet.invalidate()
        result = unpack_i32(packet)
        del packet
        return result


def wait_done(mmio: MMIO) -> None:
    for _ in range(100_000):
        if mmio.read(REG_STATUS) & 0x2:
            return
        time.sleep(0.000_05)
    raise TimeoutError(f"NPU timeout status=0x{mmio.read(REG_STATUS):X}")


def describe_tiling(m: int, k: int, n: int, tile: int = N) -> str:
    """Human-readable tiling plan: spatial C tiles and K reductions."""
    if any(d % tile for d in (m, k, n)):
        raise ValueError("M, K, and N must be multiples of the tile size")
    mt, kt, nt = m // tile, k // tile, n // tile
    runs = mt * nt * kt
    lines = [
        f"  Shapes: A[{m}×{k}] @ B[{k}×{n}] → C[{m}×{n}]  (hardware tile T={tile})",
        f"  K is the inner dim: each C[i,j] = sum_{{p=0..{k-1}}} A[i,p]*B[p,j]",
        f"  Spatial C tiles: {mt}×{nt} = {mt * nt}   |   K-steps per C tile: {kt}",
        f"  Hardware runs: {mt}×{nt}×{kt} = {runs}",
        "  Example for C[0:T, 0:T]:",
    ]
    for step, k0 in enumerate(range(0, k, tile)):
        ctrl = "CLEAR|START" if step == 0 else "START     "
        lines.append(
            f"    k0={k0:3d}  CTRL={ctrl}  "
            f"A[0:{tile}, {k0}:{k0 + tile}] @ B[{k0}:{k0 + tile}, 0:{tile}]"
        )
    if mt > 1 or nt > 1:
        lines.append(
            f"  Then repeat for every (i0,j0) in "
            f"{{0,{tile},…,{m - tile}}} × {{0,{tile},…,{n - tile}}}."
        )
    return "\n".join(lines)


def format_matrix(name: str, mat: np.ndarray, max_rows: int = 32, max_cols: int = 32) -> str:
    """Pretty-print a matrix; truncate huge ones with a corner view."""
    arr = np.asarray(mat)
    rows, cols = arr.shape
    with np.printoptions(
        threshold=max_rows * max_cols + 1,
        linewidth=120,
        suppress=True,
        formatter={"int": lambda x: f"{int(x):4d}"},
    ):
        if rows <= max_rows and cols <= max_cols:
            body = str(arr)
        else:
            r, c = min(rows, 8), min(cols, 16)
            body = (
                f"(showing top-left {r}×{c} of {rows}×{cols})\n"
                f"{arr[:r, :c]}\n"
                f"  ... full shape {rows}×{cols}, dtype={arr.dtype}"
            )
    return f"{name} shape={arr.shape} dtype={arr.dtype}\n{body}"


def npu_matmul(
    mmio: MMIO, transport: TileTransport, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, float]:
    """Run an int8 MxK times KxN matrix product through 8x8 tiled hardware."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("A must be MxK and B must be KxN")
    m, k = a.shape
    _, n = b.shape
    if any(d % N for d in (m, k, n)):
        raise ValueError("M, K, and N must be multiples of 8")

    result = np.zeros((m, n), dtype=np.int32)
    t0 = time.perf_counter()
    for i0 in range(0, m, N):
        for j0 in range(0, n, N):
            first = True
            for k0 in range(0, k, N):
                transport.load(a[i0 : i0 + N, k0 : k0 + N], b[k0 : k0 + N, j0 : j0 + N])
                mmio.write(REG_CTRL, CTRL_CLEAR | CTRL_START if first else CTRL_START)
                wait_done(mmio)
                first = False
            result[i0 : i0 + N, j0 : j0 + N] = transport.read()
    return result, time.perf_counter() - t0


def open_device(bit_path: str) -> tuple[MMIO, TileTransport]:
    """Download bitstream; prefer Overlay/DMA when HWH is present."""
    try:
        overlay = Overlay(bit_path)
        dma = getattr(overlay, "axi_dma_0")
        mmio = MMIO(BASE, SPAN)
        print(f"Using AXI DMA transport ({bit_path})")
        return mmio, DmaTransport(dma, mmio)
    except (AttributeError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"DMA Overlay unavailable ({exc}); using AXI-Lite fallback")
        Bitstream(bit_path).download()
        mmio = MMIO(BASE, SPAN)
        return mmio, MmioTransport(mmio)


def run_case(
    mmio: MMIO,
    transport: TileTransport,
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    verbose: bool | None = None,
) -> bool:
    if verbose is None:
        verbose = VERBOSE
    t_cpu0 = time.perf_counter()
    ref = a.astype(np.int32) @ b.astype(np.int32)
    cpu_s = time.perf_counter() - t_cpu0
    got, npu_s = npu_matmul(mmio, transport, a, b)
    ok = np.array_equal(got, ref)

    banner = f"{'=' * 72}\nCASE: {name}  [{'PASS' if ok else 'FAIL'}]\n{'=' * 72}"
    print(banner)
    print(
        f"  timing: npu={npu_s * 1e3:8.3f} ms   cpu(NumPy)={cpu_s * 1e3:8.3f} ms   "
        f"transport={type(transport).__name__}"
    )
    print(describe_tiling(a.shape[0], a.shape[1], b.shape[1]))
    if verbose:
        print()
        print(format_matrix("A", a))
        print()
        print(format_matrix("B", b))
        print()
        print(format_matrix("C_npu (from FPGA)", got))
        print()
        print(format_matrix("C_ref (NumPy A@B)", ref))
        if ok:
            print("\n  C_npu == C_ref  ✓")
        else:
            diff = got.astype(np.int64) - ref.astype(np.int64)
            row, col = np.argwhere(got != ref)[0]
            print(f"\n  MISMATCH first at [{row},{col}]: "
                  f"npu={got[row, col]} ref={ref[row, col]} diff={diff[row, col]}")
            print(format_matrix("C_npu - C_ref", diff))
    elif not ok:
        row, col = np.argwhere(got != ref)[0]
        print(f"  mismatch [{row},{col}] got={got[row, col]} expected={ref[row, col]}")
    print()
    return ok


def classic_8x8_cases(rng: np.random.Generator | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Original single-tile demo / edge / random suite (all 8×8)."""
    if rng is None:
        rng = np.random.default_rng(0)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    a_demo = np.arange(1, N + 1, dtype=np.int8).reshape(N, 1) * np.ones((1, N), dtype=np.int8)
    b_demo = np.ones((N, N), dtype=np.int8)
    cases.append(("demo rows×ones", a_demo, b_demo))
    cases.append(("zeros", np.zeros((N, N), dtype=np.int8), np.zeros((N, N), dtype=np.int8)))
    cases.append(("identity", np.eye(N, dtype=np.int8), np.eye(N, dtype=np.int8)))
    cases.append(
        (
            "int8 extremes",
            np.full((N, N), 127, dtype=np.int8),
            np.full((N, N), -128, dtype=np.int8),
        )
    )
    cases.append(
        (
            "neg × pos",
            np.full((N, N), -3, dtype=np.int8),
            np.full((N, N), 5, dtype=np.int8),
        )
    )
    for i in range(5):
        a = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        b = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        cases.append((f"random[{i}]", a, b))
    return cases


def tiled_demo_16x16() -> tuple[str, np.ndarray, np.ndarray]:
    """Readable 16×16 case so spatial tiles and K-steps are easy to eyeball."""
    # A row r is filled with (r+1); B is identity → C == A (padded math still clear).
    a = np.arange(1, 17, dtype=np.int8).reshape(16, 1) * np.ones((1, 16), dtype=np.int8)
    b = np.eye(16, dtype=np.int8)
    return "tiled demo 16×16 (A rows× I)", a, b


def tiled_cases(
    m: int, k: int, n: int, rng: np.random.Generator | None = None
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Larger products that exercise host tiling + K accumulation."""
    if rng is None:
        rng = np.random.default_rng(1)
    if any(d % N for d in (m, k, n)):
        raise ValueError("M, K, and N must be multiples of 8")
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []
    if m >= 16 and k >= 16 and n >= 16:
        cases.append(tiled_demo_16x16())
    cases.append(
        (
            f"tiled random {m}x{k}x{n}",
            rng.integers(-8, 8, size=(m, k), dtype=np.int8),
            rng.integers(-8, 8, size=(k, n), dtype=np.int8),
        )
    )
    return cases


def run_suite(
    mmio: MMIO,
    transport: TileTransport,
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    verbose: bool | None = None,
) -> tuple[int, int]:
    passed = sum(
        1 for name, a, b in cases if run_case(mmio, transport, name, a, b, verbose=verbose)
    )
    return passed, len(cases)


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--quiet" in argv:
        VERBOSE = False
        argv.remove("--quiet")
    if "--verbose" in argv:
        VERBOSE = True
        argv.remove("--verbose")

    bit_path = argv[0] if len(argv) >= 1 else "/home/xilinx/jupyter_notebooks/npukit.bit"
    dims = tuple(map(int, argv[1:4])) if len(argv) >= 4 else (32, 32, 32)
    if len(dims) != 3:
        raise SystemExit("dimensions must be M K N")

    mmio, transport = open_device(bit_path)
    ident = mmio.read(REG_ID)
    if ident != ID_MAGIC:
        print(f"BAD ID: got 0x{ident:08X}, expected 0x{ID_MAGIC:08X}")
        return 1
    print(f"ID OK version=0x{mmio.read(REG_VERSION):08X} N={mmio.read(REG_N)}")
    print("NPU time includes DMA/MMIO + poll; CPU is NumPy matmul.")
    print("Verbose matrix dump on (use --quiet to suppress).\n")

    m, k, n = dims
    rng = np.random.default_rng(0)
    print("--- classic 8×8 ---")
    p0, t0 = run_suite(mmio, transport, classic_8x8_cases(rng))
    print(f"\n--- tiled (includes 16×16 demo + random {m}x{k}x{n}) ---")
    p1, t1 = run_suite(mmio, transport, tiled_cases(m, k, n, rng))
    passed, total = p0 + p1, t0 + t1
    print(f"\n{passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PYNQ host: run 8x8 int8 matmul on NpuKit via AXI-Lite and check vs NumPy.

Usage on the board (after scp of npukit.bit):
  python3 npukit_matmul.py [/path/to/npukit.bit]
"""

from __future__ import annotations

import struct
import sys
import time

import numpy as np

try:
    from pynq import Bitstream, MMIO
except ImportError as exc:  # allow syntax check off-board
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

ID_MAGIC = 0x4E50554B


def pack_i8(mat: np.ndarray) -> list[int]:
    """Row-major int8 matrix → list of little-endian uint32 words (4 bytes each)."""
    flat = np.asarray(mat, dtype=np.int8).reshape(-1)
    words = []
    for i in range(0, flat.size, 4):
        words.append(struct.unpack("<I", flat[i : i + 4].tobytes())[0])
    return words


def write_matrix(mmio: MMIO, offset: int, mat: np.ndarray) -> None:
    for i, w in enumerate(pack_i8(mat)):
        mmio.write(offset + 4 * i, int(w))


def read_c(mmio: MMIO) -> np.ndarray:
    out = np.zeros((N, N), dtype=np.int32)
    for i in range(N * N):
        # MMIO returns a Python int (unsigned view); force int32 on 32-bit ARM too
        word = mmio.read(OFF_C + 4 * i) & 0xFFFFFFFF
        out.flat[i] = np.array([word], dtype=np.uint32).view(np.int32)[0]
    return out


def npu_matmul(mmio: MMIO, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """Run one matmul on the NPU; return (C, wall_seconds including MMIO)."""
    write_matrix(mmio, OFF_A, A)
    write_matrix(mmio, OFF_B, B)
    mmio.write(REG_CTRL, 0x2)  # CLEAR
    mmio.write(REG_CTRL, 0x1)  # START

    t0 = time.perf_counter()
    for _ in range(100000):
        st = mmio.read(REG_STATUS)
        if st & 0x2:
            break
        time.sleep(0.00005)
    else:
        raise TimeoutError(f"NPU timeout status=0x{mmio.read(REG_STATUS):X}")
    C = read_c(mmio)
    dt = time.perf_counter() - t0
    return C, dt


def cpu_matmul(A: np.ndarray, B: np.ndarray, repeats: int = 200) -> tuple[np.ndarray, float]:
    """NumPy int32 matmul; return (C, average wall_seconds over repeats)."""
    A32 = A.astype(np.int32)
    B32 = B.astype(np.int32)
    # warmup
    _ = A32 @ B32
    t0 = time.perf_counter()
    C = None
    for _ in range(repeats):
        C = A32 @ B32
    dt = (time.perf_counter() - t0) / repeats
    return C, dt


def run_case(mmio: MMIO, name: str, A: np.ndarray, B: np.ndarray) -> bool:
    C_ref, cpu_s = cpu_matmul(A, B)
    C, npu_s = npu_matmul(mmio, A, B)
    ok = np.array_equal(C, C_ref)
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name:28s}  "
        f"npu={npu_s * 1e3:7.3f} ms  cpu={cpu_s * 1e3:7.3f} ms  "
        f"ratio={cpu_s / npu_s if npu_s > 0 else float('inf'):.2f}x"
    )
    if not ok:
        diff = np.where(C != C_ref)
        print(f"         first mismatch at {list(zip(diff[0][:3], diff[1][:3]))}")
        print(f"         got {C[diff][:3]} exp {C_ref[diff][:3]}")
    return ok


def main() -> int:
    bit_path = sys.argv[1] if len(sys.argv) > 1 else "/home/xilinx/jupyter_notebooks/npukit.bit"
    print(f"Downloading {bit_path}")
    Bitstream(bit_path).download()

    mmio = MMIO(BASE, SPAN)
    ident = mmio.read(REG_ID)
    if ident != ID_MAGIC:
        print(f"BAD ID at 0x{BASE:08X}: got 0x{ident:08X}, expected 0x{ID_MAGIC:08X}")
        print("Check that the bitstream mapped S_AXI to 0x43C00000.")
        return 1

    print(f"ID OK  version=0x{mmio.read(REG_VERSION):08X}  N={mmio.read(REG_N)}")
    print("Cases (npu time includes AXI poll+readback; cpu is NumPy avg):\n")

    rng = np.random.default_rng(0)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Original demo stimulus
    A_demo = np.arange(1, N + 1, dtype=np.int8).reshape(N, 1) * np.ones((1, N), dtype=np.int8)
    B_demo = np.ones((N, N), dtype=np.int8)
    cases.append(("demo rows×ones", A_demo, B_demo))

    # Edge / structured
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

    # Random suite
    for i in range(5):
        A = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        B = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        cases.append((f"random[{i}]", A, B))

    passed = 0
    for name, A, B in cases:
        if run_case(mmio, name, A, B):
            passed += 1

    total = len(cases)
    print(f"\n{passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
        out.flat[i] = np.int32(mmio.read(OFF_C + 4 * i))
    return out


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

    A = np.arange(1, N + 1, dtype=np.int8).reshape(N, 1) * np.ones((1, N), dtype=np.int8)
    B = np.ones((N, N), dtype=np.int8)
    C_ref = (A.astype(np.int32) @ B.astype(np.int32))

    write_matrix(mmio, OFF_A, A)
    write_matrix(mmio, OFF_B, B)

    mmio.write(REG_CTRL, 0x2)  # CLEAR
    mmio.write(REG_CTRL, 0x1)  # START

    t0 = time.time()
    for _ in range(10000):
        st = mmio.read(REG_STATUS)
        if st & 0x2:
            break
        time.sleep(0.0001)
    else:
        print(f"TIMEOUT status=0x{mmio.read(REG_STATUS):X}")
        return 1

    C = read_c(mmio)
    dt = time.time() - t0
    print(f"done in {dt*1e3:.2f} ms  status=0x{st:X}")

    if np.array_equal(C, C_ref):
        print("PASS: C matches NumPy int32 matmul")
        print(C)
        return 0

    print("FAIL: mismatch")
    print("got:\n", C)
    print("exp:\n", C_ref)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

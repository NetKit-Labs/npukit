# Tiled matmul on NpuKit

The hardware computes one **8×8** int8 product at a time:

\[
C_{\text{tile}} = A_{\text{tile}} \cdot B_{\text{tile}}
\quad\text{with}\quad
A_{\text{tile}}, B_{\text{tile}} \in \mathbb{Z}^{8\times 8},\quad
C_{\text{tile}} \in \mathbb{Z}^{32}
\]

(each of 64 PEs holds one `int32` accumulator). Larger products are done in **software tiling** on the Zynq A9.

## Shapes and what \(K\) means

For

\[
A \in \mathbb{Z}^{M\times K},\quad
B \in \mathbb{Z}^{K\times N},\quad
C = A B \in \mathbb{Z}^{M\times N}
\]

- \(M\) — rows of \(A\) / rows of \(C\)
- \(N\) — columns of \(B\) / columns of \(C\)
- \(K\) — **inner dimension**: columns of \(A\) and rows of \(B\).  
  Each output element is a dot product of length \(K\):

\[
C_{ij} = \sum_{p=0}^{K-1} A_{ip}\, B_{pj}
\]

On NpuKit, \(M\), \(K\), and \(N\) must be multiples of the hardware tile size \(T = 8\).

## Spatial tiles vs \(K\) reduction

Split the output into \(T\times T\) blocks. For the block whose top-left is \((i_0, j_0)\):

\[
C[i_0:i_0+T,\; j_0:j_0+T]
= \sum_{k_0 = 0,\,T,\,2T,\,\ldots}^{K-T}
  A[i_0:i_0+T,\; k_0:k_0+T]
  \cdot
  B[k_0:k_0+T,\; j_0:j_0+T]
\]

| Loop | Role |
|------|------|
| \(i_0\) over \(M\) | which **row-block** of \(C\) |
| \(j_0\) over \(N\) | which **column-block** of \(C\) |
| \(k_0\) over \(K\) | **partial products** that sum into that \(C\) block |

So:

- \(i_0, j_0\) place the result in the big \(C\)
- \(k_0\) walks the inner dimension — this is “what \(K\) means” for tiling:  
  **how many 8-wide chunks of the dot-product to accumulate**

Hardware CTRL:

1. First \(k_0\) for a given \((i_0,j_0)\): `CLEAR|START` (zero PE accumulators, then run)
2. Later \(k_0\): `START` only (keep accumulating)
3. After the last \(k_0\): read the 8×8 \(C\) tile (DMA or MMIO)

## Worked example: \(16\times 16\times 16\) with \(T=8\)

\[
M=K=N=16
\quad\Rightarrow\quad
\frac{M}{T}=\frac{N}{T}=2,\quad
\frac{K}{T}=2
\]

Output has **2×2** spatial tiles. Each needs **2** \(K\)-steps.

```
A (16×16)                 B (16×16)
┌────────┬────────┐       ┌────────┬────────┐
│ A00    │ A01    │       │ B00    │ B01    │
│ 8×8    │ 8×8    │       │ 8×8    │ 8×8    │
├────────┼────────┤       ├────────┼────────┤
│ A10    │ A11    │       │ B10    │ B11    │
└────────┴────────┘       └────────┴────────┘

C (16×16)
┌─────────────────┬─────────────────┐
│ C00 = A00·B00   │ C01 = A00·B01   │
│     + A01·B10   │     + A01·B11   │
├─────────────────┼─────────────────┤
│ C10 = A10·B00   │ C11 = A10·B01   │
│     + A11·B10   │     + A11·B11   │
└─────────────────┴─────────────────┘
```

Host sequence for **C00** (top-left 8×8 of \(C\)):

| Step | CTRL | Load A tile | Load B tile | Accumulators |
|------|------|-------------|-------------|--------------|
| 1 | CLEAR\|START | `A[0:8, 0:8]` = A00 | `B[0:8, 0:8]` = B00 | ← A00·B00 |
| 2 | START | `A[0:8, 8:16]` = A01 | `B[8:16, 0:8]` = B10 | ← previous + A01·B10 |
| 3 | (read C) | — | — | write into `C[0:8, 0:8]` |

Then the same pattern for C01, C10, C11 (four output tiles × two \(K\) steps = **8** hardware runs).

For **32×32×32**: \(4\times 4 = 16\) output tiles × \(4\) \(K\) steps = **64** hardware runs.

## Tiny numeric check (single tile, no \(K\) loop)

Demo case `rows×ones` (also in the host suite):

\[
A =
\begin{bmatrix}
1&\cdots&1\\
2&\cdots&2\\
\vdots&&\vdots\\
8&\cdots&8
\end{bmatrix}
,\quad
B = \mathbf{1}_{8\times 8}
\quad\Rightarrow\quad
C_{ij} = 8\cdot i
\quad(i=1\ldots 8)
\]

Row \(r\) of \(C\) is all \(8r\) (1-based), e.g. row 0 → all 8, row 7 → all 64.  
Here \(K=8\), so one hardware run fills the whole \(C\) (no \(K\) accumulation).

## Where this lives in code

- Math / loops: `npu_matmul()` in `host/npukit_matmul.py`
- Printed plans + matrices: `describe_tiling()`, `run_case(..., verbose=True)`
- Narrative in Jupyter: cells that call the same helpers so saved outputs keep A/B/C visible

#!/usr/bin/env python3
"""One-shot PYNQ board smoke: matmul → glue → e2e → tiny-ViT.

Runs the four host suites against the same bitstream and prints a single summary.

Usage (on PYNQ, sudo + XRT/venv):
  python3 npukit_board_smoke.py /home/xilinx/jupyter_notebooks/npukit.bit
  python3 npukit_board_smoke.py /home/xilinx/jupyter_notebooks/npukit.bit --vit-n 64

Notebook wrapper: npukit_board_smoke.ipynb (save after run so dumps stay in the file).
"""

from __future__ import annotations

import argparse
import io
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BIT = "/home/xilinx/jupyter_notebooks/npukit.bit"


@dataclass
class SuiteResult:
    name: str
    ok: bool
    detail: str
    log: str


def _run_captured(name: str, fn) -> SuiteResult:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = int(fn())
        log = buf.getvalue()
        ok = rc == 0
        detail = "PASS" if ok else f"FAIL (rc={rc})"
        # Prefer last ALL / N/N line if present
        for line in reversed(log.splitlines()):
            s = line.strip()
            if s.endswith("PASS") or s.endswith("FAIL") or "/12 PASS" in s or "/12 FAIL" in s:
                detail = s
                break
        return SuiteResult(name=name, ok=ok, detail=detail, log=log)
    except Exception as exc:
        log = buf.getvalue() + "\n" + traceback.format_exc()
        return SuiteResult(name=name, ok=False, detail=f"EXCEPTION: {exc}", log=log)


def run_all(*, bit_path: str, vit_n: int = 64, matmul_quiet: bool = True) -> list[SuiteResult]:
    # Imports after path setup by caller / notebook
    import npukit_matmul as matmul
    import npukit_transformer as nt
    import npukit_vit_mnist as vit

    results: list[SuiteResult] = []

    def matmul_suite() -> int:
        argv = [bit_path, "32", "32", "32"]
        if matmul_quiet:
            argv.append("--quiet")
        return matmul.main(argv)

    results.append(_run_captured("matmul", matmul_suite))
    results.append(_run_captured("glue", lambda: nt.run_board(bit_path)))
    results.append(_run_captured("e2e", lambda: nt.run_e2e_smoke(bit_path=bit_path, seed=0)))
    results.append(
        _run_captured(
            "vit",
            lambda: vit.run_vit_smoke(bit_path=bit_path, seed=0, n=vit_n, verbose=False),
        )
    )
    return results


def print_summary(results: list[SuiteResult]) -> int:
    print("\n" + "=" * 60)
    print("NpuKit board smoke summary")
    print("=" * 60)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"  [{mark}] {r.name:8s}  {r.detail}")
    n_ok = sum(1 for r in results if r.ok)
    n = len(results)
    all_ok = n_ok == n
    print("-" * 60)
    print(f"OVERALL: {n_ok}/{n} suites PASS" + (" — ALL SMOKE PASS" if all_ok else " — SMOKE FAIL"))
    print("=" * 60)
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NpuKit full board smoke (matmul+glue+e2e+ViT)")
    p.add_argument("bit", nargs="?", default=DEFAULT_BIT, help="path to npukit.bit")
    p.add_argument("--vit-n", type=int, default=64, help="ViT smoke images (default 64)")
    p.add_argument(
        "--verbose-matmul",
        action="store_true",
        help="dump full matmul matrices (default: quiet)",
    )
    p.add_argument(
        "--dump-logs",
        action="store_true",
        help="print full per-suite logs after the summary",
    )
    args = p.parse_args(argv)

    bit = str(Path(args.bit))
    if not Path(bit).exists():
        print(f"missing bitstream: {bit}", file=sys.stderr)
        return 1

    print(f"NpuKit board smoke  bit={bit}  vit_n={args.vit_n}")
    results = run_all(
        bit_path=bit,
        vit_n=args.vit_n,
        matmul_quiet=not args.verbose_matmul,
    )
    rc = print_summary(results)
    if args.dump_logs:
        for r in results:
            print(f"\n##### LOG: {r.name} #####\n{r.log}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

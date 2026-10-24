#!/usr/bin/env bash
# Icarus Verilog checks. No Vivado, no board, no bitstream.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing '$1' (on Ubuntu: sudo apt-get install -y iverilog)" >&2
    exit 1
  fi
}
need iverilog
need vvp

run_sim() {
  local name="$1"
  shift
  local out log
  out="$(mktemp)"
  log="$(mktemp)"
  echo "== ${name} =="
  iverilog -g2012 -o "$out" "$@"
  if ! vvp "$out" | tee "$log"; then
    echo "FAIL: ${name} exited non-zero" >&2
    exit 1
  fi
  if ! grep -q "ALL PASS" "$log"; then
    echo "FAIL: ${name} did not print ALL PASS" >&2
    exit 1
  fi
  rm -f "$out" "$log"
}

# Elaborate-only: the BD wrapper is not simulated (no clock in the wrapper).
compile_top() {
  local name="$1"
  shift
  local out
  out="$(mktemp)"
  echo "== compile ${name} =="
  iverilog -g2012 -s "$name" -o "$out" "$@"
  rm -f "$out"
}

NPU_RTL=(
  "$ROOT/rtl/pe.sv"
  "$ROOT/rtl/systolic_array.sv"
  "$ROOT/rtl/npukit_glue.sv"
  "$ROOT/rtl/npukit_axil.sv"
)

compile_top npukit_pl \
  "${NPU_RTL[@]}" \
  "$ROOT/rtl/npukit_top.sv" \
  "$ROOT/rtl/npukit_pl.v"

run_sim pe_tb \
  "$ROOT/rtl/pe.sv" \
  "$ROOT/sim/pe_tb.sv"

run_sim systolic_array_tb \
  "$ROOT/rtl/pe.sv" \
  "$ROOT/rtl/systolic_array.sv" \
  "$ROOT/sim/systolic_array_tb.sv"

run_sim npukit_glue_tb \
  "$ROOT/rtl/npukit_glue.sv" \
  "$ROOT/sim/npukit_glue_tb.sv"

run_sim npukit_axil_tb \
  "${NPU_RTL[@]}" \
  "$ROOT/sim/npukit_axil_tb.sv"

echo "RTL simulation: ALL PASS"

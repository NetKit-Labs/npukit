#!/usr/bin/env bash
# Host checks that do not open /dev/mem, load a bitstream, or download datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="$ROOT/host"
CPP="$HOST/cpp"

if ! python3 -c "import numpy" >/dev/null 2>&1; then
  echo "missing numpy (pip install 'numpy>=1.26')" >&2
  exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "missing g++" >&2
  exit 1
fi

echo "== python syntax =="
python3 -m compileall -q -x '/(\.venv|__pycache__)/' \
  "$HOST" \
  "$ROOT/viz"

echo "== offline glue / transformer reference =="
(
  cd "$HOST"
  python3 npukit_transformer.py --ref-only
  python3 npukit_transformer.py --e2e-ref
  python3 npukit_vit_mnist.py --ref-only -n 2
  python3 npukit_command_lm.py --phrase "go left"
  python3 gsc_commands.py --demo 2
  python3 -c "import logmel, gsc_audio, vit_ds_stem"
)

echo "== C++ HOST=1 build and CPU parity =="
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
(
  cd "$CPP"
  make clean
  make -j"$jobs" HOST=1
  python3 export_vit_bin.py --n-samples 2
  python3 export_dscnn_bin.py --n-samples 2
  python3 export_command_lm_bin.py --n-samples 4 --out command_lm.bin
  ./npukit_vit --cpu --weights vit_mnist.bin --iters 1
  ./npukit_dscnn --weights dscnn_mnist.bin --iters 1
  ./npukit_command_lm --cpu --weights command_lm.bin --iters 1
)

echo "Host checks: ALL PASS"

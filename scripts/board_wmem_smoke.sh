#!/usr/bin/env bash
# Deploy 0x302 bit + C++ driver to PYNQ, run WMEM smoke/bench, save logs.
# Run from Docker VM after bitstream rebuild.
set -euo pipefail

BOARD="${BOARD:-xilinx@192.168.0.215}"
PASS="${PASS:-xilinx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_LOCAL="${ROOT}/results/wmem_${STAMP}"
REMOTE_CPP="/home/xilinx/npukit_cpp"
REMOTE_NB="/home/xilinx/jupyter_notebooks"
SSH=(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa)
SCP=(sshpass -p "$PASS" scp -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa)

mkdir -p "$OUT_LOCAL"

echo "==> Deploy bitstream"
"${SCP[@]}" "$ROOT/output/npukit.bit" "$ROOT/output/npukit.hwh" "${BOARD}:${REMOTE_NB}/"

echo "==> Sync C++ sources"
"${SSH[@]}" "$BOARD" "mkdir -p ${REMOTE_CPP}/include/npukit ${REMOTE_CPP}/src"
"${SCP[@]}" -r "$ROOT/host/cpp/include/npukit/"*.hpp "${BOARD}:${REMOTE_CPP}/include/npukit/"
"${SCP[@]}" "$ROOT/host/cpp/src/"*.cpp "$ROOT/host/cpp/Makefile" \
  "$ROOT/host/cpp/export_vit_bin.py" "${BOARD}:${REMOTE_CPP}/"
if [[ -f "$ROOT/host/cpp/vit_mnist.bin" ]]; then
  "${SCP[@]}" "$ROOT/host/cpp/vit_mnist.bin" "${BOARD}:${REMOTE_CPP}/"
elif [[ -f "$ROOT/host/vit_mnist_weights.npz" ]]; then
  "${SCP[@]}" "$ROOT/host/vit_mnist_weights.npz" "${BOARD}:${REMOTE_CPP}/"
fi

echo "==> Load bitstream + rebuild C++ + smoke"
"${SSH[@]}" "$BOARD" "bash -s" <<REMOTE | tee "${OUT_LOCAL}/board_smoke.log"
set -euo pipefail
echo xilinx | sudo -S true
source /etc/profile.d/xrt_setup.sh 2>/dev/null || true
source /usr/local/share/pynq-venv/bin/activate
python3 - <<'PY'
from pynq import Bitstream, MMIO
Bitstream("/home/xilinx/jupyter_notebooks/npukit.bit").download()
m = MMIO(0x43C00000, 0x1000)
print(f"ID=0x{m.read(0):08X} VERSION=0x{m.read(4):08X} FEATURES=0x{m.read(0x14):08X}")
assert m.read(0) == 0x4E50554B
ver = m.read(4)
feat = m.read(0x14)
assert ver >= 0x302, hex(ver)
assert feat & 0x10, hex(feat)
print("BITSTREAM_OK VERSION>=0x302 FEAT_WMEM")
PY

cd ${REMOTE_CPP}
make clean
make -j2
if [[ ! -f vit_mnist.bin && -f vit_mnist_weights.npz ]]; then
  python3 export_vit_bin.py --out vit_mnist.bin --weights vit_mnist_weights.npz || \
    python3 export_vit_bin.py --out vit_mnist.bin
fi

echo "===== npukit_bench (WMEM default) ====="
echo xilinx | sudo -S ./npukit_bench 2>&1 | tee /tmp/npukit_bench_wmem.log

echo "===== npukit_bench (WMEM=0 legacy A||B) ====="
echo xilinx | sudo -S env NPUKIT_WMEM=0 ./npukit_bench 2>&1 | tee /tmp/npukit_bench_legacy.log

echo "===== npukit_vit (WMEM default) ====="
echo xilinx | sudo -S ./npukit_vit --weights vit_mnist.bin 2>&1 | tee /tmp/npukit_vit_wmem.log

echo "===== npukit_vit (WMEM=0) ====="
echo xilinx | sudo -S env NPUKIT_WMEM=0 ./npukit_vit --weights vit_mnist.bin 2>&1 | tee /tmp/npukit_vit_legacy.log

echo "SMOKE_COMPLETE"
REMOTE

echo "==> Copy remote logs"
"${SCP[@]}" "${BOARD}:/tmp/npukit_bench_wmem.log" "${BOARD}:/tmp/npukit_bench_legacy.log" \
  "${BOARD}:/tmp/npukit_vit_wmem.log" "${BOARD}:/tmp/npukit_vit_legacy.log" \
  "$OUT_LOCAL/" || true

# Also keep the combined tee log
ls -la "$OUT_LOCAL"
echo "RESULTS_DIR=$OUT_LOCAL"

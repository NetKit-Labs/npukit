#!/usr/bin/env bash
# 5-minute PYNQ-Z2 bring-up → full board smoke (matmul → glue → e2e → ViT).
#
# Run on the board (after copying bit + host files), or from a laptop with sshpass:
#   BOARD=xilinx@192.168.0.215 ./host/board_bringup_5min.sh
#
# Prefer the notebook for saved dumps:
#   open host/npukit_board_smoke.ipynb → Run All → Save
set -euo pipefail

BIT_DEFAULT="/home/xilinx/jupyter_notebooks/npukit.bit"
HOST_DIR_DEFAULT="/home/xilinx/jupyter_notebooks"
VIT_N="${VIT_N:-64}"

run_local_on_board() {
  local bit="${1:-$BIT_DEFAULT}"
  echo "==> NpuKit 5-minute bring-up"
  echo "    bit=$bit  vit_n=$VIT_N"
  # shellcheck disable=SC1091
  source /etc/profile.d/xrt_setup.sh 2>/dev/null || true
  # shellcheck disable=SC1091
  source /usr/local/share/pynq-venv/bin/activate
  cd "$HOST_DIR_DEFAULT"
  python3 npukit_board_smoke.py "$bit" --vit-n "$VIT_N"
  echo ""
  echo "Tip: open npukit_board_smoke.ipynb, Run All, then Save to keep dumps in git."
}

run_via_ssh() {
  local board="${BOARD:?set BOARD=user@host}"
  local pass="${BOARD_PASS:-xilinx}"
  local repo_host
  repo_host="$(cd "$(dirname "$0")" && pwd)"
  echo "==> Copy host smoke files → $board:$HOST_DIR_DEFAULT"
  sshpass -p "$pass" scp -o StrictHostKeyChecking=no \
    -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa \
    "$repo_host/npukit_board_smoke.py" \
    "$repo_host/npukit_board_smoke.ipynb" \
    "$repo_host/npukit_matmul.py" \
    "$repo_host/npukit_transformer.py" \
    "$repo_host/npukit_vit_mnist.py" \
    "$repo_host/vit_ds_stem.py" \
    "$repo_host/vit_mnist_weights.npz" \
    "$repo_host/vit_mnist_stem.tflite" \
    "$repo_host/mnist_sample.npz" \
    "$board:$HOST_DIR_DEFAULT/"
  echo "==> Remote smoke"
  sshpass -p "$pass" ssh -o StrictHostKeyChecking=no \
    -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa \
    "$board" "echo $pass | sudo -S bash -lc '
      source /etc/profile.d/xrt_setup.sh 2>/dev/null || true
      source /usr/local/share/pynq-venv/bin/activate
      cd $HOST_DIR_DEFAULT
      python3 npukit_board_smoke.py $BIT_DEFAULT --vit-n $VIT_N
    '"
}

if [[ -f /usr/local/share/pynq-venv/bin/activate ]] || [[ -e /dev/fpga0 ]]; then
  run_local_on_board "${1:-$BIT_DEFAULT}"
elif [[ -n "${BOARD:-}" ]]; then
  run_via_ssh
else
  cat <<'EOF'
NpuKit 5-minute board bring-up
==============================

1) Copy bitstream + host (once per bit rebuild):
     scp output/npukit.bit output/npukit.hwh \
       host/board_bringup_5min.sh host/npukit_board_smoke.py host/npukit_board_smoke.ipynb \
       host/npukit_matmul.py host/npukit_transformer.py host/npukit_vit_mnist.py \
       host/vit_ds_stem.py host/vit_mnist_weights.npz host/vit_mnist_stem.tflite \
       host/mnist_sample.npz \
       xilinx@<pynq>:jupyter_notebooks/

2) On the board (sudo + XRT/venv):
     bash board_bringup_5min.sh
   or:
     python3 npukit_board_smoke.py /home/xilinx/jupyter_notebooks/npukit.bit --vit-n 64

3) For git-friendly dumps: open npukit_board_smoke.ipynb → Run All → Save.

From a laptop with sshpass:
     BOARD=xilinx@192.168.0.215 ./host/board_bringup_5min.sh
EOF
  exit 0
fi

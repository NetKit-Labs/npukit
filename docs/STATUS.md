# NpuKit status

Checkpoint **2026-07-28**: hardware MVP + MNIST edge peers remain green; **command-phrase** and **speech-peers** tracks are in-repo with host metrics (and fair-order C win).

## Where we are

**Hardware MVP is complete.** Board is on `VERSION 0x302` with layer-resident weights (`FEAT_WMEM`, `w_mem` = RAMB18):

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Layer-resident weight bank | `LOAD_W` once into BRAM (`K*N≤1024`); A-only kicks + `TILE_KJ` (`FEAT_WMEM`) |
| Weight-stationary + A ping-pong | `LOAD_CFG` A/B-only AXIS; dual-A shadow fill while busy (`FEAT_WS\|PP`) |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz (optional path) |
| Host (A9) | C++ driver (XRT CMA), per-channel quant, DS-stem, float Softmax/RMSNorm/GELU |

Fabric clock: **PS FCLK0 @ 100 MHz**. WMEM smoke build: **WNS ≈ −1.57 ns** (functional OK; timing close is follow-on). Host default remains A∥B; `NPUKIT_WMEM=1` to exercise the bank — see [`results/wmem_20260728T133256Z/`](../results/wmem_20260728T133256Z/).

## Done (summary)

| Area | Status |
|------|--------|
| GEMM + DMA + AXI-Lite + host tiling | Board **12/12 PASS** |
| Transformer glue (v0x300+) | Board unit + e2e PASS @ 100 MHz |
| WMEM / WS / PP (`VERSION 0x302`) | Board correct; not faster than A∥B on tiny-ViT |
| MNIST tiny-ViT (MID=24 stem, L=4, MLP=32) | Numpy deploy **~98.0%**; board **ALL VIT PASS** |
| MCU-class DS-CNN peer (host) | int8 **~98.4%** / ~8 KiB |
| Full board smoke notebook | **ALL SMOKE PASS** |
| Command-phrase tiny LM (text) | Host QAT intent **100%**; C++ pack; see [`command_lm.md`](command_lm.md) |
| Speech peers (audio log-mel) | Short / long / **fair** races; fair C **96.5%** vs A **27.8%** — [`speech_peers.md`](speech_peers.md) |
| Viz | Systolic Manim/Pillow GIFs, edge-peers + speech-peers + hybrid-transformer anims |

## Board results (2026-07-27 re-smoke)

Captured in [`host/npukit_board_smoke.ipynb`](../host/npukit_board_smoke.ipynb) on `192.168.0.215` with current `vit_mnist_weights.npz`.

| Notebook / suite | What it covers | Result |
|------------------|----------------|--------|
| `host/npukit_board_smoke.ipynb` | matmul → glue → e2e → ViT | **ALL SMOKE PASS** (4/4) |
| matmul | classic 8×8 + tiled 16/32 | **12/12 PASS** |
| glue | unit ops + GEMM tile | **ALL BOARD PASS** |
| e2e | synthetic 1-layer T=8×D=8 | **ALL E2E PASS** |
| vit | DS-stem + T=16×D=16×MLP32×L=4, `glue=float`, int8 GEMM HW | **ALL VIT PASS** |
| `host/dscnn_mnist.ipynb` | Host-only DS-CNN peer (not on FPGA) | float **98.00%** / int8 **98.39%** |

ViT board sample **n=64** (earlier weights in smoke notebook): ref/hw **96.9%**, pred agree **100%**. CLI:

```bash
python npukit_board_smoke.py /home/xilinx/jupyter_notebooks/npukit.bit --vit-n 64
python npukit_vit_mnist.py /home/xilinx/jupyter_notebooks/npukit.bit -n 64
```

## MNIST tiny-ViT (current)

Geometry (no resize):

- Native **28×28** → **CPU richer DS-stem** (MID=24, 3× DS blocks → D=16) → **T=16×D=16** tokens
- Model dim **D=16**, FFN hidden **MLP=32**, **L=4** transformer blocks
- **int8 GEMM** on FPGA; **A9 float32** Softmax / RMSNorm / GELU (`glue_mode=float`)
- **Per-channel weight scales** on stem + GEMM mats; per-stage act / attn-p scales
- Class head on CPU (`N_CLASS=10` not 8-aligned)

| Metric | Result |
|--------|--------|
| **Numpy quantized ref (full 10k test)** | **97.98%** |
| Float test (full 10k) | **98.05%** |
| Weight footprint (int8 + Q12) | **~13 KiB** class |

## Edge comparison (vision)

| Role | Model | Headline |
|------|--------|----------|
| **MCU-class CNN** | Host DS-CNN (int8) | **98.39%** / ~**8.3 KiB** |
| **MCU/MPU + accelerator ViT** | Tiny-ViT on NpuKit | **97.98%** / ~**13 KiB** |

Animated summary: [`viz/out/edge_peers.gif`](../viz/out/edge_peers.gif). Five-minute board path: [`host/board_bringup_5min.sh`](../host/board_bringup_5min.sh).

## C++ e2e (A9)

| Path | ~ms/img |
|------|--------:|
| Python ViT FPGA (pool + fast stem) | ~614 |
| C++ ViT FPGA GEMM (XRT CMA, A∥B, `0x302`) | **~9.8** |
| Same with `NPUKIT_WMEM=1` | ~10.0 (correct; not faster on tiny-ViT) |
| C++ DS-CNN int8 peer | ~9.3 |
| C++ ViT CPU GEMM | ~6.6 |

See `host/cpp/README.md` and [`weight_stationary.md`](weight_stationary.md).

## Command-phrase tiny LM (text track)

Speech Commands vocab + synthetic robot phrases — **T=32 × D=32 × MLP=64 × L=6** causal LM (int8 GEMM, float norms) + **30-way command/intent head**.

| Metric | Result |
|--------|--------|
| Command / intent (numpy deploy) | **100%** |
| Next-token (shared prefixes) | ~51–55% (near oracle) |

Docs: [`command_lm.md`](command_lm.md). Train/run: `host/train_command_lm.py`, `host/npukit_command_lm.py`, C++ under `host/cpp/`.

## Speech peers (audio track)

Same stitched GSC audio + shared log-mel; peers A (fat/causal DS-CNN), B (KWS+FSM), C (hybrid stem + transformer). Full write-up: [`speech_peers.md`](speech_peers.md).

| Race | Headline |
|------|----------|
| Short (1–2 word) | A **92%** > C **~76%** > B **~26%** |
| Long (9-word scripts) | A **84.5%** > C **50%** > B **0%** |
| **Fair order-only** (same multiset, permutations) | **C 96.5%** > A causal CNN **27.8%** |

Fair race is the clean “order matters” story for the hybrid transformer. Short/long metrics and checkpoints live under `host/speech_peers_metrics*.json` and `host/speech_peers_weights*/`.

```bash
python3 host/speech_peers.py --fair    # order-only
python3 host/speech_peers.py --long     # 9-word scripts
python3 viz/speech_peers_anim.py        # → viz/out/speech_peers.gif
```

## Visualize

| Script | Output |
|--------|--------|
| `viz/systolic_manim.py` + `viz/render_linkedin.sh` | LinkedIn square GIF/MP4 of the 8×8 array |
| `viz/systolic_anim.py` | Pillow systolic GIF |
| `viz/edge_peers_anim.py` | MCU DS-CNN vs tiny-ViT story |
| `viz/speech_peers_anim.py` | Speech-peers fair/long story |
| `viz/hybrid_transformer_anim.py` | Hybrid stem + transformer story |

## Still optionally valuable

1. **Close 100 MHz timing** on the WMEM bitstream (smoke WNS ≈ −1.57 ns; functional OK)
2. **Board bring-up** for command-LM / fair hybrid-transformer packs on PYNQ (host metrics done; FPGA latency vs A9 still tile-bound on short audio C)
3. Fuse GSC **keyword-spotting audio front-end** into the text command-LM body (optional product path)
4. Close the last **~1 pp** vs DS-CNN on vision MNIST
5. Multi-head attention + **per-head scales**
6. Larger PE grid / model only after host traffic is no longer the limit (prefer stay 8×8 + tile)

## Geometry notes

- GEMM tile is always **8×8**; larger matmuls are host-tiled.
- HW glue Softmax/RMSNorm/GELU stay capped at `MAX_LEN=16`; deploy paths use **float** norms when lengths exceed that.
- Multi-head attention does **not** require larger `MAX_LEN`.

Related docs: [`speech_peers.md`](speech_peers.md), [`command_lm.md`](command_lm.md), [`weight_stationary.md`](weight_stationary.md), [`transformer_split.md`](transformer_split.md), [`transformer_glue.md`](transformer_glue.md), [`glue_bringup.md`](glue_bringup.md), [`tiling.md`](tiling.md), [`../PLAN.md`](../PLAN.md).

# NpuKit status

Checkpoint **2026-07-28**: **robotic Google Speech Commands** tracks are **complete** on the host; MNIST was only a **fabric / bring-up sanity check**.

## Product track vs bring-up

| Track | Role | Status |
|-------|------|--------|
| **GSC robot commands (audio + text)** | Target application story | **Complete** (host metrics) |
| **MNIST tiny-ViT / DS-CNN** | Sanity check while bringing up systolic GEMM, DMA, tiling | Done — **not** the product target |
| **Hardware MVP** (`VERSION 0x302`) | 8×8 int8 GEMM + glue + WMEM | Board green |

## Robot commands (Google Speech Commands) — results

Shared front-end: stitched GSC WAVs, log-mel `sr=16k`, `n_fft=512`, `hop=256`, `n_mels=32`. Docs: [`speech_peers.md`](speech_peers.md), [`command_lm.md`](command_lm.md).

### Audio peers

| Race | A Fat/causal CNN | B KWS+FSM | C Hybrid transformer |
|------|-----------------:|----------:|---------------------:|
| Short (1–2 word) | **92.0%** (37.7k / ~37 KiB) | 25.8% (29.0k) | 75.2% (54.4k / ~53 KiB) |
| Long (9-word) | **84.5%** (102k / ~100 KiB) | **0%** | 50.0% (95.3k / ~93 KiB) |
| **Fair order-only** (8-word) | 27.8% (99.8k / ~98 KiB) | — | **96.5%** (97.4k / ~95 KiB) |

Fair race detail (order is the only class cue; same word multiset, permutations):

| Peer | Acc | Params | fp32 KiB | ~int8 KiB | ms/phrase |
|------|----:|-------:|---------:|----------:|----------:|
| A Causal DS-CNN (no GAP) | 27.8% | 99,849 | 390.0 | 97.5 | 16.7 |
| **C Hybrid transformer** | **96.5%** | **97,408** | **380.5** | **95.1** | **15.4** |

Takeaway: bag/content CNNs win short/long when phrases differ in words; when **order** is all that matters, **HT wins cleanly**.

### Text command-phrase LM

| Metric | Result |
|--------|--------|
| Geometry | T=32 × D=32 × MLP=64 × L=6 |
| Intent head (numpy deploy) | **100%** |
| Next-token (shared prefixes) | ~51–55% |

## Hardware (bring-up platform)

Board is on `VERSION 0x302` with layer-resident weights (`FEAT_WMEM`, `w_mem` = RAMB18):

| Block | Role |
|-------|------|
| 8×8 int8 systolic GEMM | Output-stationary; tiled MxKxN via host DMA/MMIO |
| Layer-resident weight bank | `LOAD_W` once (`K*N≤1024`); A-only kicks |
| Weight-stationary + A ping-pong | `FEAT_WS\|PP` |
| Transformer glue | Residual / GELU / RMSNorm / Softmax, `MAX_LEN=16`, 100 MHz |

Fabric clock: **PS FCLK0 @ 100 MHz**. WMEM smoke: **WNS ≈ −1.57 ns** (functional OK). Default host stays A∥B; `NPUKIT_WMEM=1` optional — [`results/wmem_20260728T133256Z/`](../results/wmem_20260728T133256Z/).

## MNIST (sanity check only)

Used to prove fabric + host before GSC work — **not** the deployment goal.

| Suite | Result |
|-------|--------|
| Board smoke (matmul → glue → e2e → ViT) | **ALL SMOKE PASS** |
| Tiny-ViT numpy deploy (10k) | **~98.0%** |
| Host DS-CNN int8 peer | **~98.4%** / ~8.3 KiB |
| ViT board n=64 pred agree | **100%** |

## Still optionally valuable

1. Board bring-up for command-LM / fair HT packs on PYNQ
2. Close **100 MHz timing** on the WMEM bitstream
3. Multi-head attention + per-head scales
4. Larger PE grid only after host traffic is no longer the limit

Related: [`speech_peers.md`](speech_peers.md), [`command_lm.md`](command_lm.md), [`weight_stationary.md`](weight_stationary.md), [`../PLAN.md`](../PLAN.md), [`../README.md`](../README.md).

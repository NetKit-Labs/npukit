# Speech peers (log-mel command phrases)

**Status: complete (host).** This is the NpuKit **robot-command** application track (Google Speech Commands audio). MNIST digit work elsewhere in the repo was only FPGA bring-up / sanity check.

Three peers on the **same** stitched Google Speech Commands phrases and the **same** log-mel front-end.

## Front-end (shared)

| Param | Short | Long (`--long`) |
|-------|-------|-----------------|
| Sample rate | 16 kHz | 16 kHz |
| STFT | `n_fft=512`, `hop=256` (50%) | same |
| Mel | `n_mels=32` | same |
| Clip | 32768 → **127** frames (2.048 s) | 262144 → **1023** frames (16.384 s) |
| Commands | 22 × 1–2 words | 22 × **9-word** order-sensitive scripts |

See `host/logmel.py`, `host/gsc_audio.py`.

## Peers

| Peer | Model | Role |
|------|--------|------|
| **A** | Fat DS-CNN | Full-phrase log-mel → 22-way |
| **B** | KWS CNN + FSM | Sliding-window word KWS → longest command match |
| **C** | Mel / hybrid DS-stem → transformer → intent | Short: thin stem T=32; long: fat DS stem T=64×D=32×L=6 |

## Long phrases (order matters) — headline

9-word minimal-pair scripts (e.g. `go left go right …` vs `go right go left …`), 80 train / 20 val per cmd, 30 epochs. Metrics: `host/speech_peers_metrics_long.json`.

| Peer | Acc % | Params | KiB (int8 est.) | ms/phrase (host) |
|------|------:|-------:|----------------:|-----------------:|
| A Fat DS-CNN | **84.5** | 102k | 99.7 | 30.1 |
| B KWS+FSM | **0.0** | ~29k | ~28 | 75.4 |
| C Hybrid DS-stem + Transformer (Torch) | 50.0 | 95.3k | 93.1 | **21.8** |

Notes:

- Even with **order-sensitive 9-word** phrases, **A still beats C** (84.5% vs 50%).
- **B** collapses: word-KWS+FSM cannot recover exact 9-word sequences reliably (0% phrase match) even though word-KWS alone trains to ~69%.
- C uses a **stronger hybrid DS stem** and T=64 tokens; body GEMM is Torch float on host (`--long` skips NpuKit int8 by default).

## Short phrases (legacy)

1–2 word commands. Metrics: `host/speech_peers_metrics.json`.

| Peer | Acc % | KiB (int8 est.) | ms/phrase (host) |
|------|------:|----------------:|-----------------:|
| A Fat DS-CNN | **92.0** | 36.8 | **1.85** |
| B KWS+FSM | 25.8 | 28.3 | 13.4 |
| C Torch / NpuKit CPU int8 | 75.2 / **75.8** | 53.1 | 4.7 / 15.1 |

Board peer-C FPGA (short deploy pack): same accuracy as CPU int8; PL was slower than A9 tiled GEMM on this geometry (~5.2 s vs ~0.5 s/phrase) due to AXI tile overhead.

## Fair order-only race (`--fair`)

Same 8-word **multiset**, 16 permutations (bag features cannot separate classes). Right-aligned 16.384 s mel.

| Peer | Recipe |
|------|--------|
| **A** | **Causal** DS-CNN, **no global pool** — last time column only |
| **C** | Hybrid DS-stem → **T=128** × D=32 × L=6 → **last-token** head + ordered word-ID aux loss |

Results (`host/speech_peers_metrics_fair.json`):

| Peer | Acc % | ms/phrase |
|------|------:|----------:|
| A Causal DS-CNN | 27.8 | 16.7 |
| **C Hybrid transformer** | **96.5** | **15.4** |

On this setup **C wins cleanly** — blocking the CNN bag + giving the transformer resolution/sequential aux flips the earlier long-phrase story.

```bash
python3 host/speech_peers.py --fair
# metrics → host/speech_peers_metrics_fair.json
```

## Run

```bash
python3 host/gsc_commands.py --download          # once
python3 host/speech_peers.py --fair              # order-only causal CNN vs transformer
python3 host/speech_peers.py --long              # 9-word scripts (legacy long)
python3 host/speech_peers.py                     # short 1–2 word race
python3 host/speech_peers.py --fair --quick      # smoke
```

## Files

| Path | Role |
|------|------|
| `host/logmel.py` | Shared STFT / log-mel (short + long canvas) |
| `host/gsc_audio.py` | Phrase stitch; `set_phrase_mode("short"|"long")` |
| `host/speech_peers.py` | Train / bench |
| `host/speech_peers_metrics.json` | Short-mode summary |
| `host/speech_peers_metrics_long.json` | Long-mode summary |
| `host/speech_peers_metrics_fair.json` | Fair order-only summary |
| `host/speech_peers_weights(_long|_fair)/` | Saved peer checkpoints |
| `host/speech_peers_val*.npz` | Val mel for `--eval-only` (gitignored; regenerable) |
| `viz/speech_peers_anim.py` | Story GIF → `viz/out/speech_peers.gif` |

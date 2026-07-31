# Command-phrase LM (Speech Commands vocab)

**Status: complete (host).** Text-side robot intents over the Google Speech Commands vocab (companion to the [audio speech peers](speech_peers.md)). MNIST was bring-up only.

Geometry: **T=32 × D=32 × MLP=64 × L=6**, causal decoder, int8 GEMM on NpuKit, float Softmax/RMSNorm/GELU on the A9.

## What this is

A **causal transformer** over the [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands) word list (35 labels) plus a few robot glue tokens (`move`, `turn`, `me`). Training uses a **prefix-sharing** robot command inventory (`go left`, `turn on`, …).

Because many commands share prefixes (`go left` / `go right`), next-token accuracy is capped near ~55%. The headline robot metric is the **command/intent head** (mean-pool → 30-way classify), with the LM head kept as a sequential auxiliary — the setting where a DS-CNN peer needs awkward glue.

Optional: download the full GSC audio tarball for later keyword-spotting work (same vocab):

```bash
python3 host/gsc_commands.py --download
```

**Audio peers** (stitched GSC WAVs, shared log-mel) live in [`speech_peers.md`](speech_peers.md).

Short (1–2 word): A **92%** > C **76%** > B **26%**.  
Long (9-word order-sensitive, hybrid stem): A **84.5%** > C **50%** > B **0%**.

## Results (host QAT export)

| Metric | Result |
|--------|--------|
| Geometry | T=32 D=32 MLP=64 L=6 V=42 N_CMD=30 |
| Command / intent accuracy (numpy deploy) | **100%** |
| Next-token accuracy (shared prefixes) | ~51–55% (near oracle) |
| Example | `classify 'go left' → go left` |

## Train / run (host)

```bash
python3 host/train_command_lm.py                  # → command_lm_weights.npz
python3 host/npukit_command_lm.py --phrase "go left" --eval 256
```

Board (GEMM on PL):

```bash
python3 host/npukit_command_lm.py /path/to/npukit.bit --phrase "go left"
```

## C++ pack

```bash
cd host/cpp
python3 export_command_lm_bin.py --out command_lm.bin
make HOST=1 npukit_command_lm
./npukit_command_lm --cpu --weights command_lm.bin
```

On PYNQ (bitstream already loaded): `sudo ./npukit_command_lm --weights command_lm.bin`.

## FPGA notes

- Keep **8×8** systolic tiling; no PE-grid change.
- HW glue `MAX_LEN=16` → norms/softmax stay **float**.
- WMEM holds at most `K*N≤1024` (e.g. 32×32 QKV); FFN 32×64 uses default A∥B.
- Expect FPGA to help more here than MNIST ViT once mats are 32-wide and weights reuse across tokens.

## Files

| Path | Role |
|------|------|
| `host/gsc_commands.py` | Vocab, phrases, optional GSC download |
| `host/train_command_lm.py` | Float warm-up → QAT → export |
| `host/npukit_command_lm.py` | Numpy deploy + greedy complete |
| `host/cpp/export_command_lm_bin.py` | `NKL1` binary |
| `host/cpp/src/command_lm*.cpp` | C++ load / forward / CLI |

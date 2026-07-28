#!/usr/bin/env python3
"""Tiny command-phrase LM for NpuKit (D=32, T=32, L=6) over Speech Commands vocab.

Token embed + positional + L transformer blocks (int8 GEMM on FPGA) + LM head.
Softmax / RMSNorm / GELU stay float on the A9 (T,D > glue MAX_LEN=16).

Usage:
  python3 host/npukit_command_lm.py --weights host/command_lm_weights.npz
  python3 host/npukit_command_lm.py /path/to/npukit.bit --phrase "go left"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gsc_commands as gsc
import npukit_transformer as nt

HOST_DIR = Path(__file__).resolve().parent

# --- Command LM geometry (host contract) ---
LM_T = 32
LM_D = 32
LM_MLP = 64  # 2×D; needs glue_mode='float'
N_LAYERS = 6
VOCAB_SIZE = gsc.VOCAB_SIZE
DEFAULT_GLUE_MODE = "float"
WEIGHTS_PATH = HOST_DIR / "command_lm_weights.npz"

assert LM_T % 8 == 0 and LM_D % 8 == 0 and LM_MLP % 8 == 0
assert N_LAYERS >= 1


@dataclass
class QuantScales:
    act: float = 64.0
    w: float = 64.0
    p: float = 127.0


@dataclass
class CommandLmWeights:
    pos: np.ndarray  # Q12 [T,D]
    w_emb: np.ndarray  # int8 [V,D]
    sw_emb: np.ndarray  # [D] or [V,D] col scales — we use per-col on embed rows@W style: emb is [V,D], lookup then scale
    scale_emb_act: float
    blocks: list[nt.TinyBlockWeights]
    scale_blocks: list[QuantScales]
    w_lm: np.ndarray  # int8 [D,V]
    scale_lm: QuantScales
    sw_lm: np.ndarray  # [V]
    w_cmd: np.ndarray | None = None  # int8 [D, N_CMD]
    scale_cmd: QuantScales | None = None
    sw_cmd: np.ndarray | None = None
    n_cmd: int = 0
    t: int = LM_T
    d: int = LM_D
    mlp: int = LM_MLP
    layers: int = N_LAYERS
    vocab: int = VOCAB_SIZE

    @staticmethod
    def load(path: Path | str) -> "CommandLmWeights":
        data = np.load(path, allow_pickle=False)
        t = int(data["meta_t"]) if "meta_t" in data.files else LM_T
        d = int(data["meta_d"]) if "meta_d" in data.files else LM_D
        mlp = int(data["meta_mlp"]) if "meta_mlp" in data.files else LM_MLP
        layers = int(data["meta_layers"]) if "meta_layers" in data.files else N_LAYERS
        vocab = int(data["meta_vocab"]) if "meta_vocab" in data.files else VOCAB_SIZE
        blocks: list[nt.TinyBlockWeights] = []
        scale_blocks: list[QuantScales] = []
        for i in range(layers):
            def _sw(name: str, n: int) -> np.ndarray:
                key = f"scale_block{i}_{name}_w"
                if key in data.files:
                    arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
                    if arr.size == 1:
                        return np.full(n, float(arr[0]), dtype=np.float64)
                    return arr
                return np.full(n, float(data.get(f"scale_block{i}_w", 64.0)), dtype=np.float64)

            blk = nt.TinyBlockWeights(
                wq=np.asarray(data[f"wq{i}"], dtype=np.int8),
                wk=np.asarray(data[f"wk{i}"], dtype=np.int8),
                wv=np.asarray(data[f"wv{i}"], dtype=np.int8),
                wo=np.asarray(data[f"wo{i}"], dtype=np.int8),
                w1=np.asarray(data[f"w1{i}"], dtype=np.int8),
                w2=np.asarray(data[f"w2{i}"], dtype=np.int8),
                gamma1=np.asarray(data[f"gamma1{i}"], dtype=np.int32),
                gamma2=np.asarray(data[f"gamma2{i}"], dtype=np.int32),
                sw_wq=_sw("wq", d),
                sw_wk=_sw("wk", d),
                sw_wv=_sw("wv", d),
                sw_wo=_sw("wo", d),
                sw_w1=_sw("w1", mlp),
                sw_w2=_sw("w2", d),
            )
            blocks.append(blk)
            scale_blocks.append(
                QuantScales(
                    act=float(data[f"scale_block{i}_act"]),
                    w=float(data[f"scale_block{i}_w"]) if f"scale_block{i}_w" in data.files else 64.0,
                    p=float(data[f"scale_block{i}_p"]),
                )
            )
        sw_lm = np.asarray(data["scale_lm_w_ch"], dtype=np.float64).reshape(-1)
        if sw_lm.size == 1:
            sw_lm = np.full(vocab, float(sw_lm[0]), dtype=np.float64)
        sw_emb = np.asarray(data["scale_emb_w_ch"], dtype=np.float64).reshape(-1)
        if sw_emb.size == 1:
            sw_emb = np.full(d, float(sw_emb[0]), dtype=np.float64)
        w_cmd = sw_cmd = None
        scale_cmd = None
        n_cmd = int(data["meta_n_cmd"]) if "meta_n_cmd" in data.files else 0
        if "w_cmd" in data.files:
            w_cmd = np.asarray(data["w_cmd"], dtype=np.int8)
            sw_cmd = np.asarray(data["scale_cmd_w_ch"], dtype=np.float64).reshape(-1)
            if sw_cmd.size == 1:
                sw_cmd = np.full(n_cmd, float(sw_cmd[0]), dtype=np.float64)
            scale_cmd = QuantScales(act=float(data["scale_cmd_act"]), w=float(sw_cmd.mean()))
            n_cmd = int(w_cmd.shape[1])
        return CommandLmWeights(
            pos=np.asarray(data["pos"], dtype=np.int32),
            w_emb=np.asarray(data["w_emb"], dtype=np.int8),
            sw_emb=sw_emb,
            scale_emb_act=float(data["scale_emb_act"]),
            blocks=blocks,
            scale_blocks=scale_blocks,
            w_lm=np.asarray(data["w_lm"], dtype=np.int8),
            scale_lm=QuantScales(
                act=float(data["scale_lm_act"]),
                w=float(sw_lm.mean()),
                p=127.0,
            ),
            sw_lm=sw_lm,
            w_cmd=w_cmd,
            scale_cmd=scale_cmd,
            sw_cmd=sw_cmd,
            n_cmd=n_cmd,
            t=t,
            d=d,
            mlp=mlp,
            layers=layers,
            vocab=vocab,
        )


def _dequant_embed_row(w_emb: np.ndarray, sw_emb: np.ndarray, token_id: int) -> np.ndarray:
    """int8 row [D] → float via per-column scales."""
    row = w_emb[token_id].astype(np.float64)
    return row / sw_emb.reshape(-1)


def embed_tokens_q12(
    token_ids: np.ndarray,
    w: CommandLmWeights,
) -> np.ndarray:
    """[T] int ids → [T,D] Q12 (embed + pos). PAD stays near zero."""
    t, d = w.t, w.d
    ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    assert ids.shape[0] == t
    x = np.zeros((t, d), dtype=np.float64)
    for i, tid in enumerate(ids):
        if int(tid) == gsc.PAD_ID:
            continue
        x[i] = _dequant_embed_row(w.w_emb, w.sw_emb, int(tid))
    x_q12 = np.rint(x * nt.ONE_Q12).astype(np.int32)
    return x_q12 + w.pos


def command_lm_encode(
    token_ids: np.ndarray,
    w: CommandLmWeights,
    *,
    mmio=None,
    transport=None,
    use_hw_gemm: bool = False,
    glue_mode: str = DEFAULT_GLUE_MODE,
) -> np.ndarray:
    """Causal body → hidden Q12 [T, D]."""
    ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    x = embed_tokens_q12(ids, w)
    pad_mask = ids == gsc.PAD_ID
    for li, blk in enumerate(w.blocks):
        sc = w.scale_blocks[li]
        x, _ = nt.transformer_block_1layer(
            x,
            blk,
            mmio=mmio,
            transport=transport,
            use_hw=False,
            use_hw_gemm=use_hw_gemm,
            glue_mode=glue_mode,
            scale_act=sc.act,
            scale_w=sc.w,
            scale_p=sc.p,
            causal=True,
            key_pad_mask=pad_mask,
            verbose=False,
        )
    return x


def command_lm_forward(
    token_ids: np.ndarray,
    w: CommandLmWeights,
    *,
    mmio=None,
    transport=None,
    use_hw_gemm: bool = False,
    glue_mode: str = DEFAULT_GLUE_MODE,
) -> np.ndarray:
    """Causal body → logits Q12 [T, V] (float norms; int8 GEMM)."""
    x = command_lm_encode(
        token_ids,
        w,
        mmio=mmio,
        transport=transport,
        use_hw_gemm=use_hw_gemm,
        glue_mode=glue_mode,
    )
    return nt._matmul_q12(
        x,
        w.w_lm,
        mmio=mmio,
        transport=transport,
        use_hw=use_hw_gemm,
        scale_act=w.scale_lm.act,
        scale_w=w.sw_lm,
    )


def classify_command(
    token_ids: np.ndarray,
    w: CommandLmWeights,
    *,
    mmio=None,
    transport=None,
    use_hw_gemm: bool = False,
    glue_mode: str = DEFAULT_GLUE_MODE,
) -> int:
    """Mean-pool + command head → command id."""
    assert w.w_cmd is not None and w.sw_cmd is not None and w.scale_cmd is not None
    ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    x = command_lm_encode(
        ids,
        w,
        mmio=mmio,
        transport=transport,
        use_hw_gemm=use_hw_gemm,
        glue_mode=glue_mode,
    )
    mask = ids != gsc.PAD_ID
    pooled = np.rint(x[mask].astype(np.float64).mean(axis=0)).astype(np.int32)
    logits = nt._matmul_q12(
        pooled.reshape(1, -1),
        w.w_cmd,
        mmio=mmio,
        transport=transport,
        use_hw=use_hw_gemm,
        scale_act=w.scale_cmd.act,
        scale_w=w.sw_cmd,
    ).reshape(-1)
    return int(np.argmax(logits))


def greedy_complete(
    prompt_words: list[str],
    w: CommandLmWeights,
    *,
    max_new: int = 6,
    use_hw_gemm: bool = False,
    mmio=None,
    transport=None,
) -> list[str]:
    ids = gsc.encode_words(prompt_words, add_eos=False)
    # Ensure BOS
    if not ids or ids[0] != gsc.BOS_ID:
        ids = [gsc.BOS_ID] + ids
    out_words = list(prompt_words)
    for _ in range(max_new):
        x = gsc.pad_to_t(ids, w.t)
        logits = command_lm_forward(
            np.asarray(x, dtype=np.int32),
            w,
            use_hw_gemm=use_hw_gemm,
            mmio=mmio,
            transport=transport,
        )
        # Predict at last real token position
        pos = min(len(ids) - 1, w.t - 1)
        nxt = int(np.argmax(logits[pos]))
        if nxt in (gsc.EOS_ID, gsc.PAD_ID, gsc.BOS_ID):
            break
        ids.append(nxt)
        tok = gsc.VOCAB[nxt] if 0 <= nxt < len(gsc.VOCAB) else "<unk>"
        out_words.append(tok)
        if len(ids) >= w.t:
            break
    return out_words


def next_token_accuracy(
    xs: np.ndarray,
    ys: np.ndarray,
    w: CommandLmWeights,
    *,
    use_hw_gemm: bool = False,
    mmio=None,
    transport=None,
    max_batches: int | None = None,
) -> float:
    """Teacher-forced token accuracy ignoring PAD targets."""
    n = xs.shape[0]
    if max_batches is not None:
        n = min(n, max_batches)
    correct = 0
    total = 0
    for i in range(n):
        logits = command_lm_forward(
            xs[i],
            w,
            use_hw_gemm=use_hw_gemm,
            mmio=mmio,
            transport=transport,
        )
        pred = np.argmax(logits, axis=-1)
        mask = ys[i] != gsc.PAD_ID
        correct += int(np.sum((pred == ys[i]) & mask))
        total += int(np.sum(mask))
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bit", nargs="?", default=None, help="Optional npukit.bit for HW GEMM")
    ap.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--phrase", type=str, default="go", help="Prompt words, space-separated")
    ap.add_argument("--eval", type=int, default=0, help="Eval next-token acc on N synthetic phrases")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"missing weights: {args.weights}  (run host/train_command_lm.py)", file=sys.stderr)
        sys.exit(1)
    w = CommandLmWeights.load(args.weights)
    print(
        f"loaded {args.weights.name}: T={w.t} D={w.d} MLP={w.mlp} L={w.layers} V={w.vocab}"
    )

    mmio = transport = None
    use_hw = False
    if args.bit:
        from npukit_matmul import open_device  # local import

        mmio, transport = open_device(args.bit)
        use_hw = True
        print("HW GEMM enabled")

    prompt = args.phrase.strip().split()
    t0 = time.perf_counter()
    completed = greedy_complete(
        prompt, w, use_hw_gemm=use_hw, mmio=mmio, transport=transport
    )
    dt = (time.perf_counter() - t0) * 1e3
    print(f"prompt: {' '.join(prompt)}")
    print(f"complete: {' '.join(completed)}  ({dt:.1f} ms)")

    if args.eval > 0:
        phrases = gsc.generate_phrases(args.eval, seed=42)
        xs, ys = gsc.phrases_to_tensors(phrases, w.t, seed=42)
        acc = next_token_accuracy(
            np.asarray(xs, dtype=np.int32),
            np.asarray(ys, dtype=np.int32),
            w,
            use_hw_gemm=use_hw,
            mmio=mmio,
            transport=transport,
        )
        print(f"next-token accuracy @N={len(xs)}: {100*acc:.1f}%")
        if w.w_cmd is not None:
            ok = 0
            for ph, xrow in zip(phrases, xs):
                # phrases_to_tensors shuffles — classify from reconstructed xrow
                pred = classify_command(
                    np.asarray(xrow, dtype=np.int32),
                    w,
                    use_hw_gemm=use_hw,
                    mmio=mmio,
                    transport=transport,
                )
                words = []
                for tid in xrow:
                    if tid in (gsc.PAD_ID, gsc.BOS_ID):
                        continue
                    if tid == gsc.EOS_ID:
                        break
                    words.append(gsc.VOCAB[int(tid)])
                gold = {tuple(c): i for i, c in enumerate(gsc.ROBOT_COMMANDS)}[tuple(words)]
                ok += int(pred == gold)
            print(f"command accuracy @N={len(xs)}: {100*ok/len(xs):.1f}%")

    if w.w_cmd is not None and prompt:
        # Classify full prompt as a command (pad like training: BOS+tokens, no EOS).
        ids = gsc.pad_to_t(gsc.encode_words(prompt, add_eos=False), w.t)
        cid = classify_command(
            np.asarray(ids, dtype=np.int32),
            w,
            use_hw_gemm=use_hw,
            mmio=mmio,
            transport=transport,
        )
        print(f"classify: {' '.join(gsc.ROBOT_COMMANDS[cid])}")


if __name__ == "__main__":
    main()

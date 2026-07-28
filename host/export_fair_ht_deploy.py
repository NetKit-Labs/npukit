#!/usr/bin/env python3
"""Export fair Hybrid Transformer deploy pack (stem tokens + int8 body).

Writes host/speech_peers_fair_ht_deploy.npz for ht_fair_board_eval.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import gsc_audio as ga
import npukit_transformer as nt
import speech_peers as sp

HOST_DIR = Path(__file__).resolve().parent
OUT = HOST_DIR / "speech_peers_fair_ht_deploy.npz"
MAX_N = 64


def main() -> None:
    sp.configure_paths(fair=True)
    x_va, y_va = sp._load_val_npz()
    c = sp.AudioCommandTransformer()
    c.load_state_dict(torch.load(sp.WEIGHTS_DIR / "audio_transformer.pt", map_location="cpu"))
    c.eval()
    n = min(len(y_va), MAX_N)
    tokens = []
    with torch.no_grad():
        for i in range(n):
            tok = c.stem(torch.from_numpy(x_va[i : i + 1])).numpy()[0]
            tok = tok + c.pos.detach().cpu().numpy()
            tokens.append(tok.astype(np.float32))
    tokens_a = np.stack(tokens, axis=0)

    # Host Torch reference accuracy on this subset (last-token).
    with torch.no_grad():
        logits = c(torch.from_numpy(x_va[:n]))
        torch_acc = float((logits.argmax(-1).numpy() == y_va[:n]).mean())

    blocks = [sp._quant_block(b) for b in c.blocks]
    payload: dict = {
        "tokens": tokens_a,
        "labels": y_va[:n].astype(np.int64),
        "n_layers": np.int32(len(blocks)),
        "t": np.int32(tokens_a.shape[1]),
        "d": np.int32(tokens_a.shape[2]),
        "n_cmd": np.int32(ga.N_CMD),
        "head": np.array("last"),
        "torch_acc": np.float64(torch_acc),
        "commands": np.array([" ".join(c) for c in ga.AUDIO_ROBOT_COMMANDS]),
    }
    for li, blk in enumerate(blocks):
        for name in ("wq", "wk", "wv", "wo", "w1", "w2"):
            payload[f"l{li}_{name}"] = getattr(blk, name)
            payload[f"l{li}_sw_{name}"] = getattr(blk, f"sw_{name}")
        payload[f"l{li}_gamma1"] = blk.gamma1
        payload[f"l{li}_gamma2"] = blk.gamma2

    w_cmd = c.w_cmd.detach().cpu().numpy()
    amax = np.maximum(np.abs(w_cmd).max(axis=0), 1e-6)
    sw = (127.0 / amax).astype(np.float64)
    payload["w_cmd_i8"] = nt.quant_weight_to_i8(w_cmd, sw)
    payload["sw_w_cmd"] = sw

    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT} n={n} tokens={tokens_a.shape} torch_acc={100*torch_acc:.1f}%")


if __name__ == "__main__":
    main()

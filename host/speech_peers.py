#!/usr/bin/env python3
"""Speech peer race on shared log-mel: (A) fat DS-CNN, (B) KWS+FSM, (C) hybrid transformer.

Same phrase inventory + log-mel for all peers
(sr=16k, n_fft=512, hop=256 → 50% overlap, n_mels=32, exact STFT).

Modes:
  short (default) — 1–2 word commands, 2.048 s / 127 frames
  --long          — 9-word scripts, 16.384 s / 1023 frames
  --fair          — order-only permutations; full-RF Fat CNN (no GAP) vs HT

Reports accuracy + params + KiB + ms/phrase for each peer.

Usage:
  python3 host/speech_peers.py              # short phrases
  python3 host/speech_peers.py --long       # 9-word scripts
  python3 host/speech_peers.py --fair       # order-only fair A vs C
  python3 host/speech_peers.py --fair --quick
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import gsc_audio as ga
import logmel
import npukit_command_lm as clm
import npukit_transformer as nt

HOST_DIR = Path(__file__).resolve().parent
# Paths are finalized in configure_paths() after mode flags are known.
OUT_JSON = HOST_DIR / "speech_peers_metrics.json"
WEIGHTS_DIR = HOST_DIR / "speech_peers_weights"
VAL_NPZ = HOST_DIR / "speech_peers_val.npz"
TRAIN_NPZ = HOST_DIR / "speech_peers_train.npz"
MEL_FRAMES = logmel.N_FRAMES
TOKEN_T = clm.LM_T  # 32 short; 64 long; 128 fair
HYBRID_STEM = False
FAIR_MODE = False
LAST_TOKEN_HEAD = False


def configure_paths(*, long: bool = False, fair: bool = False) -> None:
    global OUT_JSON, WEIGHTS_DIR, VAL_NPZ, TRAIN_NPZ, MEL_FRAMES, TOKEN_T
    global HYBRID_STEM, FAIR_MODE, LAST_TOKEN_HEAD
    if fair and long:
        raise ValueError("use either --fair or --long, not both")
    if fair:
        ga.set_phrase_mode("fair")
        OUT_JSON = HOST_DIR / "speech_peers_metrics_fair.json"
        WEIGHTS_DIR = HOST_DIR / "speech_peers_weights_fair"
        VAL_NPZ = HOST_DIR / "speech_peers_val_fair.npz"
        TRAIN_NPZ = HOST_DIR / "speech_peers_train_fair.npz"
        MEL_FRAMES = logmel.N_FRAMES_LONG
        TOKEN_T = 128  # more temporal resolution for order
        HYBRID_STEM = True
        FAIR_MODE = True
        LAST_TOKEN_HEAD = True
    elif long:
        ga.set_phrase_mode("long")
        OUT_JSON = HOST_DIR / "speech_peers_metrics_long.json"
        WEIGHTS_DIR = HOST_DIR / "speech_peers_weights_long"
        VAL_NPZ = HOST_DIR / "speech_peers_val_long.npz"
        TRAIN_NPZ = HOST_DIR / "speech_peers_train_long.npz"
        MEL_FRAMES = logmel.N_FRAMES_LONG
        TOKEN_T = 64
        HYBRID_STEM = True
        FAIR_MODE = False
        LAST_TOKEN_HEAD = False
    else:
        ga.set_phrase_mode("short")
        OUT_JSON = HOST_DIR / "speech_peers_metrics.json"
        WEIGHTS_DIR = HOST_DIR / "speech_peers_weights"
        VAL_NPZ = HOST_DIR / "speech_peers_val.npz"
        TRAIN_NPZ = HOST_DIR / "speech_peers_train.npz"
        MEL_FRAMES = logmel.N_FRAMES
        TOKEN_T = clm.LM_T
        HYBRID_STEM = False
        FAIR_MODE = False
        LAST_TOKEN_HEAD = False


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def kib_fp32(n_params: int) -> float:
    return n_params * 4 / 1024.0


def kib_int8(n_params: int) -> float:
    return n_params / 1024.0


# ---------------------------------------------------------------------------
# (A) Fat DS-CNN on full phrase mel
# ---------------------------------------------------------------------------


def _ds_block(cin: int, cout: int, k: int, stride: tuple[int, int]) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(cin, cin, k, stride=stride, padding=pad, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CausalDSBlock(nn.Module):
    """Depthwise-separable block with causal padding on time (no future frames)."""

    def __init__(self, cin: int, cout: int, k: int, stride_t: int) -> None:
        super().__init__()
        self.k = k
        self.stride_t = stride_t
        self.dw = nn.Conv2d(cin, cin, k, stride=(1, stride_t), padding=0, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad: (W_left, W_right, H_top, H_bottom)
        pad_f = self.k // 2
        x = F.pad(x, (self.k - 1, 0, pad_f, pad_f))
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


class FatDSCNN(nn.Module):
    """Deeper DS-CNN with larger kernels over [1, n_mels, T] (global-pool bag)."""

    def __init__(self, n_class: int | None = None, *, long: bool = False) -> None:
        super().__init__()
        n_class = ga.N_CMD if n_class is None else n_class
        ch0 = 48 if long else 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch0, kernel_size=(5, 5), padding=2, bias=False),
            nn.BatchNorm2d(ch0),
            nn.ReLU(inplace=True),
        )
        if long:
            self.blocks = nn.Sequential(
                _ds_block(ch0, 64, 5, (1, 2)),
                _ds_block(64, 96, 5, (1, 2)),
                _ds_block(96, 128, 7, (2, 2)),
                _ds_block(128, 160, 7, (1, 2)),
                _ds_block(160, 192, 7, (2, 2)),
            )
            feat = 192
        else:
            self.blocks = nn.Sequential(
                _ds_block(ch0, 48, 5, (1, 2)),
                _ds_block(48, 64, 5, (2, 2)),
                _ds_block(64, 96, 7, (1, 2)),
                _ds_block(96, 128, 7, (2, 2)),
            )
            feat = 128
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class CausalDSCNN(nn.Module):
    """Causal DS-CNN: no global pool; classify from the last time column only."""

    def __init__(self, n_class: int | None = None) -> None:
        super().__init__()
        n_class = ga.N_CMD if n_class is None else n_class
        self.stem_dw = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=1, padding=0, bias=False)
        self.stem_pw = nn.Conv2d(1, 48, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(48)
        self.blocks = nn.Sequential(
            CausalDSBlock(48, 64, 5, stride_t=2),
            CausalDSBlock(64, 96, 5, stride_t=2),
            CausalDSBlock(96, 128, 7, stride_t=2),
            CausalDSBlock(128, 160, 7, stride_t=2),
            CausalDSBlock(160, 192, 7, stride_t=2),
        )
        self.fc = nn.Linear(192, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal stem on time; symmetric pad on mel axis only.
        x = F.pad(x, (4, 0, 2, 2))
        x = self.stem_dw(x)
        x = F.relu(self.stem_bn(self.stem_pw(x)), inplace=True)
        x = self.blocks(x)
        # Last time column → mean over mel → logits (no bag-over-time pool).
        x = x[:, :, :, -1].mean(dim=-1)
        return self.fc(x)


class DilatedCausalDSBlock(nn.Module):
    """Depthwise-separable block: causal dilated time, symmetric mel, stride 1."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int) -> None:
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.dw = nn.Conv2d(
            cin,
            cin,
            kernel_size=(k, k),
            stride=1,
            padding=0,
            dilation=(1, dilation),
            groups=cin,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.k - 1) * self.dilation
        pad_f = self.k // 2
        x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


class FatRFCNN(nn.Module):
    """Fat CNN: downsample a bit, then dilated causal stack; no GAP.

    Two stride-2 blocks shrink T≈1023→~255, then dilations (1..128) cover that
    timeline so the last column sees the whole utterance. Order can survive
    (unlike global average pooling).
    """

    STEM_K = 5
    BLOCK_K = 3
    DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self, n_class: int | None = None) -> None:
        super().__init__()
        n_class = ga.N_CMD if n_class is None else n_class
        self.stem_dw = nn.Conv2d(
            1, 1, kernel_size=(self.STEM_K, self.STEM_K), stride=1, padding=0, bias=False
        )
        self.stem_pw = nn.Conv2d(1, 48, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(48)
        # 4× temporal downsample before the expensive dilated stack.
        self.down = nn.Sequential(
            CausalDSBlock(48, 64, 5, stride_t=2),
            CausalDSBlock(64, 80, 5, stride_t=2),
        )
        blocks: list[nn.Module] = []
        cin = 80
        outs = (80, 96, 96, 112, 112, 128, 128, 160)
        for d, cout in zip(self.DILATIONS, outs):
            blocks.append(DilatedCausalDSBlock(cin, cout, self.BLOCK_K, d))
            cin = cout
        self.dilated = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, n_class)

    @classmethod
    def theoretical_receptive_field(cls) -> int:
        """Lower-bound RF in input frames (stride-aware)."""
        # stem k=5, then two stride-2 k=5 DS blocks, then dilated k=3 stack.
        rf = float(cls.STEM_K)
        jump = 1.0
        for _ in range(2):
            rf += (5 - 1) * jump
            jump *= 2
        for d in cls.DILATIONS:
            rf += (cls.BLOCK_K - 1) * d * jump
        return int(rf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = self.STEM_K - 1
        pad_f = self.STEM_K // 2
        x = F.pad(x, (pad_t, 0, pad_f, pad_f))
        x = self.stem_dw(x)
        x = F.relu(self.stem_bn(self.stem_pw(x)), inplace=True)
        x = self.down(x)
        x = self.dilated(x)
        x = x[:, :, :, -1].mean(dim=-1)  # last time column; no GAP over time
        return self.fc(x)


# ---------------------------------------------------------------------------
# (B) KWS word CNN + FSM over sliding windows
# ---------------------------------------------------------------------------


class KwsCNN(nn.Module):
    """DS-CNN for single-word / window classification (peer B)."""

    def __init__(self, n_class: int = ga.N_KWS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False),
            nn.Conv2d(48, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=(1, 2), padding=2, groups=64, bias=False),
            nn.Conv2d(64, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 5, stride=(2, 2), padding=2, groups=96, bias=False),
            nn.Conv2d(96, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


def fsm_match(word_ids: list[int]) -> int | None:
    """Longest-prefix match of decoded word sequence against AUDIO_ROBOT_COMMANDS."""
    words = [ga.KWS_WORDS[i] for i in word_ids if 0 <= i < ga.N_KWS]
    if not words:
        return None
    best = None
    best_len = 0
    for cid, cmd in enumerate(ga.AUDIO_ROBOT_COMMANDS):
        # subsequence / contiguous match
        L = len(cmd)
        for i in range(len(words) - L + 1):
            if tuple(words[i : i + L]) == cmd and L >= best_len:
                best = cid
                best_len = L
    return best


@torch.no_grad()
def kws_fsm_predict(
    mel_bt: torch.Tensor,
    kws: KwsCNN,
    *,
    win: int = 32,
    hop: int | None = None,
    conf: float = 0.22,
) -> list[int]:
    """Sliding-window KWS on mel [B,1,M,T] → command ids (or -1)."""
    kws.eval()
    b, _, m, t = mel_bt.shape
    assert m == logmel.N_MELS
    if hop is None:
        hop = 16 if t > 200 else 8
    outs: list[int] = []
    for bi in range(b):
        seq: list[int] = []
        last = -1
        # Short canvas: speech is early. Long canvas: scan the full phrase.
        end = t if t > 200 else min(t, 96)
        for start in range(0, max(1, end - win + 1), hop):
            chunk = mel_bt[bi : bi + 1, :, :, start : start + win]
            if chunk.shape[-1] < win:
                chunk = F.pad(chunk, (0, win - chunk.shape[-1]))
            logits = kws(chunk)
            prob = torch.softmax(logits, dim=-1)[0]
            wid = int(prob.argmax())
            p = float(prob[wid])
            if p < conf:
                continue
            if wid != last:
                seq.append(wid)
                last = wid
        cid = fsm_match(seq)
        outs.append(-1 if cid is None else cid)
    return outs


# ---------------------------------------------------------------------------
# (C) Mel stem → transformer body + command head
# ---------------------------------------------------------------------------


class MelStem(nn.Module):
    """Thin log-mel stem → tokens [B, T, D] (short-phrase default)."""

    def __init__(self, d: int = clm.LM_D, t: int = clm.LM_T) -> None:
        super().__init__()
        self.t = t
        self.d = d
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), groups=32, bias=False),
            nn.Conv2d(32, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, t))
        self.proj = nn.Conv2d(d, d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.proj(x)
        return x.squeeze(2).transpose(1, 2).contiguous()


class MelStemHybrid(nn.Module):
    """Stronger DS-CNN stem → tokens (hybrid for long / fair phrases)."""

    def __init__(self, d: int = clm.LM_D, t: int = 64) -> None:
        super().__init__()
        self.t = t
        self.d = d
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(5, 5), padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # Fair mode: milder time stride so 1023 frames → richer sequence before pool-to-T.
        if t >= 128:
            self.blocks = nn.Sequential(
                _ds_block(48, 64, 5, (1, 2)),
                _ds_block(64, 96, 5, (1, 2)),
                _ds_block(96, 128, 7, (1, 2)),
                _ds_block(128, d, 7, (1, 2)),
            )
        else:
            self.blocks = nn.Sequential(
                _ds_block(48, 64, 5, (1, 2)),
                _ds_block(64, 96, 5, (1, 2)),
                _ds_block(96, 128, 7, (1, 2)),
                _ds_block(128, d, 7, (2, 2)),
            )
        self.pool = nn.AdaptiveAvgPool2d((1, t))
        self.proj = nn.Conv2d(d, d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.proj(x)
        return x.squeeze(2).transpose(1, 2).contiguous()


class AudioCommandTransformer(nn.Module):
    """Peer C: mel stem (+ optional fat DS hybrid) + L blocks + intent head."""

    def __init__(
        self,
        *,
        n_cmd: int | None = None,
        token_t: int | None = None,
        hybrid: bool | None = None,
        last_token: bool | None = None,
        n_word: int | None = None,
    ) -> None:
        super().__init__()
        d, mh = clm.LM_D, clm.LM_MLP
        t = TOKEN_T if token_t is None else token_t
        use_hybrid = HYBRID_STEM if hybrid is None else hybrid
        self.last_token = LAST_TOKEN_HEAD if last_token is None else last_token
        n_cmd = ga.N_CMD if n_cmd is None else n_cmd
        self.stem = MelStemHybrid(d, t) if use_hybrid else MelStem(d, t)
        self.pos = nn.Parameter(torch.zeros(t, d))
        from train_command_lm import TransformerBlock

        self.blocks = nn.ModuleList(TransformerBlock(d, mh) for _ in range(clm.N_LAYERS))
        self.w_cmd = nn.Parameter(torch.randn(d, n_cmd) * 0.12)
        # Sequential aux: predict ordered word IDs from last L tokens (right-aligned speech).
        nw = ga.N_KWS if n_word is None else n_word
        self.n_word_steps = len(ga.AUDIO_ROBOT_COMMANDS[0]) if FAIR_MODE else 0
        self.w_word = (
            nn.Parameter(torch.randn(d, nw) * 0.12) if self.n_word_steps > 0 else None
        )

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel) + self.pos.unsqueeze(0)
        for blk in self.blocks:
            xn = type(blk).rmsnorm(x, blk.gamma1)
            q = xn @ blk.wq
            k = xn @ blk.wk
            v = xn @ blk.wv
            scale = 1.0 / (x.shape[-1] ** 0.5)
            scores = (q @ k.transpose(-1, -2)) * scale
            p = torch.softmax(scores, dim=-1)
            x = x + (p @ v) @ blk.wo
            xn = type(blk).rmsnorm(x, blk.gamma2)
            h = F.gelu(xn @ blk.w1) @ blk.w2
            x = x + h
        return x

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.encode(mel)
        if self.last_token:
            pooled = x[:, -1, :]
        else:
            pooled = x.mean(dim=1)
        return pooled @ self.w_cmd

    def forward_with_words(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.encode(mel)
        if self.last_token:
            cmd_logits = x[:, -1, :] @ self.w_cmd
        else:
            cmd_logits = x.mean(dim=1) @ self.w_cmd
        if self.w_word is None or self.n_word_steps <= 0:
            return cmd_logits, None
        # Last L tokens ↔ L ordered words (canvas is right-aligned in fair mode).
        word_tok = x[:, -self.n_word_steps :, :]  # [B, L, D]
        word_logits = word_tok @ self.w_word  # [B, L, n_kws]
        return cmd_logits, word_logits


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def train_classifier(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    tag: str,
) -> float:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch,
        shuffle=True,
    )
    best = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            # Batched val — full-canvas models OOM / crawl on one giant forward.
            preds = []
            y_t = torch.from_numpy(y_va)
            for i in range(0, len(y_va), batch):
                preds.append(model(torch.from_numpy(x_va[i : i + batch])).argmax(-1))
            acc = float((torch.cat(preds) == y_t).float().mean())
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{tag}] epoch {ep}/{epochs}  val_acc={100*acc:.1f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def train_transformer_sequential(
    model: AudioCommandTransformer,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    tag: str = "C",
    word_loss_w: float = 0.5,
) -> float:
    """Command CE + ordered word-ID CE on last L tokens (sequential aux)."""
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_tr),
            torch.from_numpy(y_tr),
            torch.from_numpy(w_tr),
        ),
        batch_size=batch,
        shuffle=True,
    )
    best = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            opt.zero_grad(set_to_none=True)
            cmd_logits, word_logits = model.forward_with_words(xb)
            loss = F.cross_entropy(cmd_logits, yb, label_smoothing=0.05)
            if word_logits is not None:
                # wb: [B, L] with -1 pad
                flat = word_logits.reshape(-1, word_logits.shape[-1])
                tgt = wb.reshape(-1)
                mask = tgt >= 0
                if mask.any():
                    loss = loss + word_loss_w * F.cross_entropy(flat[mask], tgt[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_va))
            acc = float((logits.argmax(-1) == torch.from_numpy(y_va)).float().mean())
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{tag}] epoch {ep}/{epochs}  val_acc={100*acc:.1f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def accuracy_classifier(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    pred = model(torch.from_numpy(x)).argmax(-1).numpy()
    return float((pred == y).mean())


@torch.no_grad()
def accuracy_kws_fsm(kws: KwsCNN, x: np.ndarray, y: np.ndarray) -> float:
    kws.eval()
    # batch in chunks
    correct = 0
    n = x.shape[0]
    bs = 32
    for i in range(0, n, bs):
        xb = torch.from_numpy(x[i : i + bs])
        preds = kws_fsm_predict(xb, kws)
        for p, g in zip(preds, y[i : i + bs]):
            correct += int(p == int(g))
    return correct / max(n, 1)


def bench_ms(fn, n: int = 20) -> float:
    # warmup
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1e3 / n


def _quant_block(blk, scale_act: float = 64.0) -> nt.TinyBlockWeights:
    """Float TransformerBlock → int8 TinyBlockWeights (per-channel)."""
    def q_w(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        arr = w.detach().cpu().numpy().astype(np.float64)
        amax = np.maximum(np.abs(arr).max(axis=0), 1e-6)
        sw = (127.0 / amax).astype(np.float64)
        return nt.quant_weight_to_i8(arr, sw), sw

    wq, sw_wq = q_w(blk.wq)
    wk, sw_wk = q_w(blk.wk)
    wv, sw_wv = q_w(blk.wv)
    wo, sw_wo = q_w(blk.wo)
    w1, sw_w1 = q_w(blk.w1)
    w2, sw_w2 = q_w(blk.w2)
    return nt.TinyBlockWeights(
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        w1=w1,
        w2=w2,
        gamma1=nt.to_q12(blk.gamma1.detach().cpu().numpy()),
        gamma2=nt.to_q12(blk.gamma2.detach().cpu().numpy()),
        sw_wq=sw_wq,
        sw_wk=sw_wk,
        sw_wv=sw_wv,
        sw_wo=sw_wo,
        sw_w1=sw_w1,
        sw_w2=sw_w2,
    )


@torch.no_grad()
def peer_c_npukit_forward(
    mel_b1mt: np.ndarray,
    model: AudioCommandTransformer,
    blocks_i8: list[nt.TinyBlockWeights],
    *,
    mmio=None,
    transport=None,
    use_hw_gemm: bool = False,
) -> np.ndarray:
    """Float mel-stem on CPU → int8 GEMM transformer body (CPU or FPGA) → cmd logits."""
    model.eval()
    tok = model.stem(torch.from_numpy(mel_b1mt)).numpy()[0]  # [T,D]
    tok = tok + model.pos.detach().cpu().numpy()
    x = nt.to_q12(tok)
    for blk in blocks_i8:
        x, _ = nt.transformer_block_1layer(
            x,
            blk,
            mmio=mmio,
            transport=transport,
            use_hw=False,
            use_hw_gemm=use_hw_gemm,
            glue_mode="float",
            scale_act=64.0,
            scale_p=127.0,
            causal=False,
            verbose=False,
        )
    pooled = np.rint(x.astype(np.float64).mean(axis=0)).astype(np.int32)
    w_cmd = model.w_cmd.detach().cpu().numpy()
    amax = np.maximum(np.abs(w_cmd).max(axis=0), 1e-6)
    sw = (127.0 / amax).astype(np.float64)
    w_i8 = nt.quant_weight_to_i8(w_cmd, sw)
    # N_CMD=22 is not 8-aligned → intent head stays on CPU.
    logits = nt._matmul_q12(
        pooled.reshape(1, -1),
        w_i8,
        mmio=None,
        transport=None,
        use_hw=False,
        scale_act=64.0,
        scale_w=sw,
    ).reshape(-1)
    return logits


def peer_c_npukit_accuracy(
    model: AudioCommandTransformer,
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_hw_gemm: bool = False,
    mmio=None,
    transport=None,
    max_n: int | None = None,
) -> float:
    blocks = [_quant_block(b) for b in model.blocks]
    n = len(y) if max_n is None else min(len(y), max_n)
    ok = 0
    for i in range(n):
        logits = peer_c_npukit_forward(
            x[i : i + 1],
            model,
            blocks,
            mmio=mmio,
            transport=transport,
            use_hw_gemm=use_hw_gemm,
        )
        ok += int(int(np.argmax(logits)) == int(y[i]))
    return ok / max(n, 1)


@dataclass
class PeerResult:
    name: str
    accuracy: float
    params: int
    kib_fp32: float
    kib_int8_est: float
    ms_per_phrase: float
    notes: str = ""


def _save_val_npz(x_va: np.ndarray, y_va: np.ndarray) -> None:
    np.savez_compressed(VAL_NPZ, images=x_va.astype(np.float32), labels=y_va.astype(np.int64))
    print(f"saved val mel → {VAL_NPZ} (n={len(y_va)})", flush=True)


def _load_val_npz() -> tuple[np.ndarray, np.ndarray]:
    if not VAL_NPZ.is_file():
        raise FileNotFoundError(
            f"missing {VAL_NPZ}; run a full train bench once to export val mel"
        )
    z = np.load(VAL_NPZ)
    return z["images"].astype(np.float32), z["labels"].astype(np.int64)


def _save_train_npz(
    x_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray, *, n_per_cmd: int
) -> None:
    np.savez_compressed(
        TRAIN_NPZ,
        images=x_tr.astype(np.float32),
        labels=y_tr.astype(np.int64),
        words=w_tr.astype(np.int64),
        n_per_cmd=np.int64(n_per_cmd),
    )
    print(f"saved train mel → {TRAIN_NPZ} (n={len(y_tr)})", flush=True)


def _load_train_npz(
    n_per_cmd: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not TRAIN_NPZ.is_file():
        return None
    z = np.load(TRAIN_NPZ)
    if int(z["n_per_cmd"]) != int(n_per_cmd):
        print(
            f"train cache n_per_cmd={int(z['n_per_cmd'])} != {n_per_cmd}; rebuilding",
            flush=True,
        )
        return None
    print(f"reuse train mel ← {TRAIN_NPZ} (n={len(z['labels'])})", flush=True)
    return (
        z["images"].astype(np.float32),
        z["labels"].astype(np.int64),
        z["words"].astype(np.int64),
    )


def run(args: argparse.Namespace) -> None:
    configure_paths(long=bool(args.long), fair=bool(args.fair))
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"log-mel: sr={logmel.SR} n_fft={logmel.N_FFT} hop={logmel.HOP} "
        f"(50% overlap) n_mels={logmel.N_MELS} frames={MEL_FRAMES} "
        f"canvas_s={ga.TARGET_SAMPLES_ACTIVE / logmel.SR:.3f} align={ga.ALIGN}"
    )
    print(
        f"commands={ga.N_CMD}  kws_words={ga.N_KWS}  "
        f"token_t={TOKEN_T}  hybrid_stem={HYBRID_STEM}  fair={FAIR_MODE}"
    )

    results: list[PeerResult] = []
    long = bool(args.long)
    fair = bool(args.fair)
    c_note = (
        f"hybrid DS-stem → T={TOKEN_T}×D={clm.LM_D}×L={clm.N_LAYERS} → "
        + ("last-token + word-seq aux" if LAST_TOKEN_HEAD else "mean-pool intent")
    )

    if args.eval_only:
        x_va, y_va = _load_val_npz()
        print(f"eval-only: val={len(y_va)} mel={x_va.shape} (no train / no GSC)")
        if fair:
            a_path = WEIGHTS_DIR / "fat_rf_cnn.pt"
            if not a_path.is_file():
                a_path = WEIGHTS_DIR / "causal_dscnn.pt"
                a = CausalDSCNN()
                a_name, a_note = "A_causal_dscnn", "causal last-frame CNN (eval-only)"
            else:
                a = FatRFCNN()
                a_name = "A_fat_rf_cnn"
                a_note = (
                    f"dilated Fat CNN RF={FatRFCNN.theoretical_receptive_field()} "
                    f"frames; last column; no GAP (eval-only)"
                )
            a.load_state_dict(torch.load(a_path, map_location="cpu"))
        else:
            a = FatDSCNN(long=long)
            a.load_state_dict(torch.load(WEIGHTS_DIR / "fat_dscnn.pt", map_location="cpu"))
            a_name, a_note = "A_fat_dscnn", "bag CNN (eval-only)"
        c = AudioCommandTransformer()
        c.load_state_dict(torch.load(WEIGHTS_DIR / "audio_transformer.pt", map_location="cpu"))
        xb0 = torch.from_numpy(x_va[:1])
        acc_a = accuracy_classifier(a, x_va, y_va)
        ms_a = bench_ms(lambda: a(xb0))
        results.append(
            PeerResult(
                a_name,
                acc_a,
                count_params(a),
                kib_fp32(count_params(a)),
                kib_int8(count_params(a)),
                ms_a,
                a_note,
            )
        )
        if not fair:
            b = KwsCNN()
            b.load_state_dict(torch.load(WEIGHTS_DIR / "kws_cnn.pt", map_location="cpu"))
            acc_b = accuracy_kws_fsm(b, x_va, y_va)
            ms_b = bench_ms(lambda: kws_fsm_predict(xb0, b), n=5 if long else 20)
            results.append(
                PeerResult(
                    "B_kws_fsm",
                    acc_b,
                    count_params(b),
                    kib_fp32(count_params(b)),
                    kib_int8(count_params(b)),
                    ms_b,
                    "sliding-window KWS + FSM (eval-only)",
                )
            )
        acc_c = accuracy_classifier(c, x_va, y_va)
        ms_c = bench_ms(lambda: c(xb0))
        results.append(
            PeerResult(
                "C_transformer_torch",
                acc_c,
                count_params(c),
                kib_fp32(count_params(c)),
                kib_int8(count_params(c)),
                ms_c,
                c_note + " (eval-only)",
            )
        )
    else:
        n_tr = args.n_per_cmd
        n_va = max(8, args.n_per_cmd // 4)
        print("Building datasets (may download GSC ~2GB once)...", flush=True)
        cached = None if getattr(args, "rebuild_data", False) else _load_train_npz(n_tr)
        if cached is not None:
            x_tr, y_tr, w_tr = cached
        else:
            x_tr, y_tr, _, w_tr = ga.build_phrase_dataset(n_tr, seed=0, split="train")
            _save_train_npz(x_tr, y_tr, w_tr, n_per_cmd=n_tr)
        if VAL_NPZ.is_file() and not getattr(args, "rebuild_data", False):
            x_va, y_va = _load_val_npz()
            print(f"reuse val mel ← {VAL_NPZ} (n={len(y_va)})", flush=True)
            w_va = np.zeros((len(y_va), 0), dtype=np.int64)
        else:
            x_va, y_va, _, w_va = ga.build_phrase_dataset(n_va, seed=1, split="val")
            _save_val_npz(x_va, y_va)
        print(f"phrase train={len(y_tr)} val={len(y_va)} mel={x_tr.shape}", flush=True)

        # ---- A ----
        if fair:
            rf = FatRFCNN.theoretical_receptive_field()
            print(
                f"\n=== (A) Fat RF CNN (no GAP; theoretical RF={rf} ≥ {MEL_FRAMES} frames) ==="
            )
            a = FatRFCNN()
            n_a = count_params(a)
            print(
                f"  params={n_a:,}  fp32={kib_fp32(n_a):.1f} KiB  "
                f"int8~={kib_int8(n_a):.1f} KiB"
            )
            train_classifier(
                a, x_tr, y_tr, x_va, y_va, epochs=args.epochs, batch=args.batch, lr=2e-3, tag="A"
            )
            torch.save(a.state_dict(), WEIGHTS_DIR / "fat_rf_cnn.pt")
            a_name = "A_fat_rf_cnn"
            a_note = (
                f"dilated Fat CNN RF={rf} frames; last time column; no GAP "
                f"(params={n_a:,})"
            )
        else:
            print("\n=== (A) Fat DS-CNN ===")
            a = FatDSCNN(long=long)
            train_classifier(
                a, x_tr, y_tr, x_va, y_va, epochs=args.epochs, batch=args.batch, lr=2e-3, tag="A"
            )
            torch.save(a.state_dict(), WEIGHTS_DIR / "fat_dscnn.pt")
            a_name, a_note = (
                "A_fat_dscnn",
                "full-phrase log-mel classifier" + (" (long canvas)" if long else ""),
            )
        acc_a = accuracy_classifier(a, x_va, y_va)
        xb0 = torch.from_numpy(x_va[:1])
        ms_a = bench_ms(lambda: a(xb0))
        n_a = count_params(a)
        print(
            f"  A done: acc={100*acc_a:.1f}%  params={n_a:,}  "
            f"fp32={kib_fp32(n_a):.1f} KiB  int8~={kib_int8(n_a):.1f} KiB  ms={ms_a:.2f}"
        )
        results.append(
            PeerResult(
                a_name,
                acc_a,
                n_a,
                kib_fp32(n_a),
                kib_int8(n_a),
                ms_a,
                a_note,
            )
        )

        # ---- B (skip in fair A-vs-C race) ----
        if not fair:
            print("\n=== (B) KWS + FSM ===")
            kx_tr, ky_tr = ga.build_kws_dataset(args.n_kws, seed=0, split="train")
            kx_va, ky_va = ga.build_kws_dataset(max(16, args.n_kws // 4), seed=1, split="val")
            print(f"kws train={len(ky_tr)} val={len(ky_va)}")
            b = KwsCNN()
            train_classifier(
                b,
                kx_tr,
                ky_tr,
                kx_va,
                ky_va,
                epochs=args.epochs,
                batch=args.batch,
                lr=2e-3,
                tag="B-kws",
            )
            torch.save(b.state_dict(), WEIGHTS_DIR / "kws_cnn.pt")
            acc_b = accuracy_kws_fsm(b, x_va, y_va)
            ms_b = bench_ms(lambda: kws_fsm_predict(xb0, b), n=5 if long else 20)
            results.append(
                PeerResult(
                    "B_kws_fsm",
                    acc_b,
                    count_params(b),
                    kib_fp32(count_params(b)),
                    kib_int8(count_params(b)),
                    ms_b,
                    "sliding-window KWS + longest command match FSM",
                )
            )

        # ---- C ----
        print("\n=== (C) Mel-stem + Transformer ===")
        c = AudioCommandTransformer()
        n_c = count_params(c)
        print(
            f"  params={n_c:,}  fp32={kib_fp32(n_c):.1f} KiB  "
            f"int8~={kib_int8(n_c):.1f} KiB"
        )
        c_batch = max(4, args.batch // (4 if (long or fair) else 2))
        ht_path = WEIGHTS_DIR / "audio_transformer.pt"
        reuse_ht = bool(fair and ht_path.is_file() and not getattr(args, "retrain_ht", False))
        if reuse_ht:
            print(f"  reuse saved HT weights: {ht_path}")
            c.load_state_dict(torch.load(ht_path, map_location="cpu"))
        elif fair:
            train_transformer_sequential(
                c,
                x_tr,
                y_tr,
                w_tr,
                x_va,
                y_va,
                epochs=args.epochs,
                batch=c_batch,
                lr=2e-3,
                tag="C",
            )
            torch.save(c.state_dict(), ht_path)
        else:
            train_classifier(
                c,
                x_tr,
                y_tr,
                x_va,
                y_va,
                epochs=args.epochs,
                batch=c_batch,
                lr=2e-3,
                tag="C",
            )
            torch.save(c.state_dict(), ht_path)
        acc_c = accuracy_classifier(c, x_va, y_va)
        ms_c = bench_ms(lambda: c(xb0))
        print(
            f"  C done: acc={100*acc_c:.1f}%  params={n_c:,}  "
            f"fp32={kib_fp32(n_c):.1f} KiB  int8~={kib_int8(n_c):.1f} KiB  ms={ms_c:.2f}"
        )
        results.append(
            PeerResult(
                "C_transformer_torch",
                acc_c,
                n_c,
                kib_fp32(n_c),
                kib_int8(n_c),
                ms_c,
                c_note + (" (reused weights)" if reuse_ht else " (Torch float GEMM)"),
            )
        )

        # Fair race: also report prior causal CNN if weights exist (params/acc reference).
        if fair:
            causal_path = WEIGHTS_DIR / "causal_dscnn.pt"
            if causal_path.is_file():
                causal = CausalDSCNN()
                causal.load_state_dict(torch.load(causal_path, map_location="cpu"))
                n_ca = count_params(causal)
                acc_ca = accuracy_classifier(causal, x_va, y_va)
                ms_ca = bench_ms(lambda: causal(xb0))
                print(
                    f"  prior causal CNN: acc={100*acc_ca:.1f}%  params={n_ca:,}  "
                    f"fp32={kib_fp32(n_ca):.1f} KiB  int8~={kib_int8(n_ca):.1f} KiB"
                )
                results.append(
                    PeerResult(
                        "A_causal_dscnn_ref",
                        acc_ca,
                        n_ca,
                        kib_fp32(n_ca),
                        kib_int8(n_ca),
                        ms_ca,
                        "prior causal DS-CNN (saved weights; RF≪full canvas)",
                    )
                )

    # ---- C NpuKit int8 (skip for fair / long accuracy races by default) ----
    if not getattr(args, "skip_npukit", False):
        print("\n=== (C) NpuKit int8 GEMM body (CPU tiled) ===")
        blocks_i8 = [_quant_block(b) for b in c.blocks]
        n_np = min(len(y_va), 32 if args.quick or long or fair else 128)
        acc_cpu_i8 = peer_c_npukit_accuracy(c, x_va, y_va, use_hw_gemm=False, max_n=n_np)
        ms_cpu_i8 = bench_ms(
            lambda: peer_c_npukit_forward(x_va[:1], c, blocks_i8, use_hw_gemm=False),
            n=5 if (args.quick or long or fair) else 20,
        )
        results.append(
            PeerResult(
                "C_npukit_cpu_i8",
                acc_cpu_i8,
                count_params(c),
                kib_fp32(count_params(c)),
                kib_int8(count_params(c)),
                ms_cpu_i8,
                f"same weights; int8 tiled GEMM via npukit_transformer (n={n_np})",
            )
        )

        print("\n=== (C) NpuKit int8 GEMM body (FPGA) ===")
        fpga_note = "no bitstream / device"
        acc_fpga = float("nan")
        ms_fpga = float("nan")
        if args.bit:
            try:
                from npukit_matmul import open_device

                mmio, transport = open_device(args.bit)
                acc_fpga = peer_c_npukit_accuracy(
                    c, x_va, y_va, use_hw_gemm=True, mmio=mmio, transport=transport, max_n=n_np
                )
                ms_fpga = bench_ms(
                    lambda: peer_c_npukit_forward(
                        x_va[:1], c, blocks_i8, mmio=mmio, transport=transport, use_hw_gemm=True
                    ),
                    n=5 if (long or fair) else 10,
                )
                fpga_note = f"PL GEMM via {args.bit} (n={n_np})"
                print(f"  FPGA acc={100*acc_fpga:.1f}%  ms={ms_fpga:.2f}")
            except Exception as e:
                fpga_note = f"FPGA open/run failed: {e}"
                print(f"  {fpga_note}")
        else:
            print("  skip (pass --bit /path/to/npukit.bit to enable)")
        results.append(
            PeerResult(
                "C_npukit_fpga_i8",
                acc_fpga if acc_fpga == acc_fpga else 0.0,
                count_params(c),
                kib_fp32(count_params(c)),
                kib_int8(count_params(c)),
                ms_fpga if ms_fpga == ms_fpga else -1.0,
                fpga_note,
            )
        )

    print("\n=== Peer summary ===")
    print(
        f"\n{'peer':24} {'acc%':>8} {'params':>10} {'weights_f32':>11} "
        f"{'weights_i8~':>11} {'ms':>8}"
    )
    for r in results:
        ms_s = f"{r.ms_per_phrase:8.2f}" if r.ms_per_phrase >= 0 else f"{'n/a':>8}"
        print(
            f"{r.name:24} {100*r.accuracy:8.1f} {r.params:10,d} "
            f"{r.kib_fp32:10.1f}KiB {r.kib_int8_est:10.1f}KiB {ms_s}"
        )

    payload = {
        "phrase_mode": ga.PHRASE_MODE,
        "mel": {
            "sr": logmel.SR,
            "n_fft": logmel.N_FFT,
            "hop": logmel.HOP,
            "overlap": 0.5,
            "n_mels": logmel.N_MELS,
            "n_frames": MEL_FRAMES,
            "target_samples": ga.TARGET_SAMPLES_ACTIVE,
        },
        "token_t": TOKEN_T,
        "hybrid_stem": HYBRID_STEM,
        "fat_rf_frames": FatRFCNN.theoretical_receptive_field() if fair else None,
        "n_commands": ga.N_CMD,
        "commands": [" ".join(cmd) for cmd in ga.AUDIO_ROBOT_COMMANDS],
        "peers": [asdict(r) for r in results],
        "bit": args.bit,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--long",
        action="store_true",
        help="9-word phrases on 16.384 s canvas",
    )
    ap.add_argument(
        "--fair",
        action="store_true",
        help="order-only permutations; full-RF Fat CNN (no GAP) vs hybrid transformer",
    )
    ap.add_argument(
        "--retrain-ht",
        action="store_true",
        help="with --fair, retrain HT instead of reusing audio_transformer.pt",
    )
    ap.add_argument(
        "--rebuild-data",
        action="store_true",
        help="rebuild train/val mel caches from GSC (ignore *.npz caches)",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="load saved weights + val mel; skip train/GSC",
    )
    ap.add_argument("--skip-npukit", action="store_true", help="skip int8 CPU/FPGA peer-C rows")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n-per-cmd", type=int, default=None)
    ap.add_argument("--n-kws", type=int, default=None)
    ap.add_argument("--bit", type=str, default=None, help="npukit.bit for peer-C FPGA GEMM")
    args = ap.parse_args()
    heavy = args.long or args.fair
    if args.eval_only:
        args.epochs = args.epochs or 0
        args.n_per_cmd = args.n_per_cmd or 0
        args.n_kws = args.n_kws or 0
        args.batch = args.batch or 32
    elif args.quick:
        args.epochs = args.epochs or (8 if args.fair else (6 if args.long else 4))
        args.n_per_cmd = args.n_per_cmd or (24 if heavy else 40)
        args.n_kws = args.n_kws or (60 if heavy else 80)
        args.batch = args.batch or (8 if heavy else 32)
    else:
        args.epochs = args.epochs or (35 if args.fair else (30 if args.long else 25))
        args.n_per_cmd = args.n_per_cmd or (100 if args.fair else (80 if args.long else 150))
        args.n_kws = args.n_kws or (200 if heavy else 250)
        args.batch = args.batch or (8 if heavy else 32)
    if heavy and args.bit is None:
        args.skip_npukit = True
    run(args)


if __name__ == "__main__":
    main()

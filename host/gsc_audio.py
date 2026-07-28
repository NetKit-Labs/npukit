#!/usr/bin/env python3
"""Load Google Speech Commands WAVs and stitch robot command phrases.

Audio command inventories use **only** GSC v2 words so phrases can be built
by concatenating real clips with short silence.

Modes:
  short — 1–2 word commands on a 2.048 s canvas (legacy peer race)
  long  — 9-word order-sensitive scripts on a 16.384 s canvas
"""

from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

import gsc_commands as gsc
from logmel import (
    SR,
    TARGET_SAMPLES,
    TARGET_SAMPLES_LONG,
    pad_or_trim,
    wav_to_logmel,
)

HOST_DIR = Path(__file__).resolve().parent
GSC_DIR = gsc.GSC_DIR

# GSC-only short robot commands (legacy).
AUDIO_ROBOT_COMMANDS_SHORT: tuple[tuple[str, ...], ...] = (
    ("stop",),
    ("on",),
    ("off",),
    ("up",),
    ("down",),
    ("left",),
    ("right",),
    ("forward",),
    ("backward",),
    ("follow", "forward"),
    ("go", "left"),
    ("go", "right"),
    ("go", "up"),
    ("go", "down"),
    ("go", "forward"),
    ("go", "backward"),
    ("visual", "on"),
    ("visual", "off"),
    ("learn", "left"),
    ("learn", "right"),
    ("yes", "go"),
    ("no", "stop"),
)

# 9-word order-sensitive scripts (minimal pairs differ by word order).
AUDIO_ROBOT_COMMANDS_LONG: tuple[tuple[str, ...], ...] = (
    ("go", "left", "go", "right", "go", "forward", "stop", "on", "off"),
    ("go", "right", "go", "left", "go", "forward", "stop", "on", "off"),
    ("go", "forward", "go", "left", "go", "right", "stop", "on", "off"),
    ("go", "left", "go", "forward", "go", "right", "stop", "on", "off"),
    ("stop", "go", "left", "go", "right", "go", "forward", "on", "off"),
    ("go", "left", "go", "right", "stop", "go", "forward", "on", "off"),
    ("go", "left", "go", "right", "go", "forward", "on", "stop", "off"),
    ("go", "left", "go", "right", "go", "forward", "off", "on", "stop"),
    ("go", "up", "go", "down", "go", "left", "stop", "on", "off"),
    ("go", "down", "go", "up", "go", "left", "stop", "on", "off"),
    ("go", "left", "go", "up", "go", "down", "stop", "on", "off"),
    ("go", "backward", "go", "forward", "go", "left", "stop", "on", "off"),
    ("go", "forward", "go", "backward", "go", "left", "stop", "on", "off"),
    ("follow", "left", "go", "right", "go", "forward", "stop", "on", "off"),
    ("go", "left", "follow", "right", "go", "forward", "stop", "on", "off"),
    ("learn", "left", "go", "right", "go", "forward", "stop", "on", "off"),
    ("go", "left", "learn", "right", "go", "forward", "stop", "on", "off"),
    ("visual", "on", "go", "left", "go", "right", "stop", "off", "learn"),
    ("visual", "off", "go", "left", "go", "right", "stop", "on", "learn"),
    ("yes", "go", "left", "no", "go", "right", "stop", "on", "off"),
    ("no", "go", "left", "yes", "go", "right", "stop", "on", "off"),
    ("go", "left", "yes", "go", "right", "no", "stop", "on", "off"),
)

assert all(8 <= len(c) <= 10 for c in AUDIO_ROBOT_COMMANDS_LONG)

# Order-only: every class is a permutation of the SAME multiset (bag features fail).
_ORDER_BASE = ("go", "left", "right", "forward", "stop", "on", "off", "learn")
AUDIO_ROBOT_COMMANDS_ORDER: tuple[tuple[str, ...], ...] = (
    ("go", "left", "right", "forward", "stop", "on", "off", "learn"),
    ("go", "right", "left", "forward", "stop", "on", "off", "learn"),
    ("left", "go", "right", "forward", "stop", "on", "off", "learn"),
    ("right", "go", "left", "forward", "stop", "on", "off", "learn"),
    ("go", "left", "forward", "right", "stop", "on", "off", "learn"),
    ("go", "forward", "left", "right", "stop", "on", "off", "learn"),
    ("stop", "go", "left", "right", "forward", "on", "off", "learn"),
    ("go", "left", "right", "stop", "forward", "on", "off", "learn"),
    ("go", "left", "right", "forward", "on", "stop", "off", "learn"),
    ("go", "left", "right", "forward", "off", "on", "stop", "learn"),
    ("learn", "go", "left", "right", "forward", "stop", "on", "off"),
    ("go", "learn", "left", "right", "forward", "stop", "on", "off"),
    ("on", "off", "go", "left", "right", "forward", "stop", "learn"),
    ("off", "on", "go", "left", "right", "forward", "stop", "learn"),
    ("forward", "stop", "go", "left", "right", "on", "off", "learn"),
    ("stop", "forward", "go", "left", "right", "on", "off", "learn"),
)


def _multiset(cmd: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    from collections import Counter

    return tuple(sorted(Counter(cmd).items()))


assert all(len(c) == len(_ORDER_BASE) for c in AUDIO_ROBOT_COMMANDS_ORDER)
assert len({_multiset(c) for c in AUDIO_ROBOT_COMMANDS_ORDER}) == 1
assert _multiset(AUDIO_ROBOT_COMMANDS_ORDER[0]) == _multiset(_ORDER_BASE)

# Active mode (mutated by set_phrase_mode).
PHRASE_MODE = "short"
AUDIO_ROBOT_COMMANDS: tuple[tuple[str, ...], ...] = AUDIO_ROBOT_COMMANDS_SHORT
N_CMD = len(AUDIO_ROBOT_COMMANDS)
CMD_TO_ID = {c: i for i, c in enumerate(AUDIO_ROBOT_COMMANDS)}
KWS_WORDS: tuple[str, ...] = tuple(sorted({w for cmd in AUDIO_ROBOT_COMMANDS for w in cmd}))
KWS_TO_ID = {w: i for i, w in enumerate(KWS_WORDS)}
N_KWS = len(KWS_WORDS)
TARGET_SAMPLES_ACTIVE = TARGET_SAMPLES
ALIGN = "left"  # fair mode uses right-align for last-frame / last-token heads

SILENCE_MS = 100  # between stitched words


def set_phrase_mode(mode: str) -> None:
    """Select short, long, or fair (order-only) inventory."""
    global PHRASE_MODE, AUDIO_ROBOT_COMMANDS, N_CMD, CMD_TO_ID
    global KWS_WORDS, KWS_TO_ID, N_KWS, TARGET_SAMPLES_ACTIVE, ALIGN
    mode = mode.lower().strip()
    if mode == "short":
        AUDIO_ROBOT_COMMANDS = AUDIO_ROBOT_COMMANDS_SHORT
        TARGET_SAMPLES_ACTIVE = TARGET_SAMPLES
        ALIGN = "left"
    elif mode == "long":
        AUDIO_ROBOT_COMMANDS = AUDIO_ROBOT_COMMANDS_LONG
        TARGET_SAMPLES_ACTIVE = TARGET_SAMPLES_LONG
        ALIGN = "left"
    elif mode == "fair":
        AUDIO_ROBOT_COMMANDS = AUDIO_ROBOT_COMMANDS_ORDER
        TARGET_SAMPLES_ACTIVE = TARGET_SAMPLES_LONG
        ALIGN = "right"
    else:
        raise ValueError(f"unknown phrase mode {mode!r}; use 'short', 'long', or 'fair'")
    PHRASE_MODE = mode
    N_CMD = len(AUDIO_ROBOT_COMMANDS)
    CMD_TO_ID = {c: i for i, c in enumerate(AUDIO_ROBOT_COMMANDS)}
    KWS_WORDS = tuple(sorted({w for cmd in AUDIO_ROBOT_COMMANDS for w in cmd}))
    KWS_TO_ID = {w: i for i, w in enumerate(KWS_WORDS)}
    N_KWS = len(KWS_WORDS)
    lens = {len(c) for c in AUDIO_ROBOT_COMMANDS}
    print(
        f"phrase_mode={mode} n_cmd={N_CMD} word_lens={sorted(lens)} "
        f"kws_words={N_KWS} canvas_s={TARGET_SAMPLES_ACTIVE / SR:.3f} align={ALIGN}"
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SR
        raw = wf.readframes(wf.getnframes())
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


def ensure_gsc(root: Path | None = None) -> Path:
    root = Path(root or GSC_DIR)
    wavs = gsc.list_gsc_wavs(root)
    need = set(KWS_WORDS)
    have = {w for w, files in wavs.items() if files}
    missing = need - have
    if missing:
        print(f"GSC missing words {sorted(missing)}; downloading...")
        gsc.download_gsc(root)
        wavs = gsc.list_gsc_wavs(root)
        have = {w for w, files in wavs.items() if files}
        missing = need - have
        if missing:
            raise FileNotFoundError(f"GSC still missing: {sorted(missing)}")
    return root


def list_word_wavs(root: Path | None = None) -> dict[str, list[Path]]:
    root = ensure_gsc(root)
    all_wavs = gsc.list_gsc_wavs(root)
    return {w: all_wavs[w] for w in KWS_WORDS}


def stitch_command(
    words: tuple[str, ...] | list[str],
    word_wavs: dict[str, list[Path]],
    rng: random.Random,
    *,
    silence_ms: int = SILENCE_MS,
    target_samples: int | None = None,
) -> np.ndarray:
    gap = np.zeros(int(SR * silence_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, w in enumerate(words):
        files = word_wavs[w]
        path = files[rng.randrange(len(files))]
        parts.append(read_wav(path))
        if i + 1 < len(words):
            parts.append(gap)
    wav = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    n = TARGET_SAMPLES_ACTIVE if target_samples is None else target_samples
    return pad_or_trim(wav, n, align=ALIGN)


def build_phrase_dataset(
    n_per_cmd: int,
    *,
    seed: int = 0,
    root: Path | None = None,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]], np.ndarray]:
    """Return log-mel [N,1,M,T], labels [N], command tuples, word-ids [N, L]."""
    word_wavs = list_word_wavs(root)
    split_wavs: dict[str, list[Path]] = {}
    for w, files in word_wavs.items():
        train, val = [], []
        for p in files:
            h = hash(p.name) & 0xFF
            (val if h < 26 else train).append(p)
        pool = val if split == "val" else train
        if not pool:
            pool = files
        split_wavs[w] = pool

    rng = random.Random(seed + (1 if split == "val" else 0))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    cmds: list[tuple[str, ...]] = []
    word_rows: list[list[int]] = []
    n_tgt = TARGET_SAMPLES_ACTIVE
    max_l = max(len(c) for c in AUDIO_ROBOT_COMMANDS)
    for cid, cmd in enumerate(AUDIO_ROBOT_COMMANDS):
        wids = [KWS_TO_ID[w] for w in cmd]
        # pad word-id rows to max_l with -1
        wpad = wids + [-1] * (max_l - len(wids))
        for _ in range(n_per_cmd):
            wav = stitch_command(cmd, split_wavs, rng, target_samples=n_tgt)
            xs.append(wav_to_logmel(wav, target_samples=n_tgt))
            ys.append(cid)
            cmds.append(cmd)
            word_rows.append(wpad)
    x = np.stack(xs, axis=0)[:, None, :, :].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    warr = np.asarray(word_rows, dtype=np.int64)
    order = list(range(len(y)))
    rng.shuffle(order)
    x = x[order]
    y = y[order]
    warr = warr[order]
    cmds = [cmds[i] for i in order]
    return x, y, cmds, warr


def build_kws_dataset(
    n_per_word: int,
    *,
    seed: int = 0,
    root: Path | None = None,
    split: str = "train",
    win: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-word log-mel windows for peer B KWS (short canvas crops)."""
    from logmel import N_FRAMES

    word_wavs = list_word_wavs(root)
    rng = random.Random(seed + (7 if split == "val" else 0))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    # Word clips stay on the short canvas; speech is left-aligned.
    active = min(64, N_FRAMES)
    for w, wid in KWS_TO_ID.items():
        files = word_wavs[w]
        train, val = [], []
        for p in files:
            h = hash(p.name) & 0xFF
            (val if h < 26 else train).append(p)
        pool = val if split == "val" else train
        if not pool:
            pool = files
        for _ in range(n_per_word):
            path = pool[rng.randrange(len(pool))]
            wav = pad_or_trim(read_wav(path), TARGET_SAMPLES)
            mel = wav_to_logmel(wav, target_samples=TARGET_SAMPLES)
            max_start = max(0, active - win)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            crop = mel[:, start : start + win]
            if crop.shape[1] < win:
                pad = np.zeros((mel.shape[0], win - crop.shape[1]), dtype=np.float32)
                crop = np.concatenate([crop, pad], axis=1)
            xs.append(crop)
            ys.append(wid)
    x = np.stack(xs, axis=0)[:, None, :, :].astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    order = list(range(len(y)))
    rng.shuffle(order)
    return x[order], y[order]

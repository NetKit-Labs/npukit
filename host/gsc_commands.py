#!/usr/bin/env python3
"""Google Speech Commands vocabulary + robot command phrases for the tiny LM.

Speech Commands v2 label set (35 words) is the token vocabulary. Multi-word
robot phrases are a **prefix-free** command list built from those words so a
causal LM can actually learn (no ``go left`` vs ``go left stop`` collision).

Optional: download the official GSC v0.02 tarball to ``host/data/speech_commands``
for audio keyword spotting experiments (same word list).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import tarfile
import urllib.request
from pathlib import Path

HOST_DIR = Path(__file__).resolve().parent
GSC_DIR = HOST_DIR / "data" / "speech_commands"
GSC_URL = (
    "https://storage.cloud.google.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)

# Official Speech Commands v2 words (35). Order is stable for token ids.
GSC_WORDS: tuple[str, ...] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)

SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")

# Prefix-free robot command inventory (no command is a proper prefix of another).
# Bare singles that would collide with multi-word forms are omitted (e.g. no bare "go").
ROBOT_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("stop",),
    ("on",),
    ("off",),
    ("up",),
    ("down",),
    ("left",),
    ("right",),
    ("forward",),
    ("backward",),
    ("follow", "me"),
    ("go", "left"),
    ("go", "right"),
    ("go", "up"),
    ("go", "down"),
    ("go", "forward"),
    ("go", "backward"),
    ("move", "left"),
    ("move", "right"),
    ("move", "forward"),
    ("move", "backward"),
    ("turn", "left"),
    ("turn", "right"),
    ("turn", "on"),
    ("turn", "off"),
    ("visual", "on"),
    ("visual", "off"),
    ("learn", "left"),
    ("learn", "right"),
    ("yes", "go"),
    ("no", "stop"),
)


def _assert_prefix_free(commands: tuple[tuple[str, ...], ...]) -> None:
    for i, a in enumerate(commands):
        for j, b in enumerate(commands):
            if i == j:
                continue
            if len(a) < len(b) and b[: len(a)] == a:
                raise ValueError(f"command {a} is a prefix of {b}")


_assert_prefix_free(ROBOT_COMMANDS)


def build_vocab() -> tuple[list[str], dict[str, int]]:
    words = list(SPECIAL) + list(GSC_WORDS)
    for extra in ("move", "turn", "me"):
        if extra not in words:
            words.append(extra)
    stoi = {w: i for i, w in enumerate(words)}
    return words, stoi


VOCAB, STOI = build_vocab()
PAD_ID, BOS_ID, EOS_ID, UNK_ID = (STOI[s] for s in SPECIAL)
VOCAB_SIZE = len(VOCAB)


def encode_words(words: list[str], *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    ids: list[int] = []
    if add_bos:
        ids.append(BOS_ID)
    for w in words:
        ids.append(STOI.get(w, UNK_ID))
    if add_eos:
        ids.append(EOS_ID)
    return ids


def pad_to_t(ids: list[int], t: int) -> list[int]:
    if len(ids) > t:
        ids = ids[:t]
        if ids[-1] != EOS_ID:
            ids[-1] = EOS_ID
    return ids + [PAD_ID] * (t - len(ids))


def generate_phrases(
    n: int,
    *,
    seed: int = 0,
    include_unigrams: bool = True,
) -> list[list[str]]:
    """Sample robot commands (with replacement) from the prefix-free inventory."""
    del include_unigrams  # inventory already includes unigrams
    rng = random.Random(seed)
    return [list(rng.choice(ROBOT_COMMANDS)) for _ in range(n)]


def phrases_to_tensors(
    phrases: list[list[str]],
    t: int,
    *,
    seed: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (input_ids, target_ids) for causal LM: predict token t+1."""
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for ph in phrases:
        ids = encode_words(ph)
        if len(ids) < 2:
            continue
        xs.append(pad_to_t(ids[:-1], t))
        ys.append(pad_to_t(ids[1:], t))
    rng = random.Random(seed)
    order = list(range(len(xs)))
    rng.shuffle(order)
    return [xs[i] for i in order], [ys[i] for i in order]


def download_gsc(dest: Path | None = None, *, force: bool = False) -> Path:
    """Download + extract Speech Commands v0.02 (large ~2GB)."""
    dest = Path(dest or GSC_DIR)
    marker = dest / ".extracted"
    if marker.exists() and not force:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "speech_commands_v0.02.tar.gz"
    if force or not tar_path.exists():
        print(f"Downloading {GSC_URL} → {tar_path} (large)...")
        urllib.request.urlretrieve(GSC_URL, tar_path)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def list_gsc_wavs(root: Path | None = None) -> dict[str, list[Path]]:
    """Map label → wav paths when the dataset is present."""
    root = Path(root or GSC_DIR)
    out: dict[str, list[Path]] = {}
    if not root.exists():
        return out
    for w in GSC_WORDS:
        d = root / w
        if d.is_dir():
            out[w] = sorted(d.glob("*.wav"))
    return out


def dataset_fingerprint(phrases: list[list[str]]) -> str:
    h = hashlib.sha1()
    for p in phrases:
        h.update(" ".join(p).encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="Fetch GSC v0.02 tarball")
    ap.add_argument("--demo", type=int, default=8, help="Print N sample phrases")
    args = ap.parse_args()
    if args.download:
        download_gsc()
        wavs = list_gsc_wavs()
        n = sum(len(v) for v in wavs.values())
        print(f"GSC labels with wavs: {len(wavs)}  files: {n}")
    print(f"vocab_size={VOCAB_SIZE}  commands={len(ROBOT_COMMANDS)}")
    phrases = generate_phrases(args.demo, seed=1)
    for p in phrases:
        print(" ", " ".join(p), "→", encode_words(p))


if __name__ == "__main__":
    main()

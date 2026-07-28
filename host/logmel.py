#!/usr/bin/env python3
"""Shared log-mel front-end for speech peers (no MFCC).

Power-of-two STFT with **50% overlap** and no FFT padding:
  sr       = 16000
  n_fft    = 512   (= 2^9)
  hop      = 256   (= n_fft/2 → 50% overlap)
  n_mels   = 32    (= 2^5)

Clip lengths (exact STFT tile → frames = 2^k − 1):
  short: samples = 32768  (= 2^15 → 2.048 s) → 127 frames
  long:  samples = 262144 (= 2^18 → 16.384 s) → 1023 frames
"""

from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 512
HOP = 256  # 50% overlap
N_MELS = 32
TARGET_SAMPLES = 32768  # 2^15 short canvas (default)
TARGET_SAMPLES_LONG = 262144  # 2^18 for 8–10 word phrases
FMIN = 20.0
FMAX = 7600.0

assert N_FFT > 0 and (N_FFT & (N_FFT - 1)) == 0, "n_fft must be power of 2"
assert HOP == N_FFT // 2, "hop must be n_fft/2 for 50% overlap"
assert TARGET_SAMPLES > 0 and (TARGET_SAMPLES & (TARGET_SAMPLES - 1)) == 0
assert TARGET_SAMPLES_LONG > 0 and (TARGET_SAMPLES_LONG & (TARGET_SAMPLES_LONG - 1)) == 0
assert (TARGET_SAMPLES - N_FFT) % HOP == 0, "clip length must tile STFT exactly"
assert (TARGET_SAMPLES_LONG - N_FFT) % HOP == 0
N_FRAMES = 1 + (TARGET_SAMPLES - N_FFT) // HOP  # 127
N_FRAMES_LONG = 1 + (TARGET_SAMPLES_LONG - N_FFT) // HOP  # 1023
assert N_FRAMES == 127
assert N_FRAMES_LONG == 1023


def _mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sr: int = SR,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Triangular mel filterbank [n_mels, n_fft//2+1]."""
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(np.int64)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)
        for j in range(left, center):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i, j] = (right - j) / max(right - center, 1)
    enorm = 2.0 / np.maximum(f_pts[2 : n_mels + 2] - f_pts[:n_mels], 1e-8)
    fb *= enorm[:, None]
    return fb


_MEL_FB: np.ndarray | None = None


def mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank()
    return _MEL_FB


def pad_or_trim(
    wav: np.ndarray,
    n: int = TARGET_SAMPLES,
    *,
    align: str = "left",
) -> np.ndarray:
    """Fit clip to exact pow2 sample count (silence pad only at clip level).

    ``align='right'`` puts speech at the end (for causal / last-token heads).
    """
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size >= n:
        return x[-n:] if align == "right" else x[:n]
    out = np.zeros(n, dtype=np.float32)
    if align == "right":
        out[-x.size :] = x
    else:
        out[: x.size] = x
    return out


def stft_mag(wav: np.ndarray, *, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """|STFT| with hann window → [n_frames, n_fft//2+1].

    Requires ``len(wav) >= n_fft`` and ``(len - n_fft) % hop == 0`` so every
    frame is a full ``n_fft`` window (no zero-pad inside the FFT).
    """
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    if x.size < n_fft:
        raise ValueError(f"wav length {x.size} < n_fft {n_fft}")
    if (x.size - n_fft) % hop != 0:
        raise ValueError(
            f"wav length {x.size} does not tile STFT (n_fft={n_fft}, hop={hop})"
        )
    win = np.hanning(n_fft).astype(np.float64)
    n_frames = 1 + (x.size - n_fft) // hop
    # Framing via as_strided + one batched rfft (fast path for long canvases).
    step = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft), strides=(hop * step, step)
    )
    return np.abs(np.fft.rfft(frames * win, n=n_fft, axis=-1))


def n_frames_for(target_samples: int) -> int:
    return 1 + (int(target_samples) - N_FFT) // HOP


def wav_to_logmel(
    wav: np.ndarray,
    *,
    eps: float = 1e-6,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Waveform → log-mel [n_mels, n_frames] float32 (exact STFT tiling)."""
    x = pad_or_trim(wav, target_samples)
    mag = stft_mag(x)
    assert mag.shape[0] == n_frames_for(target_samples)
    mel = mag @ mel_fb().T
    logmel = np.log(mel + eps).astype(np.float32)
    feat = logmel.T.copy()
    mu = float(feat.mean())
    sd = float(feat.std()) + 1e-5
    feat = (feat - mu) / sd
    return feat.astype(np.float32)


def mel_shape(target_samples: int = TARGET_SAMPLES) -> tuple[int, int]:
    return N_MELS, n_frames_for(target_samples)

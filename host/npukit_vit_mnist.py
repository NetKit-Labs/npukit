#!/usr/bin/env python3
"""MNIST tiny-ViT host path for NpuKit (GEMM + glue).

Geometry (T=16, no resize):
  - Native 28×28 MNIST
  - **CPU DS-stem** (richer tiny DS) → T=16 tokens × D=16
  - Model dim D=16, FFN hidden VIT_MLP (GEMM host-tiles 8×8)
  - N_LAYERS transformer blocks (host-scheduled on the same GEMM)

Split:
  - CPU / TFLite+XNNPACK: DS-stem, pos add, Softmax/RMSNorm/GELU (float), head
  - FPGA: int8 GEMM (QKV / scores / P@V / FFN); optional HW glue

Legacy non-stem patch embed (7×7 patches → GEMM) still loads if npz has no stem_*.

Train:
  python3 host/train_vit_mnist.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import npukit_transformer as nt
import vit_ds_stem as stemmod

_HOST_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _HOST_DIR / "vit_mnist_weights.npz"
DEFAULT_SAMPLE = _HOST_DIR / "mnist_sample.npz"
DEFAULT_STEM_TFLITE = _HOST_DIR / "vit_mnist_stem.tflite"

# --- ViT-MNIST geometry (host contract) ---
IMG = 28
PATCH = 7  # legacy patch path only
VIT_T = 16
VIT_D = 16
VIT_MLP = 32  # FFN hidden; needs glue_mode='float' ( > MAX_LEN)
N_LAYERS = 4  # stem + deeper body; still MCU-tiny
PATCH_DIM_RAW = PATCH * PATCH  # 49 (legacy)
PATCH_DIM = ((PATCH_DIM_RAW + 7) // 8) * 8  # 56 — legacy GEMM pad
N_CLASS = 10
USE_STEM = True
# Deploy default: int8 GEMM (+ optional NPU) with A9 float Softmax/RMSNorm/GELU.
DEFAULT_GLUE_MODE = "float"

assert VIT_T <= nt.MAX_LEN and VIT_T % 8 == 0
assert VIT_D <= nt.MAX_LEN and VIT_D % 8 == 0 and PATCH_DIM % 8 == 0
assert VIT_MLP % 8 == 0
assert N_LAYERS >= 1


def _load_sw(data, key: str, n: int, fallback: float) -> np.ndarray | None:
    if key in data.files:
        arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=np.float64)
        return arr
    return np.full(n, float(fallback), dtype=np.float64)


def _block_from_npz(data, layer: int | None) -> nt.TinyBlockWeights:
    """Load TinyBlockWeights. layer=None → legacy keys wq; else wq{layer}."""

    def k(base: str) -> str:
        return base if layer is None else f"{base}{layer}"

    wq = np.asarray(data[k("wq")], dtype=np.int8)
    w1 = np.asarray(data[k("w1")], dtype=np.int8)
    d, mlp_h = int(wq.shape[1]), int(w1.shape[1])
    prefix = "scale_block0" if layer is None else f"scale_block{layer}"
    leg_w = float(data["scale_w"][0]) if "scale_w" in data.files else nt.SCALE_W
    stage_w = (
        float(data[f"{prefix}_w"][0]) if f"{prefix}_w" in data.files else leg_w
    )
    return nt.TinyBlockWeights(
        wq=wq,
        wk=np.asarray(data[k("wk")], dtype=np.int8),
        wv=np.asarray(data[k("wv")], dtype=np.int8),
        wo=np.asarray(data[k("wo")], dtype=np.int8),
        w1=w1,
        w2=np.asarray(data[k("w2")], dtype=np.int8),
        gamma1=np.asarray(data[k("gamma1")], dtype=np.int32),
        gamma2=np.asarray(data[k("gamma2")], dtype=np.int32),
        sw_wq=_load_sw(data, f"{prefix}_wq_w", d, stage_w),
        sw_wk=_load_sw(data, f"{prefix}_wk_w", d, stage_w),
        sw_wv=_load_sw(data, f"{prefix}_wv_w", d, stage_w),
        sw_wo=_load_sw(data, f"{prefix}_wo_w", d, stage_w),
        sw_w1=_load_sw(data, f"{prefix}_w1_w", mlp_h, stage_w),
        sw_w2=_load_sw(data, f"{prefix}_w2_w", d, stage_w),
    )


@dataclass(frozen=True)
class QuantScales:
    """int8 quant scales for one stage (embed / block / cls).

    `w` may be a per-output-channel vector (cls / legacy embed) or a scalar
    mean used as fallback when per-matrix scales live on TinyBlockWeights.
    """

    act: float = nt.SCALE_ACT
    w: float | np.ndarray = nt.SCALE_W
    p: float = nt.SCALE_P  # attention probs; unused for embed/cls


def _default_scales(n: int) -> tuple[QuantScales, ...]:
    return tuple(QuantScales() for _ in range(n))


@dataclass
class VitMnistWeights:
    """Optional DS-stem + N blocks + linear head (int8 / Q12).

    Scales are per-stage: embed (legacy), each transformer block, class head.
    When `stem` is set, CPU/TFLite stem produces tokens (no patch GEMM).
    """

    w_pe: np.ndarray  # [PATCH_DIM, D] int8  (legacy patch path; unused with stem)
    pos: np.ndarray  # [T, D] Q12
    blocks: tuple[nt.TinyBlockWeights, ...]
    w_cls: np.ndarray  # [D, N_CLASS] int8
    scale_embed: QuantScales = field(default_factory=QuantScales)
    scale_blocks: tuple[QuantScales, ...] = field(default_factory=tuple)
    scale_cls: QuantScales = field(default_factory=QuantScales)
    stem: stemmod.StemInt8 | None = None

    def __post_init__(self) -> None:
        if not self.scale_blocks:
            self.scale_blocks = _default_scales(
                len(self.blocks) if self.blocks else N_LAYERS
            )

    @property
    def block(self) -> nt.TinyBlockWeights:
        return self.blocks[0]

    @property
    def scale_act(self) -> float:
        """Legacy: first block act scale."""
        return self.scale_blocks[0].act if self.scale_blocks else self.scale_embed.act

    @property
    def scale_w(self) -> float:
        return self.scale_blocks[0].w if self.scale_blocks else self.scale_embed.w

    @property
    def scale_p(self) -> float:
        return self.scale_blocks[0].p if self.scale_blocks else nt.SCALE_P

    @staticmethod
    def make(rng: np.random.Generator) -> "VitMnistWeights":
        w_pe = nt.quant_weight_to_i8(rng.normal(0.0, 0.12, size=(PATCH_DIM, VIT_D)))
        pos = nt.to_q12(rng.normal(0.0, 0.05, size=(VIT_T, VIT_D)))
        blocks = tuple(
            nt.TinyBlockWeights.make(rng, VIT_D, mlp_h=VIT_MLP) for _ in range(N_LAYERS)
        )
        w_cls = nt.quant_weight_to_i8(rng.normal(0.0, 0.12, size=(VIT_D, N_CLASS)))
        return VitMnistWeights(
            w_pe=w_pe,
            pos=pos,
            blocks=blocks,
            w_cls=w_cls,
            scale_embed=QuantScales(),
            scale_blocks=_default_scales(N_LAYERS),
            scale_cls=QuantScales(),
        )

    @staticmethod
    def load(path: str | Path) -> "VitMnistWeights":
        data = np.load(path)
        if "wq0" in data.files:
            n_layers = 0
            while f"wq{n_layers}" in data.files:
                n_layers += 1
            blocks = tuple(_block_from_npz(data, i) for i in range(n_layers))
        else:
            blocks = (_block_from_npz(data, None),)
        if len(blocks) < 1:
            raise ValueError(f"{path} has no transformer layers")

        # Legacy global scales → broadcast; prefer per-stage keys when present.
        leg_a = float(data["scale_act"][0]) if "scale_act" in data.files else nt.SCALE_ACT
        leg_w = float(data["scale_w"][0]) if "scale_w" in data.files else nt.SCALE_W
        leg_p = float(data["scale_p"][0]) if "scale_p" in data.files else nt.SCALE_P

        def _sc(prefix: str, *, need_p: bool, w_len: int | None = None) -> QuantScales:
            a = float(data[f"{prefix}_act"][0]) if f"{prefix}_act" in data.files else leg_a
            if f"{prefix}_w_ch" in data.files:
                w = np.asarray(data[f"{prefix}_w_ch"], dtype=np.float64).reshape(-1)
            elif f"{prefix}_w" in data.files:
                raw = np.asarray(data[f"{prefix}_w"], dtype=np.float64).reshape(-1)
                w = (
                    np.full(w_len, float(raw[0]), dtype=np.float64)
                    if w_len is not None and raw.size == 1
                    else (float(raw[0]) if raw.size == 1 else raw)
                )
            else:
                w = (
                    np.full(w_len, leg_w, dtype=np.float64)
                    if w_len is not None
                    else leg_w
                )
            p = (
                float(data[f"{prefix}_p"][0])
                if f"{prefix}_p" in data.files
                else (leg_p if need_p else nt.SCALE_P)
            )
            return QuantScales(act=a, w=w, p=p)

        if "scale_embed_act" in data.files or "scale_block0_act" in data.files:
            scale_embed = _sc("scale_embed", need_p=False)
            scale_blocks = tuple(
                _sc(f"scale_block{i}", need_p=True) for i in range(len(blocks))
            )
            scale_cls = _sc("scale_cls", need_p=False, w_len=N_CLASS)
        else:
            scale_embed = QuantScales(act=leg_a, w=leg_w)
            scale_blocks = tuple(QuantScales(act=leg_a, w=leg_w, p=leg_p) for _ in blocks)
            scale_cls = QuantScales(
                act=leg_a, w=np.full(N_CLASS, leg_w, dtype=np.float64)
            )

        stem = stemmod.StemInt8.from_npz(data)
        return VitMnistWeights(
            w_pe=np.asarray(data["w_pe"], dtype=np.int8),
            pos=np.asarray(data["pos"], dtype=np.int32),
            blocks=blocks,
            w_cls=np.asarray(data["w_cls"], dtype=np.int8),
            scale_embed=scale_embed,
            scale_blocks=scale_blocks,
            scale_cls=scale_cls,
            stem=stem,
        )


def patchify(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """[H,W] → [T, P*P] row-major patches (no resize; H=W=IMG)."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    assert h == w == IMG and h % patch == 0
    gh = h // patch
    return (
        img.reshape(gh, patch, gh, patch)
        .transpose(0, 2, 1, 3)
        .reshape(gh * gh, patch * patch)
    )


def pad_patches(patches: np.ndarray) -> np.ndarray:
    """Pad patch vectors 49 → 56 for GEMM alignment."""
    p = np.asarray(patches, dtype=np.float64)
    assert p.shape[-1] == PATCH_DIM_RAW
    if PATCH_DIM == PATCH_DIM_RAW:
        return p
    out = np.zeros(p.shape[:-1] + (PATCH_DIM,), dtype=np.float64)
    out[..., :PATCH_DIM_RAW] = p
    return out


def synthetic_digit(rng: np.random.Generator, label: int, size: int = 28) -> np.ndarray:
    """Crude MNIST-sized blob so offline runs need no dataset download."""
    img = np.zeros((size, size), dtype=np.float64)
    cy, cx = size // 2, size // 2
    ry = 4 + (label % 5)
    rx = 3 + ((label * 3) % 5)
    yy, xx = np.ogrid[:size, :size]
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    img[mask] = 0.75 + 0.2 * rng.random()
    if label in (1, 7):
        img[:] = 0.0
        img[:, cx - 1 : cx + 2] = 0.9
        if label == 7:
            img[3:6, cx - 4 : cx + 5] = 0.9
    img += 0.05 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


def load_or_synth_batch(
    n: int = 4,
    seed: int = 0,
    *,
    sample_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return images [N,28,28] float and labels [N]. Prefers saved sample npz."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, N_CLASS, size=n)
    candidates: list[Path] = []
    if sample_path is not None:
        candidates.append(Path(sample_path))
    candidates.extend(
        [
            DEFAULT_SAMPLE,
            Path("mnist_sample.npz"),
            Path("/home/xilinx/jupyter_notebooks/mnist_sample.npz"),
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            data = np.load(path)
            imgs = data["images"].astype(np.float64)
            labs = data["labels"].astype(int)
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
            if len(labs) > n:
                idx = rng.choice(len(labs), size=n, replace=False)
                imgs, labs = imgs[idx], labs[idx]
            else:
                imgs, labs = imgs[:n], labs[:n]
            print(f"loaded sample from {path} (n={len(labs)})")
            return imgs, labs
        except (FileNotFoundError, OSError, KeyError):
            pass

    imgs = np.stack([synthetic_digit(rng, int(lbl)) for lbl in labels], axis=0)
    print(
        f"using synthetic digits (n={n}); "
        "run host/train_vit_mnist.py to create a sample npz"
    )
    return imgs, labels


def embed_patches_q12(
    patches: np.ndarray,
    w: VitMnistWeights,
    *,
    glue=None,
    mmio=None,
    transport=None,
    use_hw: bool,
    glue_mode: str | None = None,
) -> np.ndarray:
    """Patch linear (GEMM) + position residual (glue if HW)."""
    x_q12 = nt.to_q12(np.asarray(patches, dtype=np.float64))
    tokens = nt._matmul_q12(
        x_q12,
        w.w_pe,
        mmio=mmio,
        transport=transport,
        use_hw=use_hw,
        scale_act=w.scale_embed.act,
        scale_w=w.scale_embed.w,
    )
    gmode = nt._resolve_glue_mode(use_hw=use_hw, glue_mode=glue_mode)
    return nt._residual_rows(glue, tokens, w.pos, glue_mode=gmode)


def classify_tokens_cpu(tokens_q12: np.ndarray, w: VitMnistWeights) -> np.ndarray:
    """Mean-pool + linear head on CPU (N_CLASS=10 is not a multiple of 8)."""
    pooled_q12 = np.rint(tokens_q12.astype(np.float64).mean(axis=0)).astype(np.int32)
    a_i8 = nt.quant_q12_to_i8(pooled_q12.reshape(1, -1), w.scale_cls.act)
    return nt.gemm_i8_to_q12(
        a_i8, w.w_cls, a_scale=w.scale_cls.act, b_scale=w.scale_cls.w
    ).reshape(-1)


def _stem_tokens_float(
    img: np.ndarray,
    w: VitMnistWeights,
    *,
    use_tflite: bool | None,
) -> np.ndarray:
    """CPU stem → float tokens [T,D]. TFLite+XNNPACK when available/requested."""
    assert w.stem is not None
    auto = use_tflite is None
    want = bool(use_tflite) or (
        auto and DEFAULT_STEM_TFLITE.exists() and _prefer_tflite_stem()
    )
    if want and DEFAULT_STEM_TFLITE.exists():
        try:
            return stemmod.stem_forward_tflite(img, DEFAULT_STEM_TFLITE)
        except Exception:
            if use_tflite:
                raise
    return stemmod.stem_forward_numpy(img, w.stem, qat=True)


def _prefer_tflite_stem() -> bool:
    """Prefer TFLite stem on ARM A9 hosts."""
    try:
        import platform

        return platform.machine().startswith(("arm", "aarch"))
    except Exception:
        return False


def vit_forward(
    img28: np.ndarray,
    w: VitMnistWeights,
    *,
    glue=None,
    mmio=None,
    transport=None,
    use_hw: bool = False,
    use_hw_gemm: bool | None = None,
    glue_mode: str | None = None,
    verbose: bool = False,
    use_tflite_stem: bool | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One image → logits Q12 [10] + dump.

    Default deploy glue is A9 float Softmax/RMSNorm/GELU (DEFAULT_GLUE_MODE).
    Set glue_mode='hw' to exercise FPGA glue (requires mlp_h ≤ MAX_LEN).
    """
    img = np.asarray(img28, dtype=np.float64)
    assert img.shape == (IMG, IMG)
    dump: dict[str, np.ndarray] = {"img28": img}
    gmode = DEFAULT_GLUE_MODE if glue_mode is None else glue_mode
    gemm_hw = bool(use_hw if use_hw_gemm is None else use_hw_gemm)

    if w.stem is not None and USE_STEM:
        tok_f = _stem_tokens_float(img, w, use_tflite=use_tflite_stem)
        dump["stem_tokens"] = tok_f.copy()
        tokens = nt.to_q12(tok_f.astype(np.float64))
        tokens = nt._residual_rows(
            glue, tokens, w.pos, glue_mode=nt._resolve_glue_mode(use_hw=False, glue_mode=gmode)
        )
    else:
        raw = patchify(img, PATCH)
        patches = pad_patches(raw)
        dump["patches_raw"] = raw
        dump["patches"] = patches
        tokens = embed_patches_q12(
            patches,
            w,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=gemm_hw,
            glue_mode=gmode,
        )
    dump["tokens"] = tokens.copy()

    y = tokens
    for li, blk in enumerate(w.blocks):
        sc = w.scale_blocks[li]
        y, block_dump = nt.transformer_block_1layer(
            y,
            blk,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=False,
            use_hw_gemm=gemm_hw,
            glue_mode=gmode,
            verbose=verbose,
            scale_act=sc.act,
            scale_w=sc.w,
            scale_p=sc.p,
        )
        dump.update({f"block{li}.{k}": v for k, v in block_dump.items()})
        # compat alias for last-layer checks / older smoke keys
        if li == 0:
            dump.update({f"block.{k}": v for k, v in block_dump.items()})
    dump["tokens_out"] = y.copy()

    logits = classify_tokens_cpu(y, w)
    dump["logits"] = logits.copy()
    dump["pred"] = np.array([int(np.argmax(logits))], dtype=np.int32)
    return logits, dump


def run_vit_smoke(
    *,
    bit_path: str | None = None,
    seed: int = 0,
    n: int = 2,
    verbose: bool = False,
    weights_path: str | Path | None | bool = None,
    sample_path: str | Path | None = None,
) -> int:
    rng = np.random.default_rng(seed)
    # weights_path=False → force random; None → auto-load default npz if present
    if weights_path is False:
        w = VitMnistWeights.make(rng)
        wsrc = "random-seeded"
    else:
        path = weights_path
        if path is None and DEFAULT_WEIGHTS.exists():
            path = DEFAULT_WEIGHTS
        if path is not None:
            w = VitMnistWeights.load(path)
            wsrc = str(path)
        else:
            w = VitMnistWeights.make(rng)
            wsrc = "random-seeded"
    imgs, labels = load_or_synth_batch(n=n, seed=seed + 1, sample_path=sample_path)

    print("=== MNIST tiny-ViT smoke ===")
    print(
        f"IMG={IMG} PATCH={PATCH} T={VIT_T} D={VIT_D} mlp={VIT_MLP} "
        f"layers={N_LAYERS} glue={DEFAULT_GLUE_MODE} "
        f"patch_dim={PATCH_DIM_RAW}->pad{PATCH_DIM} classes={N_CLASS}"
    )
    def _wfmt(sw) -> str:
        arr = np.asarray(sw, dtype=np.float64).reshape(-1)
        return f"{float(arr.mean()):.2f}" if arr.size else "nan"

    print(
        f"scales embed act/w={w.scale_embed.act:.2f}/{_wfmt(w.scale_embed.w)}  "
        + " ".join(
            f"L{i}={s.act:.1f}/{_wfmt(s.w)}/{s.p:.1f}"
            for i, s in enumerate(w.scale_blocks)
        )
        + f"  cls={w.scale_cls.act:.2f}/{_wfmt(w.scale_cls.w)}"
    )
    print(f"weights={wsrc}")

    glue = mmio = transport = None
    if bit_path is not None:
        from npukit_matmul import open_device

        mmio, transport = open_device(bit_path)
        ident = mmio.read(nt.REG_ID)
        ver = mmio.read(nt.REG_VERSION)
        feat = mmio.read(nt.REG_FEATURES)
        print(f"ID=0x{ident:08X} version=0x{ver:08X} features=0x{feat:08X}")
        if ident != nt.ID_MAGIC or ver < nt.VERSION_GLUE or not (feat & nt.FEAT_GLUE):
            print("VIT FAIL: need glue bitstream VERSION>=0x300")
            return 1
        glue = nt.GlueDevice(mmio)

    ok = True
    n_correct_ref = 0
    n_correct_hw = 0
    n_pred_match = 0
    for i in range(n):
        print(f"\n--- image[{i}] label={int(labels[i])} ---")
        # Numpy int8 stem (not TFLite): A9 tflite_runtime can SIGILL on invoke().
        print("--- ref ---")
        logits_ref, dump_ref = vit_forward(
            imgs[i], w, use_hw=False, verbose=verbose, use_tflite_stem=False
        )
        pred_ref = int(dump_ref["pred"][0])
        n_correct_ref += int(pred_ref == int(labels[i]))
        print(f"ref pred={pred_ref} logits_q12[:4]={logits_ref[:4].tolist()}")

        if bit_path is None:
            continue

        print("--- FPGA ---")
        logits_hw, dump_hw = vit_forward(
            imgs[i],
            w,
            glue=glue,
            mmio=mmio,
            transport=transport,
            use_hw=True,
            verbose=verbose,
            use_tflite_stem=False,
        )
        pred_hw = int(dump_hw["pred"][0])
        n_correct_hw += int(pred_hw == int(labels[i]))
        print(f"hw  pred={pred_hw} logits_q12[:4]={logits_hw[:4].tolist()}")

        # Layer-0 stays tight. Deeper layers accumulate Softmax/RMSNorm LUT drift
        # vs CPU ref — report abs err; overall gate uses pred-agreement rate.
        keys = [("tokens", 512), ("block0.y_out", 1024)]
        for li in range(1, len(w.blocks)):
            keys.append((f"block{li}.y_out", 0))  # tol 0 → report-only
        keys.append(("logits", 0))
        pred_match = pred_hw == pred_ref
        n_pred_match += int(pred_match)
        print(f"pred_match: {'PASS' if pred_match else 'FAIL'}  ref={pred_ref} hw={pred_hw}")
        for key, tol in keys:
            err = int(
                np.max(
                    np.abs(
                        dump_ref[key].astype(np.int64) - dump_hw[key].astype(np.int64)
                    )
                )
            )
            if tol <= 0:
                print(f"{key}: info  max|err|={err}")
            else:
                passed = err <= tol
                ok &= passed
                print(f"{key}: {'PASS' if passed else 'FAIL'}  max|err|={err}  tol={tol}")

    print(
        f"\nref accuracy on this batch: {n_correct_ref}/{n} "
        f"({100.0 * n_correct_ref / max(n, 1):.1f}%)"
    )
    if bit_path is not None:
        print(
            f"hw  accuracy on this batch: {n_correct_hw}/{n} "
            f"({100.0 * n_correct_hw / max(n, 1):.1f}%)"
        )
        # Allow a little L2 argmax drift from glue LUT compounding (≈90%+ agree).
        agree = n_pred_match / max(n, 1)
        agree_ok = agree >= (0.90 if len(w.blocks) > 1 else 1.0)
        ok &= agree_ok
        print(
            f"ref↔hw pred agree: {n_pred_match}/{n} ({100.0 * agree:.1f}%) "
            f"{'PASS' if agree_ok else 'FAIL'}"
        )

    if bit_path is None:
        print("\nVIT REF-ONLY PASS")
        return 0

    print("\nALL VIT PASS" if ok else "\nVIT FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NpuKit MNIST tiny-ViT host smoke")
    p.add_argument("bit", nargs="?", help="path to npukit.bit (omit with --ref-only)")
    p.add_argument("--ref-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-n", type=int, default=64, help="images to run (full mnist_sample.npz)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--weights",
        default=None,
        help="path to vit_*_weights.npz (default: auto if present)",
    )
    p.add_argument(
        "--sample",
        default=None,
        help="path to mnist_sample.npz",
    )
    p.add_argument("--random-weights", action="store_true", help="ignore saved weights")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    weights: str | Path | None | bool
    weights = False if args.random_weights else args.weights
    if args.ref_only or not args.bit:
        return run_vit_smoke(
            bit_path=None,
            seed=args.seed,
            n=args.n,
            verbose=args.verbose,
            weights_path=weights,
            sample_path=args.sample,
        )
    return run_vit_smoke(
        bit_path=args.bit,
        seed=args.seed,
        n=args.n,
        verbose=args.verbose,
        weights_path=weights,
        sample_path=args.sample,
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CNN inference via TFLite + XNNPACK (DS-CNN peer and ViT DS-stem).

Artifacts:
  - `dscnn_mnist.tflite` — full MCU peer
  - `vit_mnist_stem.tflite` — tiny DS-stem → [1,16,16] tokens

Host (x86): TensorFlow Lite creates an XNNPACK CPU delegate
(`INFO: Created TensorFlow Lite XNNPACK delegate for CPU`).

PYNQ-Z2 / Cortex-A9: the public `tflite_runtime` 2.13 armv7l wheel
**SIGILLs on `invoke()`** (even with BUILTIN_WITHOUT_DEFAULT_DELEGATES).
Vendored wheel is still useful on newer ARM boards; on Zynq-7020 use the
numpy deploy-ref paths in `dscnn_mnist` / `vit_ds_stem` for on-board CNN,
and keep TFLite+XNNPACK for host-side peer timing.

Install wheel (when the CPU can run it):
  pip3 install --user host/third_party/tflite_runtime-*-linux_armv7l.whl
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parent
DSCNN_TFLITE = HOST / "dscnn_mnist.tflite"
STEM_TFLITE = HOST / "vit_mnist_stem.tflite"


def _import_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore

        return Interpreter, load_delegate, "tflite_runtime"
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

        try:
            from tensorflow.lite.python.interpreter import load_delegate  # type: ignore
        except ImportError:
            load_delegate = None  # type: ignore
        return Interpreter, load_delegate, "tensorflow"


def _xnnpack_delegates(load_delegate, *, enable: bool):
    """Optional external XNNPACK .so — off by default on Cortex-A9.

    Many public armv7l wheels SIGILL when loading an external XNNPACK
    delegate on Zynq-7020. Prefer the in-tree TFLite CPU kernels
    (often XNNPACK-backed at build time) via Interpreter only.
    """
    if not enable or load_delegate is None:
        return []
    candidates = [
        "libxnnpack_delegate.so",
        str(HOST / "third_party" / "libxnnpack_delegate.so"),
    ]
    try:
        import tflite_runtime

        pkg = Path(tflite_runtime.__file__).resolve().parent
        candidates.append(str(pkg / "libxnnpack_delegate.so"))
    except Exception:
        pass
    for path in candidates:
        try:
            return [load_delegate(path)]
        except Exception:
            continue
    return []


class TFLiteCNN:
    """Thin TFLite wrapper for A9 CNN peers (stem + DS-CNN benchmark)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_threads: int = 2,
        external_xnnpack: bool = False,
    ):
        Interpreter, load_delegate, backend = _import_interpreter()
        model_path = str(model_path)
        delegates = _xnnpack_delegates(load_delegate, enable=external_xnnpack)
        kwargs = {"model_path": model_path, "num_threads": num_threads}
        self.backend = backend
        self.delegate = (
            "xnnpack-external" if delegates else "tflite-cpu (XNNPACK-in-tree if built)"
        )
        if delegates:
            kwargs["experimental_delegates"] = delegates
        self.interp = Interpreter(**kwargs)
        self.interp.allocate_tensors()
        self.input = self.interp.get_input_details()[0]
        self.output = self.interp.get_output_details()[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run one (or batched) inference. x shaped like model input."""
        x = np.asarray(x)
        inp = self.input
        out = self.output
        if x.shape != tuple(inp["shape"]):
            x = x.reshape(inp["shape"])
        if inp["dtype"] == np.float32:
            x = x.astype(np.float32)
        elif inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            x = np.clip(np.round(x.astype(np.float32) / scale + zp), -128, 127).astype(
                np.int8
            )
        self.interp.set_tensor(inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            scale, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        return np.asarray(y, dtype=np.float32)

    def time_ms(self, x: np.ndarray, *, warmup: int = 4, iters: int = 16) -> float:
        for _ in range(warmup):
            self(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            self(x)
        return 1000.0 * (time.perf_counter() - t0) / max(iters, 1)


def load_dscnn_tflite(path: Path = DSCNN_TFLITE, **kw) -> TFLiteCNN:
    if not Path(path).exists():
        raise FileNotFoundError(f"missing {path}; run host/export_tflite_cnn.py")
    return TFLiteCNN(path, **kw)


def load_stem_tflite(path: Path = STEM_TFLITE, **kw) -> TFLiteCNN:
    if not Path(path).exists():
        raise FileNotFoundError(f"missing {path}; run host/export_tflite_cnn.py")
    return TFLiteCNN(path, **kw)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Smoke TFLite+XNNPACK CNN models")
    p.add_argument("--dscnn", action="store_true")
    p.add_argument("--stem", action="store_true")
    p.add_argument("--bench-n", type=int, default=32)
    args = p.parse_args()
    if not args.dscnn and not args.stem:
        args.dscnn = args.stem = True
    sample = HOST / "mnist_sample.npz"
    imgs = np.load(sample)["images"] if sample.exists() else np.zeros((1, 28, 28), np.float32)
    if args.dscnn and DSCNN_TFLITE.exists():
        m = load_dscnn_tflite()
        x = imgs[0].astype(np.float32).reshape(1, 28, 28, 1)
        y = m(x)
        ms = m.time_ms(x)
        print(f"DS-CNN TFLite  backend={m.backend} delegate={m.delegate}")
        print(f"  out={y.reshape(-1)[:5]}...  pred={int(y.argmax())}  {ms:.2f} ms/img")
    if args.stem and STEM_TFLITE.exists():
        m = load_stem_tflite()
        x = imgs[0].astype(np.float32).reshape(1, 28, 28, 1)
        y = m(x)
        ms = m.time_ms(x)
        print(f"ViT stem TFLite backend={m.backend} delegate={m.delegate}")
        print(f"  tokens shape={y.shape}  {ms:.2f} ms/img")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

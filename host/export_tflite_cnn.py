#!/usr/bin/env python3
"""Export DS-CNN peer + ViT DS-stem to TFLite for A9 XNNPACK inference.

Produces:
  host/dscnn_mnist.tflite       — full MCU peer (NHWC float32)
  host/vit_mnist_stem.tflite    — tiny DS-stem → [1,16,16] tokens (NHWC float32)

Requires TensorFlow on the build host (not on the PYNQ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

import dscnn_mnist as dscnn
import train_vit_mnist as tv
import vit_ds_stem as stem

HOST = Path(__file__).resolve().parent
DSCNN_TFLITE = HOST / "dscnn_mnist.tflite"
STEM_TFLITE = HOST / "vit_mnist_stem.tflite"


def _keras_conv(cin, cout, k, stride, padding, groups=1, name=None):
    pad = "same" if padding else "valid"
    if groups == 1:
        return tf.keras.layers.Conv2D(
            cout, k, strides=stride, padding=pad, use_bias=True, name=name
        )
    # Depthwise
    return tf.keras.layers.DepthwiseConv2D(
        k, strides=stride, padding=pad, use_bias=True, depth_multiplier=1, name=name
    )


def export_dscnn_tflite() -> Path:
    """Float FoldedDSCNN → Keras → TFLite float32 (XNNPACK-friendly)."""
    device = torch.device("cpu")
    if dscnn.INT8_PATH.exists():
        # Prefer BN-folded float weights from int8 export (dequant)
        pt = dscnn.load_int8(dscnn.INT8_PATH, device)
        pt.eval()
        # run once to materialize
    else:
        pt = dscnn.fold_dscnn(dscnn.load_model(dscnn.WEIGHTS_PATH, device))

    inp = tf.keras.Input(shape=(28, 28, 1), name="image")
    x = inp
    # mirror FoldedDSCNN layer order
    layers_spec = [
        ("stem", 1, 16, 3, 1, 1, 1),
        ("b1_dw", 16, 16, 3, 2, 1, 16),
        ("b1_pw", 16, 32, 1, 1, 0, 1),
        ("b2_dw", 32, 32, 3, 2, 1, 32),
        ("b2_pw", 32, 64, 1, 1, 0, 1),
        ("b3_dw", 64, 64, 3, 1, 1, 64),
        ("b3_pw", 64, 64, 1, 1, 0, 1),
    ]
    # Build with random then set weights
    built = []
    for name, cin, cout, k, stride, pad, groups in layers_spec:
        if groups == cin and cin == cout:
            layer = tf.keras.layers.DepthwiseConv2D(
                k, strides=stride, padding="same" if pad else "valid", use_bias=True, name=name
            )
        else:
            layer = tf.keras.layers.Conv2D(
                cout, k, strides=stride, padding="same" if pad else "valid", use_bias=True, name=name
            )
        x = tf.nn.relu(layer(x))
        built.append((name, layer, groups))
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    fc = tf.keras.layers.Dense(10, name="fc")
    logits = fc(x)
    km = tf.keras.Model(inp, logits, name="dscnn_mnist")

    # Copy weights from torch FoldedDSCNN (OIHW → HWIO / depthwise)
    with torch.no_grad():
        for name, layer, groups in built:
            mod = getattr(pt, name)
            w = mod.weight.detach().cpu().numpy()
            b = mod.bias.detach().cpu().numpy()
            if groups > 1:
                # torch DW: [C,1,k,k] → TF depthwise [k,k,C,1]
                w_tf = np.transpose(w, (2, 3, 0, 1))
            else:
                # torch [Co,Ci,k,k] → TF [k,k,Ci,Co]
                w_tf = np.transpose(w, (2, 3, 1, 0))
            layer.set_weights([w_tf, b])
        w = pt.fc.weight.detach().cpu().numpy().T  # [in, out]
        b = pt.fc.bias.detach().cpu().numpy()
        fc.set_weights([w, b])

    converter = tf.lite.TFLiteConverter.from_keras_model(km)
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    buf = converter.convert()
    DSCNN_TFLITE.write_bytes(buf)
    print(f"wrote {DSCNN_TFLITE} ({len(buf)} bytes)")
    return DSCNN_TFLITE


def export_stem_tflite(
    model: tv.TinyViT | None = None,
    *,
    out_path: Path | None = None,
) -> Path:
    """Tiny DS-stem → tokens [1,T,D] as TFLite float32."""
    if model is None:
        model = tv.TinyViT()
        if tv.WEIGHTS_PATH.exists():
            # may fail if old weights without stem — then random stem ok for structure
            try:
                tv.load_weights_into_model(model, tv.WEIGHTS_PATH, torch.device("cpu"))
            except Exception as exc:
                print(f"warn: could not load vit weights for stem export ({exc})")
    model.eval()
    st = model.stem

    mid, c = stem.STEM_MID, stem.STEM_C
    inp = tf.keras.Input(shape=(28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(mid, 3, strides=2, padding="same", use_bias=True, name="stem")(inp)
    x = tf.nn.relu(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=True, name="dw")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(mid, 1, strides=1, padding="valid", use_bias=True, name="pw")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=True, name="dw2")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(mid, 1, strides=1, padding="valid", use_bias=True, name="pw2")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=True, name="dw3")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(c, 1, strides=1, padding="valid", use_bias=True, name="pw3")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name="pad8")(x)  # 7→8
    x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="valid", name="pool")(x)
    # 4x4xC → [T,D]
    x = tf.keras.layers.Reshape((stem.STEM_T, stem.STEM_C), name="tokens")(x)
    km = tf.keras.Model(inp, x, name="vit_ds_stem")

    with torch.no_grad():
        for name, conv in (
            ("stem", st.stem),
            ("dw", st.dw),
            ("pw", st.pw),
            ("dw2", st.dw2),
            ("pw2", st.pw2),
            ("dw3", st.dw3),
            ("pw3", st.pw3),
        ):
            w = conv.weight.detach().cpu().numpy()
            b = conv.bias.detach().cpu().numpy()
            if "dw" in name:
                km.get_layer(name).set_weights([np.transpose(w, (2, 3, 0, 1)), b])
            else:
                km.get_layer(name).set_weights([np.transpose(w, (2, 3, 1, 0)), b])

    # Verify vs torch
    img = np.random.randn(1, 28, 28, 1).astype(np.float32) * 0.1 + 0.5
    yt = st(torch.from_numpy(img.transpose(0, 3, 1, 2))).detach().numpy()
    yk = km.predict(img, verbose=0)
    err = float(np.max(np.abs(yt - yk)))
    print(f"stem keras vs torch max|err|={err:.6f}")

    converter = tf.lite.TFLiteConverter.from_keras_model(km)
    converter.optimizations = []
    buf = converter.convert()
    dest = Path(out_path) if out_path is not None else STEM_TFLITE
    dest.write_bytes(buf)
    print(f"wrote {dest} ({len(buf)} bytes)")
    return dest


def main() -> int:
    export_dscnn_tflite()
    # Stem export needs trained TinyViT with stem — export structure even if untrained
    export_stem_tflite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

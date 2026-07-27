Open these in https://netron.app (Open Model… / drag-and-drop).

dscnn_mnist_peer.onnx|.tflite  — MCU DS-CNN benchmark (CPU-only)
npukit_vit_system.onnx         — CPU DS-stem + FPGA transformer + CPU head

In the ViT graph, node prefixes CPU_* vs FPGA_* show the deploy split.

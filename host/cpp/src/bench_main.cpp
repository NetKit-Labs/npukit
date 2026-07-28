// NpuKit C++ driver bench / smoke.
//
// Prerequisites on PYNQ:
//   1) Load bitstream once (Python Overlay or fpgautil)
//   2) sudo ./npukit_bench [--mmio] [--tiles N]
//
// Times tiled GEMM without Python in the kick loop (DMA buffer pool via libcma).

#include "npukit/cpu_ops.hpp"
#include "npukit/device.hpp"
#include "npukit/stem.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

int main(int argc, char** argv) {
  bool use_dma = true;
  int vit_like_tiles = 320;  // ~L=4 tiny-ViT body tile kicks
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mmio")) use_dma = false;
    if (!std::strcmp(argv[i], "--tiles") && i + 1 < argc) vit_like_tiles = std::atoi(argv[++i]);
  }

  try {
    npukit::Device dev(use_dma);
    std::printf("NpuKit C++ driver\n");
    std::printf("  ID=0x%08X VERSION=0x%08X FEATURES=0x%08X DMA=%s WS=%d PP=%d WMEM=%d\n",
                dev.id(), dev.version(), dev.features(), dev.dma_backend_name(),
                dev.weight_stationary() ? 1 : 0, dev.ping_pong() ? 1 : 0,
                dev.weight_bank() ? 1 : 0);


    // Correctness: 8x8 identity-ish
    {
      std::vector<int8_t> a(64), b(64, 0);
      std::vector<int32_t> c(64);
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) a[i * 8 + j] = int8_t(i + 1);
        b[i * 8 + i] = 1;
      }
      dev.matmul_i8(a.data(), b.data(), c.data(), 8, 8, 8);
      bool ok = true;
      for (int i = 0; i < 64; ++i)
        if (c[i] != int32_t(a[i])) ok = false;  // A @ I == A
      std::printf("  GEMM 8x8 A@I: %s\n", ok ? "PASS" : "FAIL");
      if (!ok) return 1;
    }

    // Warmup
    {
      std::vector<int8_t> a(256, 1), b(256, 1);
      std::vector<int32_t> c(256);
      for (int i = 0; i < 4; ++i) dev.matmul_i8(a.data(), b.data(), c.data(), 16, 16, 16);
    }

    // Steady 16x16x16 matmul (8 tiles)
    {
      std::vector<int8_t> a(256), b(256);
      std::vector<int32_t> c(256);
      std::mt19937 rng(0);
      std::uniform_int_distribution<int> dist(-8, 8);
      for (auto& v : a) v = int8_t(dist(rng));
      for (auto& v : b) v = int8_t(dist(rng));
      const int iters = 32;
      auto t0 = clock_type::now();
      for (int i = 0; i < iters; ++i) dev.matmul_i8(a.data(), b.data(), c.data(), 16, 16, 16);
      double ms = ms_since(t0) / iters;
      std::printf("  matmul 16x16x16:     %7.2f ms  (%d HW tiles)\n", ms, 8);
    }

    // Approximate ViT body kick load: vit_like_tiles single 8x8 kicks via 8x8 matmul loop
    {
      std::vector<int8_t> a(64, 2), b(64, 3);
      std::vector<int32_t> c(64);
      auto t0 = clock_type::now();
      for (int i = 0; i < vit_like_tiles; ++i) {
        dev.matmul_i8(a.data(), b.data(), c.data(), 8, 8, 8);
      }
      double ms = ms_since(t0);
      std::printf("  %d x GEMM 8x8 kicks: %7.2f ms  (%.2f ms/kick)\n", vit_like_tiles, ms,
                  ms / vit_like_tiles);
      std::printf("  implied ViT-body-like: ~%.0f ms/img (GEMM only, no stem/norms)\n", ms);
    }

    // Glue smoke if present
    if (dev.features() & npukit::FEAT_GLUE) {
      int32_t x[8], y[8], out[8];
      for (int i = 0; i < 8; ++i) {
        x[i] = npukit::to_q12(0.25f * float(i));
        y[i] = npukit::to_q12(0.1f);
      }
      dev.glue_run(npukit::OP_RESIDUAL, x, 8, out, y);
      bool ok = true;
      for (int i = 0; i < 8; ++i)
        if (out[i] != x[i] + y[i]) ok = false;
      std::printf("  glue residual: %s\n", ok ? "PASS" : "FAIL");

      // Hybrid demo: HW softmax + float GELU
      int32_t scores[16], probs[16], h[32], h2[32];
      for (int i = 0; i < 16; ++i) scores[i] = npukit::to_q12(float(i) * 0.1f);
      dev.glue_run(npukit::OP_SOFTMAX, scores, 16, probs);
      for (int i = 0; i < 32; ++i) h[i] = npukit::to_q12(float(i - 16) * 0.05f);
      npukit::float_gelu_row(h, 32, h2);
      std::printf("  hybrid glue: HW Softmax(len=16) + float GELU(len=32) OK\n");
    }

    // Stem microbench (synthetic weights — measures CPU path only)
    {
      npukit::StemWeights sw;
      sw.mid = npukit::STEM_MID;
      sw.c = npukit::STEM_C;
      auto fill = [](std::vector<float>& v, size_t n, float s) {
        v.assign(n, s);
      };
      fill(sw.w_stem, size_t(sw.mid) * 9, 0.01f);
      fill(sw.b_stem, sw.mid, 0.f);
      fill(sw.w_dw, size_t(sw.mid) * 9, 0.01f);
      fill(sw.b_dw, sw.mid, 0.f);
      fill(sw.w_pw, size_t(sw.mid) * sw.mid, 0.01f);
      fill(sw.b_pw, sw.mid, 0.f);
      fill(sw.w_dw2, size_t(sw.mid) * 9, 0.01f);
      fill(sw.b_dw2, sw.mid, 0.f);
      fill(sw.w_pw2, size_t(sw.mid) * sw.mid, 0.01f);
      fill(sw.b_pw2, sw.mid, 0.f);
      fill(sw.w_dw3, size_t(sw.mid) * 9, 0.01f);
      fill(sw.b_dw3, sw.mid, 0.f);
      fill(sw.w_pw3, size_t(sw.c) * sw.mid, 0.01f);
      fill(sw.b_pw3, sw.c, 0.f);
      sw.sa_in = sw.sa_stem = sw.sa_dw = sw.sa_pw = sw.sa_dw2 = sw.sa_dw3 = sw.sa_pw2 =
          sw.sa_pw3 = 32.f;

      std::vector<float> img(28 * 28, 0.5f), tok(16 * 16);
      for (int i = 0; i < 4; ++i) npukit::stem_forward(img.data(), sw, tok.data(), true);
      auto t0 = clock_type::now();
      const int n = 32;
      for (int i = 0; i < n; ++i) npukit::stem_forward(img.data(), sw, tok.data(), true);
      std::printf("  stem (C, synth w):   %7.2f ms/img\n", ms_since(t0) / n);
    }

    std::printf("DONE\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
}

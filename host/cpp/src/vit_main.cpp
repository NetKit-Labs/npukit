// Tiny-ViT end-to-end on NpuKit (C++).
//
//   python3 export_vit_bin.py
//   make npukit_vit
//   sudo ./npukit_vit --weights vit_mnist.bin [--mmio] [--cpu] [--hybrid] [--iters N]

#include "npukit/device.hpp"
#include "npukit/vit.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

static int argmax(const int32_t* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[best]) best = i;
  return best;
}

int main(int argc, char** argv) {
  bool use_dma = true;
  bool cpu_only = false;
  npukit::GlueMode glue = npukit::GlueMode::Float;
  std::string weights = "vit_mnist.bin";
  int iters = 16;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mmio")) use_dma = false;
    else if (!std::strcmp(argv[i], "--cpu")) cpu_only = true;
    else if (!std::strcmp(argv[i], "--hybrid")) glue = npukit::GlueMode::Hybrid;
    else if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--weights FILE] [--cpu|--mmio] [--hybrid] [--iters N]\n", argv[0]);
      return 0;
    }
  }

  try {
    auto w = npukit::load_vit_bin(weights);
    std::printf("ViT e2e  T=%d D=%d MLP=%d L=%d mid=%d  samples=%zu  glue=%s\n", w.t, w.d,
                w.mlp, w.layers, w.mid, w.samples.size(),
                glue == npukit::GlueMode::Hybrid ? "hybrid" : "float");

    std::unique_ptr<npukit::Device> dev;
    npukit::Device* dptr = nullptr;
    if (!cpu_only) {
      dev = std::make_unique<npukit::Device>(use_dma);
      dptr = dev.get();
      std::printf("  ID=0x%08X VERSION=0x%08X FEATURES=0x%08X DMA=%s WS=%d PP=%d\n",
                  dptr->id(), dptr->version(), dptr->features(), dptr->dma_backend_name(),
                  dptr->weight_stationary() ? 1 : 0, dptr->ping_pong() ? 1 : 0);

    } else {
      std::printf("  GEMM=CPU (no /dev/mem)\n");
    }

    // Parity vs embedded Python reference logits
    int match = 0, label_ok = 0;
    int max_abs = 0;
    for (size_t i = 0; i < w.samples.size(); ++i) {
      const auto& sm = w.samples[i];
      std::vector<int32_t> logits(w.n_class);
      npukit::vit_forward(dptr, w, sm.img.data(), logits.data(), glue);
      int mad = 0;
      for (int c = 0; c < w.n_class; ++c) {
        int e = std::abs(logits[c] - sm.ref_logits_q12[c]);
        if (e > mad) mad = e;
      }
      if (mad > max_abs) max_abs = mad;
      const int pred = argmax(logits.data(), w.n_class);
      const int refp = argmax(sm.ref_logits_q12.data(), w.n_class);
      if (pred == refp) ++match;
      if (pred == sm.label) ++label_ok;
      std::printf("  sample[%zu] label=%d pred=%d ref=%d max|dQ12|=%d\n", i, sm.label, pred,
                  refp, mad);
    }
    if (!w.samples.empty()) {
      std::printf("  agree_argmax %d/%zu  label_ok %d/%zu  max|err|=%d\n", match,
                  w.samples.size(), label_ok, w.samples.size(), max_abs);
    }

    // Latency
    std::vector<float> img(size_t(w.img) * w.img, 0.5f);
    if (!w.samples.empty()) img = w.samples[0].img;
    std::vector<int32_t> logits(w.n_class);
    for (int i = 0; i < 2; ++i) npukit::vit_forward(dptr, w, img.data(), logits.data(), glue);
    auto t0 = clock_type::now();
    for (int i = 0; i < iters; ++i)
      npukit::vit_forward(dptr, w, img.data(), logits.data(), glue);
    const double ms = ms_since(t0) / iters;
    std::printf("  e2e latency: %.2f ms/img  (iters=%d)\n", ms, iters);
    std::printf("DONE\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
}

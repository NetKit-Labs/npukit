// DS-CNN int8 peer (CPU) — fair runtime vs C++ tiny-ViT.
//
//   python3 export_dscnn_bin.py
//   make HOST=1 npukit_dscnn   # or make npukit_dscnn on PYNQ
//   ./npukit_dscnn --weights dscnn_mnist.bin

#include "npukit/dscnn.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

static int argmax(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[best]) best = i;
  return best;
}

int main(int argc, char** argv) {
  std::string weights = "dscnn_mnist.bin";
  int iters = 32;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
  }

  try {
    auto w = npukit::load_dscnn_bin(weights);
    std::printf("DS-CNN int8 peer (C++)  samples=%zu\n", w.samples.size());

    int match = 0, label_ok = 0;
    float max_abs = 0.f;
    for (size_t i = 0; i < w.samples.size(); ++i) {
      const auto& sm = w.samples[i];
      float logits[10];
      npukit::dscnn_forward(w, sm.img.data(), logits);
      float mad = 0.f;
      for (int c = 0; c < 10; ++c) {
        float e = std::fabs(logits[c] - sm.ref_logits[c]);
        if (e > mad) mad = e;
      }
      if (mad > max_abs) max_abs = mad;
      const int pred = argmax(logits, 10);
      const int refp = argmax(sm.ref_logits.data(), 10);
      if (pred == refp) ++match;
      if (pred == sm.label) ++label_ok;
      std::printf("  sample[%zu] label=%d pred=%d ref=%d max|d|=%.3g\n", i, sm.label, pred,
                  refp, mad);
    }
    if (!w.samples.empty()) {
      std::printf("  agree_argmax %d/%zu  label_ok %d/%zu  max|err|=%.3g\n", match,
                  w.samples.size(), label_ok, w.samples.size(), max_abs);
    }

    std::vector<float> img(size_t(w.img) * w.img, 0.5f);
    if (!w.samples.empty()) img = w.samples[0].img;
    float logits[10];
    for (int i = 0; i < 2; ++i) npukit::dscnn_forward(w, img.data(), logits);
    auto t0 = clock_type::now();
    for (int i = 0; i < iters; ++i) npukit::dscnn_forward(w, img.data(), logits);
    std::printf("  e2e latency: %.2f ms/img  (iters=%d)\n", ms_since(t0) / iters, iters);
    std::printf("DONE\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
}

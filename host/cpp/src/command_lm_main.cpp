#include "npukit/command_lm.hpp"
#include "npukit/device.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  bool use_dma = true;
  bool cpu_only = false;
  npukit::GlueMode glue = npukit::GlueMode::Float;
  std::string weights = "command_lm.bin";
  int iters = 16;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mmio"))
      use_dma = false;
    else if (!std::strcmp(argv[i], "--cpu"))
      cpu_only = true;
    else if (!std::strcmp(argv[i], "--hybrid"))
      glue = npukit::GlueMode::Hybrid;
    else if (!std::strcmp(argv[i], "--weights") && i + 1 < argc)
      weights = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--weights FILE] [--cpu|--mmio] [--hybrid] [--iters N]\n", argv[0]);
      return 0;
    }
  }

  try {
    auto w = npukit::load_command_lm_bin(weights);
    std::printf("command_lm T=%d D=%d MLP=%d L=%d V=%d samples=%zu\n", w.t, w.d, w.mlp,
                w.layers, w.vocab, w.samples.size());

    std::unique_ptr<npukit::Device> dev;
    npukit::Device* dptr = nullptr;
    if (!cpu_only) {
      dev = std::make_unique<npukit::Device>(use_dma);
      dptr = dev.get();
      std::printf("  ID=0x%08X VERSION=0x%08X DMA=%s\n", dptr->id(), dptr->version(),
                  dptr->dma_backend_name());
    } else {
      std::printf("  GEMM=CPU\n");
    }

    int correct = 0, total = 0;
    std::vector<int32_t> logits(size_t(w.t) * w.vocab);
    for (const auto& sm : w.samples) {
      npukit::command_lm_forward(dptr, w, sm.input_ids.data(), logits.data(), glue);
      for (int i = 0; i < w.t; ++i) {
        if (sm.target_ids[size_t(i)] == w.pad_id) continue;
        int best = 0;
        int32_t bestv = logits[size_t(i) * w.vocab];
        for (int v = 1; v < w.vocab; ++v) {
          const int32_t val = logits[size_t(i) * w.vocab + v];
          if (val > bestv) {
            bestv = val;
            best = v;
          }
        }
        if (best == sm.target_ids[size_t(i)]) ++correct;
        ++total;
      }
    }
    if (total > 0) {
      std::printf("sample next-token acc: %d/%d (%.1f%%)\n", correct, total,
                  100.0 * double(correct) / double(total));
    }

    if (!w.samples.empty()) {
      const auto& sm = w.samples[0];
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        npukit::command_lm_forward(dptr, w, sm.input_ids.data(), logits.data(), glue);
      }
      auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / double(iters);
      std::printf("latency: %.2f ms/fwd (%d iters)\n", ms, iters);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

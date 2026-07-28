#pragma once

#include "npukit/cpu_ops.hpp"
#include "npukit/device.hpp"
#include "npukit/stem.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace npukit {

struct VitBlockWeights {
  std::vector<int8_t> wq, wk, wv, wo, w1, w2;  // row-major
  std::vector<int32_t> gamma1, gamma2;         // Q12 [D]
  double scale_act{64.0};
  double scale_p{127.0};
  // Per-output-channel *inverse* weight scales for dequant.
  std::vector<double> inv_sw_wq, inv_sw_wk, inv_sw_wv, inv_sw_wo, inv_sw_w1, inv_sw_w2;
};

struct VitSample {
  std::vector<float> img;   // [28*28]
  int32_t label{0};
  std::vector<int32_t> ref_logits_q12;  // [N_CLASS] from Python float-glue CPU
};

struct VitWeights {
  int t{16}, d{16}, mlp{32}, layers{4}, n_class{10}, mid{24}, c{16}, img{28};
  StemWeights stem;
  std::vector<int32_t> pos;  // Q12 [T*D]
  std::vector<VitBlockWeights> blocks;
  std::vector<int8_t> w_cls;  // [D*N_CLASS]
  double scale_cls_act{64.0};
  std::vector<double> inv_sw_cls;  // [N_CLASS]
  std::vector<VitSample> samples;
};

VitWeights load_vit_bin(const std::string& path);

enum class GlueMode { Float, Hybrid };

// End-to-end: 28x28 float image → logits Q12 [n_class].
// If dev == nullptr, uses CPU int8 GEMM (host bring-up / parity checks).
void vit_forward(Device* dev, const VitWeights& w, const float* img28,
                 int32_t* logits_q12, GlueMode glue = GlueMode::Float);

}  // namespace npukit

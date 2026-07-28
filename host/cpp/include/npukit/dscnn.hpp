#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npukit {

struct DscnnSample {
  std::vector<float> img;       // [28*28]
  int32_t label{0};
  std::vector<float> ref_logits;  // [10] from Python int8 numpy path
};

struct DscnnWeights {
  int img{28};
  int n_class{10};
  // Input act scales: sa_in, sa_stem, sa_b1_dw, sa_b1_pw, sa_b2_dw, sa_b2_pw, sa_b3_dw, sa_b3_pw
  float sa[8]{};
  // Dequantized float weights (NCHW)
  std::vector<float> w_stem, b_stem;      // [16,1,3,3], [16]
  std::vector<float> w_b1_dw, b_b1_dw;    // [16,1,3,3], [16]
  std::vector<float> w_b1_pw, b_b1_pw;    // [32,16,1,1], [32]
  std::vector<float> w_b2_dw, b_b2_dw;    // [32,1,3,3], [32]
  std::vector<float> w_b2_pw, b_b2_pw;    // [64,32,1,1], [64]
  std::vector<float> w_b3_dw, b_b3_dw;    // [64,1,3,3], [64]
  std::vector<float> w_b3_pw, b_b3_pw;    // [64,64,1,1], [64]
  std::vector<float> w_fc, b_fc;          // [64,10], [10]
  std::vector<DscnnSample> samples;
};

DscnnWeights load_dscnn_bin(const std::string& path);

// 28x28 float [0,1] → logits float[10] (fake-int8 style, matches Python peer).
void dscnn_forward(const DscnnWeights& w, const float* img28, float* logits);

}  // namespace npukit

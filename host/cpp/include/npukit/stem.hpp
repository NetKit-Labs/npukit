#pragma once

#include <cstdint>
#include <vector>

namespace npukit {

constexpr int STEM_C = 16;
constexpr int STEM_MID = 24;
constexpr int STEM_T = 16;
constexpr int STEM_IMG = 28;

struct StemWeights {
  // float dequantized weights ready for inference
  std::vector<float> w_stem;  // [MID,1,3,3]
  std::vector<float> b_stem;
  std::vector<float> w_dw, b_dw, w_pw, b_pw;
  std::vector<float> w_dw2, b_dw2, w_pw2, b_pw2;
  std::vector<float> w_dw3, b_dw3, w_pw3, b_pw3;
  float sa_in{}, sa_stem{}, sa_dw{}, sa_pw{}, sa_dw2{}, sa_pw2{}, sa_dw3{}, sa_pw3{};
  int mid{STEM_MID};
  int c{STEM_C};
};

// 28x28 float image [0,1] → tokens [T,C]
void stem_forward(const float* img28, const StemWeights& w, float* tokens_tc,
                  bool qat = true);

}  // namespace npukit

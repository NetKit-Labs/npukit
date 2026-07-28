#pragma once
// Tiny CPU-side ops: DS-stem helpers, float Softmax/RMSNorm/GELU, Q12 helpers.
// No XNNPACK — keep the A9 path small.

#include "npukit/regs.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace npukit {

inline float from_q12(int32_t v) { return float(v) / float(ONE_Q12); }
inline int32_t to_q12(float v) {
  return static_cast<int32_t>(std::lround(v * float(ONE_Q12)));
}
inline float from_q16(int32_t v) { return float(v) / float(ONE_Q16); }
inline int32_t to_q16(float v) {
  return static_cast<int32_t>(std::lround(v * float(ONE_Q16)));
}

inline void quant_q12_to_i8(const int32_t* x, int n, float scale, int8_t* out) {
  for (int i = 0; i < n; ++i) {
    float q = std::round(from_q12(x[i]) * scale);
    if (q > 127.f) q = 127.f;
    if (q < -128.f) q = -128.f;
    out[i] = static_cast<int8_t>(q);
  }
}

inline void dequant_gemm_to_q12(const int32_t* c, int n, float a_scale, float b_scale,
                                int32_t* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = to_q12(float(c[i]) * a_scale * b_scale);
  }
}

inline void dequant_gemm_to_q12_per_ch(const int32_t* c, int rows, int cols, float a_scale,
                                       const float* b_scale_col, int32_t* out) {
  for (int r = 0; r < rows; ++r) {
    for (int col = 0; col < cols; ++col) {
      const int i = r * cols + col;
      out[i] = to_q12(float(c[i]) * a_scale * b_scale_col[col]);
    }
  }
}

// Softmax row → Q16 (float path).
inline void float_softmax_row(const int32_t* x_q12, int n, int32_t* out_q16) {
  float m = from_q12(x_q12[0]);
  for (int i = 1; i < n; ++i) {
    float v = from_q12(x_q12[i]);
    if (v > m) m = v;
  }
  float sum = 0.f;
  std::vector<float> e(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    e[static_cast<size_t>(i)] = std::exp(from_q12(x_q12[i]) - m);
    sum += e[static_cast<size_t>(i)];
  }
  for (int i = 0; i < n; ++i) out_q16[i] = to_q16(e[static_cast<size_t>(i)] / sum);
}

inline void float_rmsnorm_row(const int32_t* x_q12, const int32_t* gamma_q12, int n,
                              int32_t* out_q12) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = from_q12(x_q12[i]);
    acc += v * v;
  }
  float inv = 1.f / std::sqrt(float(acc / n) + 1e-5f);
  for (int i = 0; i < n; ++i) {
    out_q12[i] = to_q12(from_q12(x_q12[i]) * inv * from_q12(gamma_q12[i]));
  }
}

// Torch-default GELU on Q12 vector — used when mlp_h > MAX_LEN.
inline void float_gelu_row(const int32_t* x_q12, int n, int32_t* out_q12) {
  constexpr float inv_sqrt2 = 0.70710678118f;
  for (int i = 0; i < n; ++i) {
    float x = from_q12(x_q12[i]);
    out_q12[i] = to_q12(0.5f * x * (1.f + std::erf(x * inv_sqrt2)));
  }
}

inline void residual_row(const int32_t* x, const int32_t* y, int n, int32_t* out) {
  for (int i = 0; i < n; ++i) out[i] = x[i] + y[i];
}

// Fake-quant helper for stem.
inline void fq_inplace(float* x, int n, float scale) {
  for (int i = 0; i < n; ++i) {
    float q = std::round(x[i] * scale);
    if (q > 127.f) q = 127.f;
    if (q < -128.f) q = -128.f;
    x[i] = q / scale;
  }
}

}  // namespace npukit

#include "npukit/stem.hpp"
#include "npukit/cpu_ops.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace npukit {
namespace {

void relu(std::vector<float>& x) {
  for (float& v : x) v = v > 0.f ? v : 0.f;
}

// NCHW conv3x3 dense
void conv3x3(const float* x, int cin, int hin, int win, const float* w, const float* b,
             int cout, int stride, int pad, std::vector<float>& y) {
  const int hout = (hin + 2 * pad - 3) / stride + 1;
  const int wout = (win + 2 * pad - 3) / stride + 1;
  y.assign(size_t(cout) * hout * wout, 0.f);
  for (int oc = 0; oc < cout; ++oc) {
    for (int oh = 0; oh < hout; ++oh) {
      for (int ow = 0; ow < wout; ++ow) {
        float acc = b[oc];
        for (int ic = 0; ic < cin; ++ic) {
          for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
              int ih = oh * stride + kh - pad;
              int iw = ow * stride + kw - pad;
              if (ih < 0 || iw < 0 || ih >= hin || iw >= win) continue;
              float xv = x[(ic * hin + ih) * win + iw];
              float wv = w[((oc * cin + ic) * 3 + kh) * 3 + kw];
              acc += xv * wv;
            }
          }
        }
        y[(oc * hout + oh) * wout + ow] = acc;
      }
    }
  }
}

void dw_conv3x3(const float* x, int c, int hin, int win, const float* w, const float* b,
                int stride, int pad, std::vector<float>& y) {
  const int hout = (hin + 2 * pad - 3) / stride + 1;
  const int wout = (win + 2 * pad - 3) / stride + 1;
  y.assign(size_t(c) * hout * wout, 0.f);
  for (int ch = 0; ch < c; ++ch) {
    for (int oh = 0; oh < hout; ++oh) {
      for (int ow = 0; ow < wout; ++ow) {
        float acc = b[ch];
        for (int kh = 0; kh < 3; ++kh) {
          for (int kw = 0; kw < 3; ++kw) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih < 0 || iw < 0 || ih >= hin || iw >= win) continue;
            acc += x[(ch * hin + ih) * win + iw] * w[(ch * 3 + kh) * 3 + kw];
          }
        }
        y[(ch * hout + oh) * wout + ow] = acc;
      }
    }
  }
}

void pw_conv1x1(const float* x, int cin, int h, int w, const float* wt, const float* b,
                int cout, std::vector<float>& y) {
  y.assign(size_t(cout) * h * w, 0.f);
  for (int oc = 0; oc < cout; ++oc) {
    for (int i = 0; i < h * w; ++i) {
      float acc = b[oc];
      for (int ic = 0; ic < cin; ++ic) {
        acc += x[ic * h * w + i] * wt[oc * cin + ic];
      }
      y[oc * h * w + i] = acc;
    }
  }
}

}  // namespace

void stem_forward(const float* img28, const StemWeights& sw, float* tokens_tc, bool qat) {
  const int mid = sw.mid;
  const int c = sw.c;
  std::vector<float> x(size_t(1) * STEM_IMG * STEM_IMG);
  std::memcpy(x.data(), img28, x.size() * sizeof(float));
  if (qat) fq_inplace(x.data(), int(x.size()), sw.sa_in);

  std::vector<float> y;
  // reshape as NCHW cin=1
  conv3x3(x.data(), 1, STEM_IMG, STEM_IMG, sw.w_stem.data(), sw.b_stem.data(), mid, 2, 1, y);
  relu(y);
  if (qat) fq_inplace(y.data(), int(y.size()), sw.sa_stem);

  dw_conv3x3(y.data(), mid, 14, 14, sw.w_dw.data(), sw.b_dw.data(), 2, 1, x);
  relu(x);
  if (qat) fq_inplace(x.data(), int(x.size()), sw.sa_dw);

  pw_conv1x1(x.data(), mid, 7, 7, sw.w_pw.data(), sw.b_pw.data(), mid, y);
  relu(y);
  if (qat) fq_inplace(y.data(), int(y.size()), sw.sa_pw);

  dw_conv3x3(y.data(), mid, 7, 7, sw.w_dw2.data(), sw.b_dw2.data(), 1, 1, x);
  relu(x);
  if (qat) fq_inplace(x.data(), int(x.size()), sw.sa_dw2);

  pw_conv1x1(x.data(), mid, 7, 7, sw.w_pw2.data(), sw.b_pw2.data(), mid, y);
  relu(y);
  if (qat) fq_inplace(y.data(), int(y.size()), sw.sa_pw2);

  dw_conv3x3(y.data(), mid, 7, 7, sw.w_dw3.data(), sw.b_dw3.data(), 1, 1, x);
  relu(x);
  if (qat) fq_inplace(x.data(), int(x.size()), sw.sa_dw3);

  pw_conv1x1(x.data(), mid, 7, 7, sw.w_pw3.data(), sw.b_pw3.data(), c, y);
  relu(y);

  // pad 7→8, avg pool 2 → 4x4, then [C,16] → [T,C]
  std::vector<float> pad(size_t(c) * 8 * 8, 0.f);
  for (int ch = 0; ch < c; ++ch) {
    for (int h = 0; h < 7; ++h) {
      for (int w = 0; w < 7; ++w) {
        pad[(ch * 8 + h) * 8 + w] = y[(ch * 7 + h) * 7 + w];
      }
    }
  }
  for (int t = 0; t < STEM_T; ++t) {
    int oh = t / 4, ow = t % 4;
    for (int ch = 0; ch < c; ++ch) {
      float s = 0.f;
      for (int dh = 0; dh < 2; ++dh)
        for (int dw = 0; dw < 2; ++dw)
          s += pad[(ch * 8 + oh * 2 + dh) * 8 + ow * 2 + dw];
      tokens_tc[t * c + ch] = s * 0.25f;
    }
  }
}

}  // namespace npukit

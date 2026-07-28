#include "npukit/dscnn.hpp"
#include "npukit/cpu_ops.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace npukit {
namespace {

class Cursor {
 public:
  explicit Cursor(const std::vector<uint8_t>& buf) : p_(buf.data()), end_(buf.data() + buf.size()) {}
  void need(size_t n) const {
    if (size_t(end_ - p_) < n) throw std::runtime_error("dscnn.bin: truncated");
  }
  template <typename T>
  T rd() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  void rd_bytes(void* dst, size_t n) {
    need(n);
    std::memcpy(dst, p_, n);
    p_ += n;
  }
  template <typename T>
  void rd_vec(std::vector<T>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n * sizeof(T));
  }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

void conv3x3(const float* x, int cin, int hin, int win, const float* w, const float* b, int cout,
             int stride, int pad, std::vector<float>& y) {
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
              acc += x[(ic * hin + ih) * win + iw] * w[((oc * cin + ic) * 3 + kh) * 3 + kw];
            }
          }
        }
        y[(oc * hout + oh) * wout + ow] = acc;
      }
    }
  }
}

void dw_conv3x3(const float* x, int c, int hin, int win, const float* w, const float* b, int stride,
                int pad, std::vector<float>& y) {
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

void pw_conv1x1(const float* x, int cin, int h, int w, const float* wt, const float* b, int cout,
                std::vector<float>& y) {
  y.assign(size_t(cout) * h * w, 0.f);
  for (int oc = 0; oc < cout; ++oc) {
    for (int i = 0; i < h * w; ++i) {
      float acc = b[oc];
      for (int ic = 0; ic < cin; ++ic) acc += x[ic * h * w + i] * wt[oc * cin + ic];
      y[oc * h * w + i] = acc;
    }
  }
}

void relu(std::vector<float>& x) {
  for (float& v : x) v = v > 0.f ? v : 0.f;
}

}  // namespace

DscnnWeights load_dscnn_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const auto sz = size_t(f.tellg());
  f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(sz));
  Cursor cur(buf);

  char magic[4];
  cur.rd_bytes(magic, 4);
  if (std::memcmp(magic, "NKD1", 4) != 0) throw std::runtime_error("bad dscnn.bin magic");
  if (cur.rd<uint32_t>() != 1) throw std::runtime_error("unsupported dscnn.bin version");

  DscnnWeights w;
  w.img = int(cur.rd<uint32_t>());
  for (int i = 0; i < 8; ++i) w.sa[i] = cur.rd<float>();

  cur.rd_vec(w.w_stem, 16 * 1 * 9);
  cur.rd_vec(w.b_stem, 16);
  cur.rd_vec(w.w_b1_dw, 16 * 1 * 9);
  cur.rd_vec(w.b_b1_dw, 16);
  cur.rd_vec(w.w_b1_pw, 32 * 16);
  cur.rd_vec(w.b_b1_pw, 32);
  cur.rd_vec(w.w_b2_dw, 32 * 1 * 9);
  cur.rd_vec(w.b_b2_dw, 32);
  cur.rd_vec(w.w_b2_pw, 64 * 32);
  cur.rd_vec(w.b_b2_pw, 64);
  cur.rd_vec(w.w_b3_dw, 64 * 1 * 9);
  cur.rd_vec(w.b_b3_dw, 64);
  cur.rd_vec(w.w_b3_pw, 64 * 64);
  cur.rd_vec(w.b_b3_pw, 64);
  cur.rd_vec(w.w_fc, 64 * 10);
  cur.rd_vec(w.b_fc, 10);

  const uint32_t ns = cur.rd<uint32_t>();
  w.samples.resize(ns);
  for (uint32_t i = 0; i < ns; ++i) {
    auto& sm = w.samples[i];
    cur.rd_vec(sm.img, size_t(w.img) * w.img);
    sm.label = cur.rd<int32_t>();
    cur.rd_vec(sm.ref_logits, w.n_class);
  }
  return w;
}

void dscnn_forward(const DscnnWeights& w, const float* img28, float* logits) {
  // sa indices: 0=in,1=stem,2=b1_dw,3=b1_pw,4=b2_dw,5=b2_pw,6=b3_dw,7=b3_pw
  std::vector<float> x(size_t(1) * 28 * 28), y;
  std::memcpy(x.data(), img28, x.size() * sizeof(float));
  fq_inplace(x.data(), int(x.size()), w.sa[0]);

  conv3x3(x.data(), 1, 28, 28, w.w_stem.data(), w.b_stem.data(), 16, 1, 1, y);
  relu(y);
  fq_inplace(y.data(), int(y.size()), w.sa[1]);

  dw_conv3x3(y.data(), 16, 28, 28, w.w_b1_dw.data(), w.b_b1_dw.data(), 2, 1, x);
  relu(x);
  fq_inplace(x.data(), int(x.size()), w.sa[2]);

  pw_conv1x1(x.data(), 16, 14, 14, w.w_b1_pw.data(), w.b_b1_pw.data(), 32, y);
  relu(y);
  fq_inplace(y.data(), int(y.size()), w.sa[3]);

  dw_conv3x3(y.data(), 32, 14, 14, w.w_b2_dw.data(), w.b_b2_dw.data(), 2, 1, x);
  relu(x);
  fq_inplace(x.data(), int(x.size()), w.sa[4]);

  pw_conv1x1(x.data(), 32, 7, 7, w.w_b2_pw.data(), w.b_b2_pw.data(), 64, y);
  relu(y);
  fq_inplace(y.data(), int(y.size()), w.sa[5]);

  dw_conv3x3(y.data(), 64, 7, 7, w.w_b3_dw.data(), w.b_b3_dw.data(), 1, 1, x);
  relu(x);
  fq_inplace(x.data(), int(x.size()), w.sa[6]);

  pw_conv1x1(x.data(), 64, 7, 7, w.w_b3_pw.data(), w.b_b3_pw.data(), 64, y);
  relu(y);

  // GAP → [64], then fq with sa_b3_pw
  float pooled[64];
  for (int c = 0; c < 64; ++c) {
    float s = 0.f;
    const float* plane = y.data() + c * 49;
    for (int i = 0; i < 49; ++i) s += plane[i];
    pooled[c] = s / 49.f;
  }
  fq_inplace(pooled, 64, w.sa[7]);

  for (int j = 0; j < 10; ++j) {
    float acc = w.b_fc[j];
    for (int i = 0; i < 64; ++i) acc += pooled[i] * w.w_fc[i * 10 + j];
    logits[j] = acc;
  }
}

}  // namespace npukit

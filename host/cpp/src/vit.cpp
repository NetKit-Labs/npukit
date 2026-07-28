#include "npukit/vit.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npukit {
namespace {

class Cursor {
 public:
  explicit Cursor(const std::vector<uint8_t>& buf) : p_(buf.data()), end_(buf.data() + buf.size()) {}

  void need(size_t n) const {
    if (size_t(end_ - p_) < n) throw std::runtime_error("vit.bin: truncated");
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

  size_t remaining() const { return size_t(end_ - p_); }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

void cpu_matmul_i8(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t acc = 0;
      for (int t = 0; t < k; ++t) acc += int32_t(a[i * k + t]) * int32_t(b[t * n + j]);
      c[i * n + j] = acc;
    }
  }
}

void matmul(Device* dev, const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n) {
  if (dev && (m % 8) == 0 && (k % 8) == 0) {
    if ((n % 8) == 0) {
      dev->matmul_i8(a, b, c, m, k, n);
      return;
    }
    // Pad N to next multiple of 8 (cls head N=10 → 16).
    const int np = (n + 7) & ~7;
    std::vector<int8_t> bp(size_t(k) * np, 0);
    for (int t = 0; t < k; ++t)
      for (int j = 0; j < n; ++j) bp[t * np + j] = b[t * n + j];
    std::vector<int32_t> cp(size_t(m) * np);
    dev->matmul_i8(a, bp.data(), cp.data(), m, k, np);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) c[i * n + j] = cp[i * np + j];
    return;
  }
  cpu_matmul_i8(a, b, c, m, k, n);
}

void quant_q16_to_i8(const int32_t* x, int n, float scale, int8_t* out) {
  for (int i = 0; i < n; ++i) {
    float q = std::round(from_q16(x[i]) * scale);
    if (q > 127.f) q = 127.f;
    if (q < -128.f) q = -128.f;
    out[i] = static_cast<int8_t>(q);
  }
}

void matmul_q12(Device* dev, const int32_t* a_q12, int m, int k, const int8_t* w_i8, int n,
                float scale_act, const double* inv_sw, int32_t* out_q12) {
  std::vector<int8_t> a_i8(size_t(m) * k);
  quant_q12_to_i8(a_q12, m * k, scale_act, a_i8.data());
  std::vector<int32_t> c(size_t(m) * n);
  matmul(dev, a_i8.data(), w_i8, c.data(), m, k, n);
  const float a_deq = 1.f / scale_act;
  for (int r = 0; r < m; ++r) {
    for (int col = 0; col < n; ++col) {
      const int i = r * n + col;
      out_q12[i] = to_q12(float(c[i]) * a_deq * float(inv_sw[col]));
    }
  }
}

void rmsnorm_rows(Device* dev, GlueMode glue, const int32_t* x, const int32_t* gamma, int rows,
                  int n, int32_t* out) {
  for (int r = 0; r < rows; ++r) {
    const int32_t* xr = x + r * n;
    int32_t* orow = out + r * n;
    if (glue == GlueMode::Hybrid && dev && n <= MAX_LEN) {
      dev->glue_run(OP_RMSNORM, xr, n, orow, nullptr, gamma, 1);
    } else {
      float_rmsnorm_row(xr, gamma, n, orow);
    }
  }
}

void softmax_rows(Device* dev, GlueMode glue, const int32_t* x, int rows, int n, int32_t* out_q16) {
  for (int r = 0; r < rows; ++r) {
    const int32_t* xr = x + r * n;
    int32_t* orow = out_q16 + r * n;
    if (glue == GlueMode::Hybrid && dev && n <= MAX_LEN) {
      dev->glue_run(OP_SOFTMAX, xr, n, orow);
    } else {
      float_softmax_row(xr, n, orow);
    }
  }
}

void residual_rows(Device* dev, GlueMode glue, const int32_t* x, const int32_t* y, int rows, int n,
                   int32_t* out) {
  for (int r = 0; r < rows; ++r) {
    const int32_t* xr = x + r * n;
    const int32_t* yr = y + r * n;
    int32_t* orow = out + r * n;
    if (glue == GlueMode::Hybrid && dev && n <= MAX_LEN) {
      dev->glue_run(OP_RESIDUAL, xr, n, orow, yr);
    } else {
      residual_row(xr, yr, n, orow);
    }
  }
}

void gelu_rows(const int32_t* x, int rows, int n, int32_t* out) {
  // Always float when n can exceed MAX_LEN (MLP=32).
  for (int r = 0; r < rows; ++r) float_gelu_row(x + r * n, n, out + r * n);
}

void block_forward(Device* dev, const VitBlockWeights& blk, int t, int d, int mlp,
                   GlueMode glue, const int32_t* x_in, int32_t* x_out) {
  const float sa = float(blk.scale_act);
  const float sp = float(blk.scale_p);

  std::vector<int32_t> xn(size_t(t) * d);
  rmsnorm_rows(dev, glue, x_in, blk.gamma1.data(), t, d, xn.data());

  std::vector<int32_t> q(size_t(t) * d), k(size_t(t) * d), v(size_t(t) * d);
  matmul_q12(dev, xn.data(), t, d, blk.wq.data(), d, sa, blk.inv_sw_wq.data(), q.data());
  matmul_q12(dev, xn.data(), t, d, blk.wk.data(), d, sa, blk.inv_sw_wk.data(), k.data());
  matmul_q12(dev, xn.data(), t, d, blk.wv.data(), d, sa, blk.inv_sw_wv.data(), v.data());

  // scores = (Q @ K^T) / sqrt(D)
  std::vector<int8_t> q_i8(size_t(t) * d), kt_i8(size_t(d) * t);
  quant_q12_to_i8(q.data(), t * d, sa, q_i8.data());
  // K^T [D,T]
  std::vector<int32_t> kt(size_t(d) * t);
  for (int i = 0; i < t; ++i)
    for (int j = 0; j < d; ++j) kt[j * t + i] = k[i * d + j];
  quant_q12_to_i8(kt.data(), d * t, sa, kt_i8.data());

  std::vector<int32_t> scores_i32(size_t(t) * t), scores(size_t(t) * t);
  matmul(dev, q_i8.data(), kt_i8.data(), scores_i32.data(), t, d, t);
  const float a_deq = 1.f / sa;
  for (int i = 0; i < t * t; ++i) scores[i] = to_q12(float(scores_i32[i]) * a_deq * a_deq);

  const int32_t inv_sqrt = to_q12(1.f / std::sqrt(float(d)));
  for (int i = 0; i < t * t; ++i) {
    scores[i] = int32_t((int64_t(scores[i]) * int64_t(inv_sqrt)) >> Q12);
  }

  std::vector<int32_t> p_q16(size_t(t) * t);
  softmax_rows(dev, glue, scores.data(), t, t, p_q16.data());

  std::vector<int8_t> p_i8(size_t(t) * t), v_i8(size_t(t) * d);
  quant_q16_to_i8(p_q16.data(), t * t, sp, p_i8.data());
  quant_q12_to_i8(v.data(), t * d, sa, v_i8.data());
  std::vector<int32_t> attn_i32(size_t(t) * d), attn(size_t(t) * d);
  matmul(dev, p_i8.data(), v_i8.data(), attn_i32.data(), t, t, d);
  const float p_deq = 1.f / sp;
  for (int i = 0; i < t * d; ++i) attn[i] = to_q12(float(attn_i32[i]) * p_deq * a_deq);

  std::vector<int32_t> attn_o(size_t(t) * d);
  matmul_q12(dev, attn.data(), t, d, blk.wo.data(), d, sa, blk.inv_sw_wo.data(), attn_o.data());

  std::vector<int32_t> x2(size_t(t) * d);
  residual_rows(dev, glue, x_in, attn_o.data(), t, d, x2.data());

  // FFN
  rmsnorm_rows(dev, glue, x2.data(), blk.gamma2.data(), t, d, xn.data());
  std::vector<int32_t> h(size_t(t) * mlp), h2(size_t(t) * mlp);
  matmul_q12(dev, xn.data(), t, d, blk.w1.data(), mlp, sa, blk.inv_sw_w1.data(), h.data());
  gelu_rows(h.data(), t, mlp, h2.data());
  matmul_q12(dev, h2.data(), t, mlp, blk.w2.data(), d, sa, blk.inv_sw_w2.data(), attn_o.data());
  residual_rows(dev, glue, x2.data(), attn_o.data(), t, d, x_out);
}

}  // namespace

VitWeights load_vit_bin(const std::string& path) {
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
  if (std::memcmp(magic, "NKV1", 4) != 0) throw std::runtime_error("bad vit.bin magic");
  const uint32_t ver = cur.rd<uint32_t>();
  if (ver != 1) throw std::runtime_error("unsupported vit.bin version");

  VitWeights w;
  w.t = int(cur.rd<uint32_t>());
  w.d = int(cur.rd<uint32_t>());
  w.mlp = int(cur.rd<uint32_t>());
  w.layers = int(cur.rd<uint32_t>());
  w.n_class = int(cur.rd<uint32_t>());
  w.mid = int(cur.rd<uint32_t>());
  w.c = int(cur.rd<uint32_t>());
  w.img = int(cur.rd<uint32_t>());

  StemWeights& s = w.stem;
  s.mid = w.mid;
  s.c = w.c;
  s.sa_in = cur.rd<float>();
  s.sa_stem = cur.rd<float>();
  s.sa_dw = cur.rd<float>();
  s.sa_pw = cur.rd<float>();
  s.sa_dw2 = cur.rd<float>();
  s.sa_pw2 = cur.rd<float>();
  s.sa_dw3 = cur.rd<float>();
  s.sa_pw3 = cur.rd<float>();

  auto rd_stem = [&](std::vector<float>& wt, size_t wn, std::vector<float>& b, size_t bn) {
    cur.rd_vec(wt, wn);
    cur.rd_vec(b, bn);
  };
  rd_stem(s.w_stem, size_t(w.mid) * 9, s.b_stem, w.mid);
  rd_stem(s.w_dw, size_t(w.mid) * 9, s.b_dw, w.mid);
  rd_stem(s.w_pw, size_t(w.mid) * w.mid, s.b_pw, w.mid);
  rd_stem(s.w_dw2, size_t(w.mid) * 9, s.b_dw2, w.mid);
  rd_stem(s.w_pw2, size_t(w.mid) * w.mid, s.b_pw2, w.mid);
  rd_stem(s.w_dw3, size_t(w.mid) * 9, s.b_dw3, w.mid);
  rd_stem(s.w_pw3, size_t(w.c) * w.mid, s.b_pw3, w.c);

  cur.rd_vec(w.pos, size_t(w.t) * w.d);

  w.blocks.resize(w.layers);
  for (int li = 0; li < w.layers; ++li) {
    VitBlockWeights& b = w.blocks[li];
    const size_t dd = size_t(w.d) * w.d;
    const size_t dmlp = size_t(w.d) * w.mlp;
    const size_t mlpd = size_t(w.mlp) * w.d;
    cur.rd_vec(b.wq, dd);
    cur.rd_vec(b.wk, dd);
    cur.rd_vec(b.wv, dd);
    cur.rd_vec(b.wo, dd);
    cur.rd_vec(b.w1, dmlp);
    cur.rd_vec(b.w2, mlpd);
    cur.rd_vec(b.gamma1, w.d);
    cur.rd_vec(b.gamma2, w.d);
    b.scale_act = cur.rd<double>();
    b.scale_p = cur.rd<double>();
    cur.rd_vec(b.inv_sw_wq, w.d);
    cur.rd_vec(b.inv_sw_wk, w.d);
    cur.rd_vec(b.inv_sw_wv, w.d);
    cur.rd_vec(b.inv_sw_wo, w.d);
    cur.rd_vec(b.inv_sw_w1, w.mlp);
    cur.rd_vec(b.inv_sw_w2, w.d);
  }

  cur.rd_vec(w.w_cls, size_t(w.d) * w.n_class);
  w.scale_cls_act = cur.rd<double>();
  cur.rd_vec(w.inv_sw_cls, w.n_class);

  const uint32_t ns = cur.rd<uint32_t>();
  w.samples.resize(ns);
  for (uint32_t i = 0; i < ns; ++i) {
    VitSample& sm = w.samples[i];
    cur.rd_vec(sm.img, size_t(w.img) * w.img);
    sm.label = cur.rd<int32_t>();
    cur.rd_vec(sm.ref_logits_q12, w.n_class);
  }
  return w;
}

void vit_forward(Device* dev, const VitWeights& w, const float* img28, int32_t* logits_q12,
                 GlueMode glue) {
  std::vector<float> tok(size_t(w.t) * w.c);
  stem_forward(img28, w.stem, tok.data(), true);

  std::vector<int32_t> x(size_t(w.t) * w.d);
  for (int i = 0; i < w.t * w.d; ++i) x[i] = to_q12(tok[i]);
  // + pos
  {
    std::vector<int32_t> y(size_t(w.t) * w.d);
    residual_rows(dev, glue, x.data(), w.pos.data(), w.t, w.d, y.data());
    x.swap(y);
  }

  std::vector<int32_t> y(size_t(w.t) * w.d);
  for (int li = 0; li < w.layers; ++li) {
    block_forward(dev, w.blocks[li], w.t, w.d, w.mlp, glue, x.data(), y.data());
    x.swap(y);
  }

  // Mean-pool + linear head (CPU; N_CLASS=10)
  std::vector<int32_t> pooled(w.d);
  for (int j = 0; j < w.d; ++j) {
    double acc = 0.0;
    for (int i = 0; i < w.t; ++i) acc += double(x[i * w.d + j]);
    pooled[j] = int32_t(std::llround(acc / double(w.t)));
  }
  matmul_q12(dev, pooled.data(), 1, w.d, w.w_cls.data(), w.n_class, float(w.scale_cls_act),
             w.inv_sw_cls.data(), logits_q12);
}

}  // namespace npukit

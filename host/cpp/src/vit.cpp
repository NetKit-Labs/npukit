#include "npukit/vit.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npukit {

void matmul_q12(Device* dev, const int32_t* a_q12, int m, int k, const int8_t* w_i8, int n,
                float scale_act, const double* inv_sw, int32_t* out_q12) {
  std::vector<int8_t> a_i8(size_t(m) * k);
  quant_q12_to_i8(a_q12, m * k, scale_act, a_i8.data());
  std::vector<int32_t> c(size_t(m) * n);
  if (dev && (m % 8) == 0 && (k % 8) == 0 && (n % 8) == 0) {
    dev->matmul_i8(a_i8.data(), w_i8, c.data(), m, k, n);
  } else if (dev && (m % 8) == 0 && (k % 8) == 0) {
    const int np = (n + 7) & ~7;
    std::vector<int8_t> bp(size_t(k) * np, 0);
    for (int t = 0; t < k; ++t)
      for (int j = 0; j < n; ++j) bp[size_t(t) * np + j] = w_i8[size_t(t) * n + j];
    std::vector<int32_t> cp(size_t(m) * np);
    dev->matmul_i8(a_i8.data(), bp.data(), cp.data(), m, k, np);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) c[size_t(i) * n + j] = cp[size_t(i) * np + j];
  } else {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int32_t acc = 0;
        for (int t = 0; t < k; ++t)
          acc += int32_t(a_i8[size_t(i) * k + t]) * int32_t(w_i8[size_t(t) * n + j]);
        c[size_t(i) * n + j] = acc;
      }
    }
  }
  const float a_deq = 1.f / scale_act;
  for (int r = 0; r < m; ++r) {
    for (int col = 0; col < n; ++col) {
      const int i = r * n + col;
      out_q12[i] = to_q12(float(c[i]) * a_deq * float(inv_sw[col]));
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

void block_forward(Device* dev, const VitBlockWeights& blk, int t, int d, int mlp,
                   GlueMode glue, const int32_t* x_in, int32_t* x_out, bool causal) {
  const float sa = float(blk.scale_act);
  const float sp = float(blk.scale_p);

  std::vector<int32_t> xn(size_t(t) * d);
  for (int r = 0; r < t; ++r) {
    if (glue == GlueMode::Hybrid && dev && d <= MAX_LEN) {
      dev->glue_run(OP_RMSNORM, x_in + r * d, d, xn.data() + r * d, nullptr, blk.gamma1.data(), 1);
    } else {
      float_rmsnorm_row(x_in + r * d, blk.gamma1.data(), d, xn.data() + r * d);
    }
  }

  std::vector<int32_t> q(size_t(t) * d), k(size_t(t) * d), v(size_t(t) * d);
  matmul_q12(dev, xn.data(), t, d, blk.wq.data(), d, sa, blk.inv_sw_wq.data(), q.data());
  matmul_q12(dev, xn.data(), t, d, blk.wk.data(), d, sa, blk.inv_sw_wk.data(), k.data());
  matmul_q12(dev, xn.data(), t, d, blk.wv.data(), d, sa, blk.inv_sw_wv.data(), v.data());

  std::vector<int8_t> q_i8(size_t(t) * d), kt_i8(size_t(d) * t);
  quant_q12_to_i8(q.data(), t * d, sa, q_i8.data());
  std::vector<int32_t> kt(size_t(d) * t);
  for (int i = 0; i < t; ++i)
    for (int j = 0; j < d; ++j) kt[size_t(j) * t + i] = k[size_t(i) * d + j];
  quant_q12_to_i8(kt.data(), d * t, sa, kt_i8.data());

  std::vector<int32_t> scores_i32(size_t(t) * t), scores(size_t(t) * t);
  if (dev && (t % 8) == 0 && (d % 8) == 0) {
    dev->matmul_i8(q_i8.data(), kt_i8.data(), scores_i32.data(), t, d, t);
  } else {
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < t; ++j) {
        int32_t acc = 0;
        for (int kk = 0; kk < d; ++kk)
          acc += int32_t(q_i8[size_t(i) * d + kk]) * int32_t(kt_i8[size_t(kk) * t + j]);
        scores_i32[size_t(i) * t + j] = acc;
      }
    }
  }
  const float a_deq = 1.f / sa;
  for (int i = 0; i < t * t; ++i) scores[i] = to_q12(float(scores_i32[i]) * a_deq * a_deq);

  const int32_t inv_sqrt = to_q12(1.f / std::sqrt(float(d)));
  for (int i = 0; i < t * t; ++i) {
    scores[i] = int32_t((int64_t(scores[i]) * int64_t(inv_sqrt)) >> Q12);
  }
  if (causal) {
    constexpr int32_t neg = INT32_MIN / 4;
    for (int r = 0; r < t; ++r)
      for (int c = r + 1; c < t; ++c) scores[size_t(r) * t + c] = neg;
  }

  std::vector<int32_t> p_q16(size_t(t) * t);
  for (int r = 0; r < t; ++r) {
    if (glue == GlueMode::Hybrid && dev && t <= MAX_LEN) {
      dev->glue_run(OP_SOFTMAX, scores.data() + r * t, t, p_q16.data() + r * t);
    } else {
      float_softmax_row(scores.data() + r * t, t, p_q16.data() + r * t);
    }
  }

  std::vector<int8_t> p_i8(size_t(t) * t), v_i8(size_t(t) * d);
  for (int i = 0; i < t * t; ++i) {
    float qq = std::round(from_q16(p_q16[i]) * sp);
    if (qq > 127.f) qq = 127.f;
    if (qq < -128.f) qq = -128.f;
    p_i8[i] = static_cast<int8_t>(qq);
  }
  quant_q12_to_i8(v.data(), t * d, sa, v_i8.data());
  std::vector<int32_t> attn_i32(size_t(t) * d), attn(size_t(t) * d);
  if (dev && (t % 8) == 0 && (d % 8) == 0) {
    dev->matmul_i8(p_i8.data(), v_i8.data(), attn_i32.data(), t, t, d);
  } else {
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < d; ++j) {
        int32_t acc = 0;
        for (int kk = 0; kk < t; ++kk)
          acc += int32_t(p_i8[size_t(i) * t + kk]) * int32_t(v_i8[size_t(kk) * d + j]);
        attn_i32[size_t(i) * d + j] = acc;
      }
    }
  }
  const float p_deq = 1.f / sp;
  for (int i = 0; i < t * d; ++i) attn[i] = to_q12(float(attn_i32[i]) * p_deq * a_deq);

  std::vector<int32_t> attn_o(size_t(t) * d);
  matmul_q12(dev, attn.data(), t, d, blk.wo.data(), d, sa, blk.inv_sw_wo.data(), attn_o.data());

  std::vector<int32_t> x2(size_t(t) * d);
  residual_rows(dev, glue, x_in, attn_o.data(), t, d, x2.data());

  for (int r = 0; r < t; ++r) {
    if (glue == GlueMode::Hybrid && dev && d <= MAX_LEN) {
      dev->glue_run(OP_RMSNORM, x2.data() + r * d, d, xn.data() + r * d, nullptr, blk.gamma2.data(),
                    1);
    } else {
      float_rmsnorm_row(x2.data() + r * d, blk.gamma2.data(), d, xn.data() + r * d);
    }
  }
  std::vector<int32_t> h(size_t(t) * mlp), h2(size_t(t) * mlp);
  matmul_q12(dev, xn.data(), t, d, blk.w1.data(), mlp, sa, blk.inv_sw_w1.data(), h.data());
  for (int r = 0; r < t; ++r) float_gelu_row(h.data() + r * mlp, mlp, h2.data() + r * mlp);
  matmul_q12(dev, h2.data(), t, mlp, blk.w2.data(), d, sa, blk.inv_sw_w2.data(), attn_o.data());
  residual_rows(dev, glue, x2.data(), attn_o.data(), t, d, x_out);
}

VitWeights load_vit_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const auto sz = size_t(f.tellg());
  f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(sz));

  const uint8_t* p = buf.data();
  const uint8_t* end = buf.data() + buf.size();
  auto need = [&](size_t n) {
    if (size_t(end - p) < n) throw std::runtime_error("vit.bin: truncated");
  };
  auto rd_bytes = [&](void* dst, size_t n) {
    need(n);
    std::memcpy(dst, p, n);
    p += n;
  };
  auto rd_u32 = [&]() {
    uint32_t v;
    rd_bytes(&v, 4);
    return v;
  };
  auto rd_f32 = [&]() {
    float v;
    rd_bytes(&v, 4);
    return v;
  };
  auto rd_f64 = [&]() {
    double v;
    rd_bytes(&v, 8);
    return v;
  };
  auto rd_vec_i8 = [&](std::vector<int8_t>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n);
  };
  auto rd_vec_i32 = [&](std::vector<int32_t>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n * 4);
  };
  auto rd_vec_f32 = [&](std::vector<float>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n * 4);
  };
  auto rd_vec_f64 = [&](std::vector<double>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n * 8);
  };

  char magic[4];
  rd_bytes(magic, 4);
  if (std::memcmp(magic, "NKV1", 4) != 0) throw std::runtime_error("bad vit.bin magic");
  const uint32_t ver = rd_u32();
  if (ver != 1) throw std::runtime_error("unsupported vit.bin version");

  VitWeights w;
  w.t = int(rd_u32());
  w.d = int(rd_u32());
  w.mlp = int(rd_u32());
  w.layers = int(rd_u32());
  w.n_class = int(rd_u32());
  w.mid = int(rd_u32());
  w.c = int(rd_u32());
  w.img = int(rd_u32());

  StemWeights& s = w.stem;
  s.mid = w.mid;
  s.c = w.c;
  s.sa_in = rd_f32();
  s.sa_stem = rd_f32();
  s.sa_dw = rd_f32();
  s.sa_pw = rd_f32();
  s.sa_dw2 = rd_f32();
  s.sa_pw2 = rd_f32();
  s.sa_dw3 = rd_f32();
  s.sa_pw3 = rd_f32();

  auto rd_stem = [&](std::vector<float>& wt, size_t wn, std::vector<float>& b, size_t bn) {
    rd_vec_f32(wt, wn);
    rd_vec_f32(b, bn);
  };
  rd_stem(s.w_stem, size_t(w.mid) * 9, s.b_stem, w.mid);
  rd_stem(s.w_dw, size_t(w.mid) * 9, s.b_dw, w.mid);
  rd_stem(s.w_pw, size_t(w.mid) * w.mid, s.b_pw, w.mid);
  rd_stem(s.w_dw2, size_t(w.mid) * 9, s.b_dw2, w.mid);
  rd_stem(s.w_pw2, size_t(w.mid) * w.mid, s.b_pw2, w.mid);
  rd_stem(s.w_dw3, size_t(w.mid) * 9, s.b_dw3, w.mid);
  rd_stem(s.w_pw3, size_t(w.c) * w.mid, s.b_pw3, w.c);

  rd_vec_i32(w.pos, size_t(w.t) * w.d);

  w.blocks.resize(w.layers);
  for (int li = 0; li < w.layers; ++li) {
    VitBlockWeights& b = w.blocks[li];
    const size_t dd = size_t(w.d) * w.d;
    const size_t dmlp = size_t(w.d) * w.mlp;
    const size_t mlpd = size_t(w.mlp) * w.d;
    rd_vec_i8(b.wq, dd);
    rd_vec_i8(b.wk, dd);
    rd_vec_i8(b.wv, dd);
    rd_vec_i8(b.wo, dd);
    rd_vec_i8(b.w1, dmlp);
    rd_vec_i8(b.w2, mlpd);
    rd_vec_i32(b.gamma1, w.d);
    rd_vec_i32(b.gamma2, w.d);
    b.scale_act = rd_f64();
    b.scale_p = rd_f64();
    rd_vec_f64(b.inv_sw_wq, w.d);
    rd_vec_f64(b.inv_sw_wk, w.d);
    rd_vec_f64(b.inv_sw_wv, w.d);
    rd_vec_f64(b.inv_sw_wo, w.d);
    rd_vec_f64(b.inv_sw_w1, w.mlp);
    rd_vec_f64(b.inv_sw_w2, w.d);
  }

  rd_vec_i8(w.w_cls, size_t(w.d) * w.n_class);
  w.scale_cls_act = rd_f64();
  rd_vec_f64(w.inv_sw_cls, w.n_class);

  const uint32_t ns = rd_u32();
  w.samples.resize(ns);
  for (uint32_t i = 0; i < ns; ++i) {
    VitSample& sm = w.samples[i];
    rd_vec_f32(sm.img, size_t(w.img) * w.img);
    int32_t lab;
    rd_bytes(&lab, 4);
    sm.label = lab;
    rd_vec_i32(sm.ref_logits_q12, w.n_class);
  }
  return w;
}

void vit_forward(Device* dev, const VitWeights& w, const float* img28, int32_t* logits_q12,
                 GlueMode glue) {
  std::vector<float> tok(size_t(w.t) * w.c);
  stem_forward(img28, w.stem, tok.data(), true);

  std::vector<int32_t> x(size_t(w.t) * w.d);
  for (int i = 0; i < w.t * w.d; ++i) x[i] = to_q12(tok[i]);
  {
    std::vector<int32_t> y(size_t(w.t) * w.d);
    residual_rows(dev, glue, x.data(), w.pos.data(), w.t, w.d, y.data());
    x.swap(y);
  }

  std::vector<int32_t> y(size_t(w.t) * w.d);
  for (int li = 0; li < w.layers; ++li) {
    block_forward(dev, w.blocks[li], w.t, w.d, w.mlp, glue, x.data(), y.data(), false);
    x.swap(y);
  }

  std::vector<int32_t> pooled(w.d);
  for (int j = 0; j < w.d; ++j) {
    double acc = 0.0;
    for (int i = 0; i < w.t; ++i) acc += double(x[size_t(i) * w.d + j]);
    pooled[j] = int32_t(std::llround(acc / double(w.t)));
  }
  matmul_q12(dev, pooled.data(), 1, w.d, w.w_cls.data(), w.n_class, float(w.scale_cls_act),
             w.inv_sw_cls.data(), logits_q12);
}

}  // namespace npukit

#include "npukit/command_lm.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace npukit {

CommandLmWeights load_command_lm_bin(const std::string& path) {
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
    if (size_t(end - p) < n) throw std::runtime_error("command_lm.bin: truncated");
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
  auto rd_vec_f64 = [&](std::vector<double>& v, size_t n) {
    v.resize(n);
    rd_bytes(v.data(), n * 8);
  };

  char magic[4];
  rd_bytes(magic, 4);
  if (std::memcmp(magic, "NKL1", 4) != 0) throw std::runtime_error("bad command_lm.bin magic");
  if (rd_u32() != 1) throw std::runtime_error("unsupported command_lm.bin version");

  CommandLmWeights w;
  w.t = int(rd_u32());
  w.d = int(rd_u32());
  w.mlp = int(rd_u32());
  w.layers = int(rd_u32());
  w.vocab = int(rd_u32());
  w.pad_id = int(rd_u32());

  rd_vec_i32(w.pos, size_t(w.t) * w.d);
  rd_vec_i8(w.w_emb, size_t(w.vocab) * w.d);
  w.scale_emb_act = rd_f64();
  rd_vec_f64(w.inv_sw_emb, w.d);

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

  rd_vec_i8(w.w_lm, size_t(w.d) * w.vocab);
  w.scale_lm_act = rd_f64();
  rd_vec_f64(w.inv_sw_lm, w.vocab);

  const uint32_t ns = rd_u32();
  w.samples.resize(ns);
  for (uint32_t i = 0; i < ns; ++i) {
    rd_vec_i32(w.samples[i].input_ids, w.t);
    rd_vec_i32(w.samples[i].target_ids, w.t);
  }
  return w;
}

void command_lm_forward(Device* dev, const CommandLmWeights& w, const int32_t* token_ids,
                        int32_t* logits_q12, GlueMode glue) {
  std::vector<int32_t> x(size_t(w.t) * w.d, 0);
  for (int i = 0; i < w.t; ++i) {
    const int tid = int(token_ids[i]);
    if (tid == w.pad_id || tid < 0 || tid >= w.vocab) continue;
    for (int j = 0; j < w.d; ++j) {
      const double inv = w.inv_sw_emb[size_t(j)];
      const float val = float(w.w_emb[size_t(tid) * w.d + j]) * float(inv);
      x[size_t(i) * w.d + j] = to_q12(val);
    }
  }
  {
    std::vector<int32_t> y(size_t(w.t) * w.d);
    residual_rows(dev, glue, x.data(), w.pos.data(), w.t, w.d, y.data());
    x.swap(y);
  }

  std::vector<int32_t> y(size_t(w.t) * w.d);
  for (int li = 0; li < w.layers; ++li) {
    block_forward(dev, w.blocks[li], w.t, w.d, w.mlp, glue, x.data(), y.data(), true);
    x.swap(y);
  }

  matmul_q12(dev, x.data(), w.t, w.d, w.w_lm.data(), w.vocab, float(w.scale_lm_act),
             w.inv_sw_lm.data(), logits_q12);
}

}  // namespace npukit

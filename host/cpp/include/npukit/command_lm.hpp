#pragma once

#include "npukit/vit.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace npukit {

struct CommandLmSample {
  std::vector<int32_t> input_ids;   // [T]
  std::vector<int32_t> target_ids;  // [T]
};

struct CommandLmWeights {
  int t{32}, d{32}, mlp{64}, layers{6}, vocab{42};
  int pad_id{0};
  std::vector<int32_t> pos;       // Q12 [T*D]
  std::vector<int8_t> w_emb;      // [V*D]
  std::vector<double> inv_sw_emb; // [D]
  double scale_emb_act{64.0};
  std::vector<VitBlockWeights> blocks;
  std::vector<int8_t> w_lm;  // [D*V]
  double scale_lm_act{64.0};
  std::vector<double> inv_sw_lm;  // [V]
  std::vector<CommandLmSample> samples;
};

CommandLmWeights load_command_lm_bin(const std::string& path);

// token_ids [T] → logits_q12 [T*V]
void command_lm_forward(Device* dev, const CommandLmWeights& w, const int32_t* token_ids,
                        int32_t* logits_q12, GlueMode glue = GlueMode::Float);

}  // namespace npukit

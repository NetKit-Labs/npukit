#pragma once
// NpuKit AXI-Lite + AXI DMA register map (matches host/npukit_matmul.py / RTL).

#include <cstdint>

namespace npukit {

constexpr uint32_t NPU_BASE = 0x43C00000u;
constexpr uint32_t NPU_SPAN = 0x1000u;
constexpr uint32_t DMA_BASE = 0x40400000u;
constexpr uint32_t DMA_SPAN = 0x10000u;

constexpr int TILE = 8;
constexpr int TILE_ELEMS = TILE * TILE;           // 64
constexpr int A_WORDS = TILE_ELEMS / 4;           // 16 uint32
constexpr int B_WORDS = TILE_ELEMS / 4;           // 16
constexpr int AB_WORDS = A_WORDS + B_WORDS;       // 32
constexpr int C_WORDS = TILE_ELEMS;               // 64 int32
constexpr int MAX_LEN = 16;

constexpr uint32_t REG_ID = 0x000;
constexpr uint32_t REG_VERSION = 0x004;
constexpr uint32_t REG_STATUS = 0x008;
constexpr uint32_t REG_CTRL = 0x00C;
constexpr uint32_t REG_N = 0x010;
constexpr uint32_t REG_FEATURES = 0x014;
constexpr uint32_t REG_GLUE_CTRL = 0x018;
constexpr uint32_t REG_GLUE_LEN = 0x01C;
constexpr uint32_t REG_GLUE_PARAM = 0x020;
constexpr uint32_t REG_GLUE_COUNT = 0x024;
constexpr uint32_t REG_LOAD_CFG = 0x028;

constexpr uint32_t OFF_A = 0x100;
constexpr uint32_t OFF_B = 0x200;
constexpr uint32_t OFF_C = 0x400;
constexpr uint32_t OFF_GLUE_X = 0x500;
constexpr uint32_t OFF_GLUE_Y = 0x600;
constexpr uint32_t OFF_GLUE_OUT = 0x700;
constexpr uint32_t OFF_GLUE_GAMMA = 0x800;

constexpr uint32_t CTRL_START = 0x1;
constexpr uint32_t CTRL_CLEAR = 0x2;
constexpr uint32_t CTRL_TX_ARM = 0x4;

constexpr uint32_t STATUS_BUSY = 0x1;
constexpr uint32_t STATUS_GEMM_DONE = 0x2;
constexpr uint32_t STATUS_GLUE_DONE = 0x10;

constexpr uint32_t FEAT_GEMM = 0x1;
constexpr uint32_t FEAT_GLUE = 0x2;
constexpr uint32_t FEAT_WS = 0x4;   // weight-stationary: A-only / B-only AXIS loads
constexpr uint32_t FEAT_PP = 0x8;   // dual-A ping-pong (shadow fill while busy)
constexpr uint32_t ID_MAGIC = 0x4E50554Bu;  // "NPUK"

constexpr uint32_t LOAD_AB = 0;  // 32 words A|B (legacy)
constexpr uint32_t LOAD_A = 1;   // 16 words A-only
constexpr uint32_t LOAD_B = 2;   // 16 words B-only

constexpr uint32_t STATUS_SHADOW_A = 0x20;

constexpr uint32_t OP_RESIDUAL = 0x1;
constexpr uint32_t OP_GELU = 0x2;
constexpr uint32_t OP_RMSNORM = 0x3;
constexpr uint32_t OP_SOFTMAX = 0x4;

// AXI DMA (simple mode) — PG021
constexpr uint32_t MM2S_DMACR = 0x00;
constexpr uint32_t MM2S_DMASR = 0x04;
constexpr uint32_t MM2S_SA = 0x18;
constexpr uint32_t MM2S_LENGTH = 0x28;
constexpr uint32_t S2MM_DMACR = 0x30;
constexpr uint32_t S2MM_DMASR = 0x34;
constexpr uint32_t S2MM_DA = 0x48;
constexpr uint32_t S2MM_LENGTH = 0x58;

constexpr uint32_t DMACR_RS = 0x1;
constexpr uint32_t DMACR_RESET = 0x4;
constexpr uint32_t DMASR_IDLE = 0x2;
constexpr uint32_t DMASR_IOC_IRQ = 0x1000;
constexpr uint32_t DMASR_ERR_IRQ = 0x4000;

constexpr int Q12 = 12;
constexpr int ONE_Q12 = 1 << Q12;
constexpr int Q16 = 16;
constexpr int ONE_Q16 = 1 << Q16;

}  // namespace npukit

#pragma once

#include "npukit/regs.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace npukit {

enum class DmaBackend {
  None,   // MMIO tile load/store
  Xrt,    // modern PYNQ: XRT/zocl CMA BO (no /dev/xlnk)
  Xlnk,   // legacy libcma + /dev/xlnk
};

class Device {
 public:
  // Maps NPU + DMA. Bitstream must already be loaded (e.g. via PYNQ Overlay once).
  // Prefers XRT CMA BO, then legacy xlnk, else MMIO.
  // Layer-resident W (FEAT_WMEM): opt in with NPUKIT_WMEM=1.
  // Tile WS+PP: NPUKIT_WS_PP=1. Default remains legacy A∥B (fastest here).
  explicit Device(bool use_dma = true);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  uint32_t id() const;
  uint32_t version() const;
  uint32_t features() const;

  // int8 MxK @ KxN → int32 MxN (M,K,N multiples of 8).
  void matmul_i8(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n);

  // Transformer glue (vector length 1..MAX_LEN).
  void glue_run(uint32_t opcode, const int32_t* x, int n, int32_t* out,
                const int32_t* y = nullptr, const int32_t* gamma = nullptr,
                int32_t param = 1);

  bool dma_enabled() const { return use_dma_; }
  DmaBackend dma_backend() const { return dma_backend_; }
  const char* dma_backend_name() const;
  bool weight_stationary() const { return (features() & FEAT_WS) != 0; }
  bool ping_pong() const { return (features() & FEAT_PP) != 0; }
  bool weight_bank() const { return (features() & FEAT_WMEM) != 0; }

 private:
  void* map_region(uint32_t phys, size_t span);
  void unmap_region(void* ptr, size_t span);

  uint32_t npu_rd(uint32_t off) const;
  void npu_wr(uint32_t off, uint32_t val);
  uint32_t dma_rd(uint32_t off) const;
  void dma_wr(uint32_t off, uint32_t val);

  void wait_gemm_done();
  void dma_reset();
  bool try_init_dma_xrt(size_t tx_bytes, size_t rx_bytes);
  bool try_init_dma_xlnk(size_t tx_bytes, size_t rx_bytes);

  void set_load_cfg(uint32_t mode);
  void load_tile_ab_dma(const int8_t a[TILE_ELEMS], const int8_t b[TILE_ELEMS]);
  void load_tile_ab_mmio(const int8_t a[TILE_ELEMS], const int8_t b[TILE_ELEMS]);
  void load_b_dma(const int8_t b[TILE_ELEMS]);
  void load_b_mmio(const int8_t b[TILE_ELEMS]);
  void load_a_dma(const int8_t a[TILE_ELEMS], bool wait_idle);
  void load_a_mmio(const int8_t a[TILE_ELEMS]);
  void load_weight_bank_dma(const int8_t* b, int k, int n);
  void wait_mm2s_idle();
  void read_tile_dma(int32_t c[TILE_ELEMS]);
  void read_tile_mmio(int32_t c[TILE_ELEMS]);
  void sync_tx_to_device(int which, size_t nbytes);
  void sync_rx_from_device(size_t nbytes);

  void matmul_i8_legacy(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n);
  void matmul_i8_wmem(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n);

  int fd_mem_{-1};
  volatile uint32_t* npu_{nullptr};
  volatile uint32_t* dma_{nullptr};
  bool use_dma_{false};
  DmaBackend dma_backend_{DmaBackend::None};

  // Pooled CMA: two TX buffers for A ping-pong + one RX for C.
  uint32_t* tx_virt_[2]{nullptr, nullptr};
  uint32_t* rx_virt_{nullptr};
  uint32_t tx_phys_[2]{0, 0};
  uint32_t rx_phys_{0};
  size_t tx_bytes_{0};
  size_t rx_bytes_{0};
  int tx_idx_{0};

  void* xrt_ctx_{nullptr};
};

}  // namespace npukit

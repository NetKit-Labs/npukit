#include "npukit/device.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(NPUKIT_HAS_XRT)
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#endif

#if defined(NPUKIT_HAS_XLNK)
extern "C" {
#include <libxlnk_cma.h>
}
#endif

namespace npukit {
namespace {

inline void pack_i8_tile(const int8_t* tile, uint32_t* words) {
  const auto* u = reinterpret_cast<const uint8_t*>(tile);
  for (int i = 0; i < A_WORDS; ++i) {
    words[i] = uint32_t(u[4 * i]) | (uint32_t(u[4 * i + 1]) << 8) |
               (uint32_t(u[4 * i + 2]) << 16) | (uint32_t(u[4 * i + 3]) << 24);
  }
}

#if defined(NPUKIT_HAS_XRT)
struct XrtCtx {
  xrt::device device;
  xrt::bo tx0;
  xrt::bo tx1;
  xrt::bo rx;
  XrtCtx(xrt::device d, size_t tx_n, size_t rx_n)
      : device(std::move(d)),
        tx0(device, tx_n, xrt::bo::flags::normal, 0),
        tx1(device, tx_n, xrt::bo::flags::normal, 0),
        rx(device, rx_n, xrt::bo::flags::normal, 0) {}
  xrt::bo& tx(int i) { return i ? tx1 : tx0; }
};
#endif

bool file_exists(const char* path) {
  struct stat st {};
  return ::stat(path, &st) == 0;
}

}  // namespace

const char* Device::dma_backend_name() const {
  switch (dma_backend_) {
    case DmaBackend::Xrt:
      return "xrt-cma";
    case DmaBackend::Xlnk:
      return "xlnk-cma";
    default:
      return "mmio-fallback";
  }
}

bool Device::try_init_dma_xrt(size_t tx_bytes, size_t rx_bytes) {
#if defined(NPUKIT_HAS_XRT)
  try {
    const size_t tx_n = tx_bytes < 4096 ? 4096 : tx_bytes;
    const size_t rx_n = rx_bytes < 4096 ? 4096 : rx_bytes;
    auto* ctx = new XrtCtx(xrt::device(0), tx_n, rx_n);
    tx_virt_[0] = ctx->tx0.map<uint32_t*>();
    tx_virt_[1] = ctx->tx1.map<uint32_t*>();
    rx_virt_ = ctx->rx.map<uint32_t*>();
    tx_phys_[0] = static_cast<uint32_t>(ctx->tx0.address());
    tx_phys_[1] = static_cast<uint32_t>(ctx->tx1.address());
    rx_phys_ = static_cast<uint32_t>(ctx->rx.address());
    if (!tx_virt_[0] || !tx_virt_[1] || !rx_virt_ || !tx_phys_[0] || !tx_phys_[1] ||
        !rx_phys_) {
      delete ctx;
      tx_virt_[0] = tx_virt_[1] = rx_virt_ = nullptr;
      return false;
    }
    xrt_ctx_ = ctx;
    tx_bytes_ = tx_n;
    rx_bytes_ = rx_n;
    dma_backend_ = DmaBackend::Xrt;
    return true;
  } catch (...) {
    xrt_ctx_ = nullptr;
    tx_virt_[0] = tx_virt_[1] = rx_virt_ = nullptr;
    return false;
  }
#else
  (void)tx_bytes;
  (void)rx_bytes;
  return false;
#endif
}

bool Device::try_init_dma_xlnk(size_t tx_bytes, size_t rx_bytes) {
#if defined(NPUKIT_HAS_XLNK)
  if (!file_exists("/dev/xlnk")) return false;
  auto* tx0 = cma_alloc(static_cast<uint32_t>(tx_bytes), 0);
  auto* tx1 = cma_alloc(static_cast<uint32_t>(tx_bytes), 0);
  auto* rx = cma_alloc(static_cast<uint32_t>(rx_bytes), 0);
  auto ok = [](void* p) { return p && p != reinterpret_cast<void*>(-1); };
  if (!ok(tx0) || !ok(tx1) || !ok(rx)) {
    if (ok(tx0)) cma_free(tx0);
    if (ok(tx1)) cma_free(tx1);
    if (ok(rx)) cma_free(rx);
    return false;
  }
  tx_virt_[0] = static_cast<uint32_t*>(tx0);
  tx_virt_[1] = static_cast<uint32_t*>(tx1);
  rx_virt_ = static_cast<uint32_t*>(rx);
  tx_phys_[0] = static_cast<uint32_t>(cma_get_phy_addr(tx_virt_[0]));
  tx_phys_[1] = static_cast<uint32_t>(cma_get_phy_addr(tx_virt_[1]));
  rx_phys_ = static_cast<uint32_t>(cma_get_phy_addr(rx_virt_));
  tx_bytes_ = tx_bytes;
  rx_bytes_ = rx_bytes;
  dma_backend_ = DmaBackend::Xlnk;
  return true;
#else
  (void)tx_bytes;
  (void)rx_bytes;
  return false;
#endif
}

void Device::sync_tx_to_device(int which, size_t nbytes) {
#if defined(NPUKIT_HAS_XRT)
  if (dma_backend_ == DmaBackend::Xrt) {
    static_cast<XrtCtx*>(xrt_ctx_)->tx(which).sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0);
    return;
  }
#endif
#if defined(NPUKIT_HAS_XLNK)
  if (dma_backend_ == DmaBackend::Xlnk) {
    cma_flush_cache(tx_virt_[which], tx_phys_[which], int(nbytes));
    return;
  }
#endif
  (void)which;
  (void)nbytes;
}

void Device::sync_rx_from_device(size_t nbytes) {
#if defined(NPUKIT_HAS_XRT)
  if (dma_backend_ == DmaBackend::Xrt) {
    static_cast<XrtCtx*>(xrt_ctx_)->rx.sync(XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0);
    return;
  }
#endif
#if defined(NPUKIT_HAS_XLNK)
  if (dma_backend_ == DmaBackend::Xlnk) {
    cma_invalidate_cache(rx_virt_, rx_phys_, int(nbytes));
    return;
  }
#endif
  (void)nbytes;
}

Device::Device(bool use_dma) {
  fd_mem_ = ::open("/dev/mem", O_RDWR | O_SYNC);
  if (fd_mem_ < 0) throw std::runtime_error("open /dev/mem failed (need root/sudo)");
  npu_ = static_cast<volatile uint32_t*>(map_region(NPU_BASE, NPU_SPAN));
  if (id() != ID_MAGIC) throw std::runtime_error("NPU ID mismatch — is npukit.bit loaded?");
  if (!(features() & FEAT_GEMM)) throw std::runtime_error("FEATURES missing GEMM bit");

  use_dma_ = use_dma;
  if (use_dma_) {
    dma_ = static_cast<volatile uint32_t*>(map_region(DMA_BASE, DMA_SPAN));
    // TX must fit full weight bank (W_CAP int8) for LOAD_W.
    const size_t tx_bytes = std::max(size_t(AB_WORDS * sizeof(uint32_t)), size_t(W_CAP));
    const size_t rx_bytes = C_WORDS * sizeof(uint32_t);
    const bool ok = try_init_dma_xrt(tx_bytes, rx_bytes) || try_init_dma_xlnk(tx_bytes, rx_bytes);
    if (!ok) {
      use_dma_ = false;
      dma_backend_ = DmaBackend::None;
      unmap_region(const_cast<uint32_t*>(dma_), DMA_SPAN);
      dma_ = nullptr;
    } else {
      dma_reset();
    }
  }
}

Device::~Device() {
#if defined(NPUKIT_HAS_XRT)
  if (dma_backend_ == DmaBackend::Xrt && xrt_ctx_) {
    delete static_cast<XrtCtx*>(xrt_ctx_);
    xrt_ctx_ = nullptr;
    tx_virt_[0] = tx_virt_[1] = rx_virt_ = nullptr;
  }
#endif
#if defined(NPUKIT_HAS_XLNK)
  if (dma_backend_ == DmaBackend::Xlnk) {
    if (tx_virt_[0]) cma_free(tx_virt_[0]);
    if (tx_virt_[1]) cma_free(tx_virt_[1]);
    if (rx_virt_) cma_free(rx_virt_);
    tx_virt_[0] = tx_virt_[1] = rx_virt_ = nullptr;
  }
#endif
  if (dma_) unmap_region(const_cast<uint32_t*>(dma_), DMA_SPAN);
  if (npu_) unmap_region(const_cast<uint32_t*>(npu_), NPU_SPAN);
  if (fd_mem_ >= 0) ::close(fd_mem_);
}

void* Device::map_region(uint32_t phys, size_t span) {
  void* p = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem_,
                   static_cast<off_t>(phys));
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed for 0x" + std::to_string(phys));
  return p;
}

void Device::unmap_region(void* ptr, size_t span) {
  if (ptr && ptr != MAP_FAILED) ::munmap(ptr, span);
}

uint32_t Device::npu_rd(uint32_t off) const { return npu_[off / 4]; }
void Device::npu_wr(uint32_t off, uint32_t val) { npu_[off / 4] = val; }
uint32_t Device::dma_rd(uint32_t off) const { return dma_[off / 4]; }
void Device::dma_wr(uint32_t off, uint32_t val) { dma_[off / 4] = val; }

uint32_t Device::id() const { return npu_rd(REG_ID); }
uint32_t Device::version() const { return npu_rd(REG_VERSION); }
uint32_t Device::features() const { return npu_rd(REG_FEATURES); }

void Device::wait_gemm_done() {
  for (int i = 0; i < 2'000'000; ++i) {
    if (npu_rd(REG_STATUS) & STATUS_GEMM_DONE) return;
  }
  throw std::runtime_error("GEMM timeout");
}

void Device::wait_mm2s_idle() {
  for (int i = 0; i < 2'000'000; ++i) {
    if (dma_rd(MM2S_DMASR) & DMASR_IDLE) return;
  }
  throw std::runtime_error("MM2S DMA timeout");
}

void Device::dma_reset() {
  dma_wr(MM2S_DMACR, DMACR_RESET);
  dma_wr(S2MM_DMACR, DMACR_RESET);
  for (int i = 0; i < 100000; ++i) {
    if (!(dma_rd(MM2S_DMACR) & DMACR_RESET) && !(dma_rd(S2MM_DMACR) & DMACR_RESET)) break;
  }
  dma_wr(MM2S_DMACR, DMACR_RS);
  dma_wr(S2MM_DMACR, DMACR_RS);
}

void Device::set_load_cfg(uint32_t mode) {
  if (version() >= 0x301u) npu_wr(REG_LOAD_CFG, mode & 0x3u);
}

void Device::load_weight_bank_dma(const int8_t* b, int k, int n) {
  if (k < 1 || n < 1 || size_t(k) * size_t(n) > size_t(W_CAP))
    throw std::invalid_argument("weight bank K*N out of range");
  npu_wr(REG_W_SHAPE, (uint32_t(n) << 16) | uint32_t(k));
  set_load_cfg(LOAD_W);
  const size_t nbytes = size_t(k) * size_t(n);
  const size_t nwords = (nbytes + 3) / 4;
  auto* dst = reinterpret_cast<uint8_t*>(tx_virt_[tx_idx_]);
  std::memcpy(dst, b, nbytes);
  // Pad last word so DMA length is whole uint32s.
  for (size_t i = nbytes; i < nwords * 4; ++i) dst[i] = 0;
  sync_tx_to_device(tx_idx_, nwords * sizeof(uint32_t));
  dma_wr(MM2S_DMASR, dma_rd(MM2S_DMASR));
  dma_wr(MM2S_SA, tx_phys_[tx_idx_]);
  dma_wr(MM2S_LENGTH, static_cast<uint32_t>(nwords * sizeof(uint32_t)));
  wait_mm2s_idle();
  tx_idx_ ^= 1;
}

void Device::load_tile_ab_mmio(const int8_t a[TILE_ELEMS], const int8_t b[TILE_ELEMS]) {
  uint32_t wa[A_WORDS], wb[B_WORDS];
  pack_i8_tile(a, wa);
  pack_i8_tile(b, wb);
  for (int i = 0; i < A_WORDS; ++i) npu_wr(OFF_A + 4 * i, wa[i]);
  for (int i = 0; i < B_WORDS; ++i) npu_wr(OFF_B + 4 * i, wb[i]);
}

void Device::load_b_mmio(const int8_t b[TILE_ELEMS]) {
  uint32_t wb[B_WORDS];
  pack_i8_tile(b, wb);
  for (int i = 0; i < B_WORDS; ++i) npu_wr(OFF_B + 4 * i, wb[i]);
}

void Device::load_a_mmio(const int8_t a[TILE_ELEMS]) {
  uint32_t wa[A_WORDS];
  pack_i8_tile(a, wa);
  for (int i = 0; i < A_WORDS; ++i) npu_wr(OFF_A + 4 * i, wa[i]);
}

void Device::load_tile_ab_dma(const int8_t a[TILE_ELEMS], const int8_t b[TILE_ELEMS]) {
  set_load_cfg(LOAD_AB);
  pack_i8_tile(a, tx_virt_[tx_idx_]);
  pack_i8_tile(b, tx_virt_[tx_idx_] + A_WORDS);
  sync_tx_to_device(tx_idx_, AB_WORDS * sizeof(uint32_t));
  dma_wr(MM2S_DMASR, dma_rd(MM2S_DMASR));
  dma_wr(MM2S_SA, tx_phys_[tx_idx_]);
  dma_wr(MM2S_LENGTH, AB_WORDS * sizeof(uint32_t));
  wait_mm2s_idle();
  tx_idx_ ^= 1;
}

void Device::load_b_dma(const int8_t b[TILE_ELEMS]) {
  set_load_cfg(LOAD_B);
  pack_i8_tile(b, tx_virt_[tx_idx_]);
  sync_tx_to_device(tx_idx_, B_WORDS * sizeof(uint32_t));
  dma_wr(MM2S_DMASR, dma_rd(MM2S_DMASR));
  dma_wr(MM2S_SA, tx_phys_[tx_idx_]);
  dma_wr(MM2S_LENGTH, B_WORDS * sizeof(uint32_t));
  wait_mm2s_idle();
  tx_idx_ ^= 1;
}

void Device::load_a_dma(const int8_t a[TILE_ELEMS], bool wait_idle) {
  set_load_cfg(LOAD_A);
  pack_i8_tile(a, tx_virt_[tx_idx_]);
  sync_tx_to_device(tx_idx_, A_WORDS * sizeof(uint32_t));
  dma_wr(MM2S_DMASR, dma_rd(MM2S_DMASR));
  dma_wr(MM2S_SA, tx_phys_[tx_idx_]);
  dma_wr(MM2S_LENGTH, A_WORDS * sizeof(uint32_t));
  if (wait_idle) wait_mm2s_idle();
  tx_idx_ ^= 1;
}

void Device::read_tile_mmio(int32_t c[TILE_ELEMS]) {
  for (int i = 0; i < C_WORDS; ++i) c[i] = static_cast<int32_t>(npu_rd(OFF_C + 4 * i));
}

void Device::read_tile_dma(int32_t c[TILE_ELEMS]) {
  dma_wr(S2MM_DMASR, dma_rd(S2MM_DMASR));
  dma_wr(S2MM_DA, rx_phys_);
  dma_wr(S2MM_LENGTH, C_WORDS * sizeof(uint32_t));
  npu_wr(REG_CTRL, CTRL_TX_ARM);
  for (int i = 0; i < 2'000'000; ++i) {
    if (dma_rd(S2MM_DMASR) & DMASR_IDLE) break;
    if (i + 1 == 2'000'000) throw std::runtime_error("S2MM DMA timeout");
  }
  sync_rx_from_device(C_WORDS * sizeof(uint32_t));
  std::memcpy(c, rx_virt_, C_WORDS * sizeof(int32_t));
}

void Device::matmul_i8_legacy(const int8_t* a, const int8_t* b, int32_t* c, int m, int k,
                              int n) {
  std::memset(c, 0, size_t(m) * size_t(n) * sizeof(int32_t));
  int8_t at[TILE_ELEMS], bt[TILE_ELEMS];
  int32_t ct[TILE_ELEMS];
  for (int i0 = 0; i0 < m; i0 += TILE) {
    for (int j0 = 0; j0 < n; j0 += TILE) {
      bool first = true;
      for (int k0 = 0; k0 < k; k0 += TILE) {
        for (int r = 0; r < TILE; ++r) {
          std::memcpy(at + r * TILE, a + (i0 + r) * k + k0, TILE);
          std::memcpy(bt + r * TILE, b + (k0 + r) * n + j0, TILE);
        }
        if (use_dma_)
          load_tile_ab_dma(at, bt);
        else
          load_tile_ab_mmio(at, bt);
        npu_wr(REG_CTRL, first ? (CTRL_CLEAR | CTRL_START) : CTRL_START);
        wait_gemm_done();
        first = false;
      }
      if (use_dma_)
        read_tile_dma(ct);
      else
        read_tile_mmio(ct);
      for (int r = 0; r < TILE; ++r)
        std::memcpy(c + (i0 + r) * n + j0, ct + r * TILE, TILE * sizeof(int32_t));
    }
  }
}

void Device::matmul_i8_wmem(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n) {
  load_weight_bank_dma(b, k, n);
  std::memset(c, 0, size_t(m) * size_t(n) * sizeof(int32_t));
  int8_t at[TILE_ELEMS];
  int32_t ct[TILE_ELEMS];

  for (int i0 = 0; i0 < m; i0 += TILE) {
    for (int j0 = 0; j0 < n; j0 += TILE) {
      bool first = true;
      for (int k0 = 0; k0 < k; k0 += TILE) {
        npu_wr(REG_TILE_KJ, (uint32_t(j0 / TILE) << 8) | uint32_t(k0 / TILE));
        for (int r = 0; r < TILE; ++r)
          std::memcpy(at + r * TILE, a + (i0 + r) * k + k0, TILE);
        load_a_dma(at, true);
        const uint32_t ctrl =
            (first ? (CTRL_CLEAR | CTRL_START) : CTRL_START) | CTRL_USE_WMEM;
        npu_wr(REG_CTRL, ctrl);
        wait_gemm_done();
        first = false;
      }
      read_tile_dma(ct);
      for (int r = 0; r < TILE; ++r)
        std::memcpy(c + (i0 + r) * n + j0, ct + r * TILE, TILE * sizeof(int32_t));
    }
  }
}

void Device::matmul_i8(const int8_t* a, const int8_t* b, int32_t* c, int m, int k, int n) {
  if (m % TILE || k % TILE || n % TILE)
    throw std::invalid_argument("M,K,N must be multiples of 8");

  // Layer-resident W: one DMA of B, then A-only kicks. Opt in with NPUKIT_WMEM=1.
  // On this 8×8 @ 100 MHz tiny-ViT, BFILL + A-only DMA loses to A∥B (~10 vs ~9.8 ms).
  const char* wmem_env = std::getenv("NPUKIT_WMEM");
  const bool force_wmem = wmem_env && wmem_env[0] == '1';
  const bool use_wmem = force_wmem && weight_bank() && use_dma_ &&
                        (size_t(k) * size_t(n) <= size_t(W_CAP));
  if (use_wmem) {
    matmul_i8_wmem(a, b, c, m, k, n);
    return;
  }

  // WS+PP is available when FEAT_WS|PP, but on this tiny 8×8 @ 100 MHz the
  // extra MM2S setups usually lose to a single A∥B transfer. Opt in with
  // NPUKIT_WS_PP=1 (and K>8 so a shadow prefetch exists).
  const char* ws_env = std::getenv("NPUKIT_WS_PP");
  const bool force_ws_pp = ws_env && ws_env[0] == '1';
  const bool use_ws_pp =
      force_ws_pp && weight_stationary() && ping_pong() && use_dma_ && (k > TILE);

  if (!use_ws_pp) {
    matmul_i8_legacy(a, b, c, m, k, n);
    return;
  }

  std::memset(c, 0, size_t(m) * size_t(n) * sizeof(int32_t));
  int8_t at[TILE_ELEMS], bt[TILE_ELEMS], at_next[TILE_ELEMS];
  int32_t ct[TILE_ELEMS];

  for (int i0 = 0; i0 < m; i0 += TILE) {
    for (int j0 = 0; j0 < n; j0 += TILE) {
      bool first = true;
      bool a_in_shadow = false;
      for (int k0 = 0; k0 < k; k0 += TILE) {
        for (int r = 0; r < TILE; ++r)
          std::memcpy(bt + r * TILE, b + (k0 + r) * n + j0, TILE);
        load_b_dma(bt);

        if (!a_in_shadow) {
          for (int r = 0; r < TILE; ++r)
            std::memcpy(at + r * TILE, a + (i0 + r) * k + k0, TILE);
          load_a_dma(at, true);
        }
        a_in_shadow = false;

        const int k_next = k0 + TILE;
        const bool prefetch = (k_next < k);
        if (prefetch) {
          for (int r = 0; r < TILE; ++r)
            std::memcpy(at_next + r * TILE, a + (i0 + r) * k + k_next, TILE);
        }

        npu_wr(REG_CTRL, first ? (CTRL_CLEAR | CTRL_START) : CTRL_START);
        if (prefetch) {
          load_a_dma(at_next, false);
          a_in_shadow = true;
        }
        wait_gemm_done();
        if (prefetch) wait_mm2s_idle();
        first = false;
      }
      read_tile_dma(ct);
      for (int r = 0; r < TILE; ++r)
        std::memcpy(c + (i0 + r) * n + j0, ct + r * TILE, TILE * sizeof(int32_t));
    }
  }
}

void Device::glue_run(uint32_t opcode, const int32_t* x, int n, int32_t* out, const int32_t* y,
                      const int32_t* gamma, int32_t param) {
  if (n < 1 || n > MAX_LEN) throw std::invalid_argument("glue length");
  if (!(features() & FEAT_GLUE)) throw std::runtime_error("no glue in FEATURES");

  npu_wr(REG_GLUE_LEN, static_cast<uint32_t>(n));
  npu_wr(REG_GLUE_PARAM, static_cast<uint32_t>(param));
  for (int i = 0; i < n; ++i) npu_wr(OFF_GLUE_X + 4 * i, static_cast<uint32_t>(x[i]));
  if (y) {
    for (int i = 0; i < n; ++i) npu_wr(OFF_GLUE_Y + 4 * i, static_cast<uint32_t>(y[i]));
  }
  if (gamma) {
    for (int i = 0; i < n; ++i)
      npu_wr(OFF_GLUE_GAMMA + 4 * i, static_cast<uint32_t>(gamma[i]));
  }
  const uint32_t before = npu_rd(REG_GLUE_COUNT);
  npu_wr(REG_GLUE_CTRL, ((opcode & 0xF) << 4) | 0x1);
  for (int i = 0; i < 2'000'000; ++i) {
    if (npu_rd(REG_GLUE_COUNT) != before) break;
    if (i + 1 == 2'000'000 && !(npu_rd(REG_STATUS) & STATUS_GLUE_DONE))
      throw std::runtime_error("glue timeout");
  }
  for (int i = 0; i < n; ++i) out[i] = static_cast<int32_t>(npu_rd(OFF_GLUE_OUT + 4 * i));
}

}  // namespace npukit

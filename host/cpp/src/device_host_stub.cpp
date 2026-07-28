// Host-only stub: allows linking npukit_vit --cpu without PYNQ libcma /dev/mem.
#include "npukit/device.hpp"

#include <stdexcept>

namespace npukit {

Device::Device(bool) { throw std::runtime_error("Device unavailable in HOST build; use --cpu"); }
Device::~Device() = default;
uint32_t Device::id() const { return 0; }
uint32_t Device::version() const { return 0; }
uint32_t Device::features() const { return 0; }
const char* Device::dma_backend_name() const { return "none"; }
void Device::matmul_i8(const int8_t*, const int8_t*, int32_t*, int, int, int) {
  throw std::runtime_error("matmul_i8 stub");
}
void Device::glue_run(uint32_t, const int32_t*, int, int32_t*, const int32_t*, const int32_t*,
                      int32_t) {
  throw std::runtime_error("glue_run stub");
}
void* Device::map_region(uint32_t, size_t) { return nullptr; }
void Device::unmap_region(void*, size_t) {}
uint32_t Device::npu_rd(uint32_t) const { return 0; }
void Device::npu_wr(uint32_t, uint32_t) {}
uint32_t Device::dma_rd(uint32_t) const { return 0; }
void Device::dma_wr(uint32_t, uint32_t) {}
void Device::wait_gemm_done() {}
void Device::dma_reset() {}
bool Device::try_init_dma_xrt(size_t, size_t) { return false; }
bool Device::try_init_dma_xlnk(size_t, size_t) { return false; }
void Device::set_load_cfg(uint32_t) {}
void Device::load_tile_ab_dma(const int8_t[TILE_ELEMS], const int8_t[TILE_ELEMS]) {}
void Device::load_tile_ab_mmio(const int8_t[TILE_ELEMS], const int8_t[TILE_ELEMS]) {}
void Device::load_b_dma(const int8_t[TILE_ELEMS]) {}
void Device::load_b_mmio(const int8_t[TILE_ELEMS]) {}
void Device::load_a_dma(const int8_t[TILE_ELEMS], bool) {}
void Device::load_a_mmio(const int8_t[TILE_ELEMS]) {}
void Device::wait_mm2s_idle() {}
void Device::sync_tx_to_device(int, size_t) {}
void Device::sync_rx_from_device(size_t) {}
void Device::read_tile_dma(int32_t[TILE_ELEMS]) {}
void Device::read_tile_mmio(int32_t[TILE_ELEMS]) {}
void Device::matmul_i8_legacy(const int8_t*, const int8_t*, int32_t*, int, int, int) {}

}  // namespace npukit

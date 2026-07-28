# Build on PYNQ-Z2 (ARMv7) with sudo for /dev/mem + CMA.
#   make
#   python3 export_vit_bin.py
#   sudo ./npukit_vit --weights vit_mnist.bin
#
# Host bring-up (no FPGA / no libcma):
#   make HOST=1 npukit_vit
#   ./npukit_vit --cpu --weights vit_mnist.bin

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
LDFLAGS  ?= -lm

# Board build: prefer XRT/zocl CMA (PYNQ 3.x, no /dev/xlnk). Optional legacy xlnk.
ifeq ($(HOST),1)
  DEVICE_SRC := src/device_host_stub.cpp
else
  DEVICE_SRC := src/device.cpp
  # Detect XRT headers on PYNQ images (layout: /usr/include/xrt/xrt/xrt_*.h).
  ifneq ($(wildcard /usr/include/xrt/xrt/xrt_device.h),)
    CXXFLAGS += -DNPUKIT_HAS_XRT -I/usr/include/xrt
    LDFLAGS  += -lxrt_coreutil -lxrt_core
  else ifneq ($(wildcard /usr/include/xrt/xrt_device.h),)
    CXXFLAGS += -DNPUKIT_HAS_XRT -I/usr/include
    LDFLAGS  += -lxrt_coreutil -lxrt_core
  endif
  ifneq ($(wildcard /usr/include/libxlnk_cma.h),)
    CXXFLAGS += -DNPUKIT_HAS_XLNK
    LDFLAGS  += -lcma
  endif
endif

COMMON_SRCS := $(DEVICE_SRC) src/stem.cpp src/vit.cpp
BENCH_SRCS  := $(COMMON_SRCS) src/bench_main.cpp
VIT_SRCS    := $(COMMON_SRCS) src/vit_main.cpp
DSCNN_SRCS  := src/dscnn.cpp src/dscnn_main.cpp
CLM_SRCS    := $(DEVICE_SRC) src/stem.cpp src/vit.cpp src/command_lm.cpp src/command_lm_main.cpp

BENCH_OBJS := $(BENCH_SRCS:.cpp=.o)
VIT_OBJS   := $(VIT_SRCS:.cpp=.o)
DSCNN_OBJS := $(DSCNN_SRCS:.cpp=.o)
CLM_OBJS   := $(CLM_SRCS:.cpp=.o)

.PHONY: all clean vit_bin dscnn_bin command_lm_bin

all: npukit_bench npukit_vit npukit_dscnn npukit_command_lm

npukit_bench: $(BENCH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

npukit_vit: $(VIT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

npukit_dscnn: $(DSCNN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

npukit_command_lm: $(CLM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

vit_bin:
	python3 export_vit_bin.py --out vit_mnist.bin

dscnn_bin:
	python3 export_dscnn_bin.py --out dscnn_mnist.bin

command_lm_bin:
	python3 export_command_lm_bin.py --out command_lm.bin

clean:
	rm -f src/*.o npukit_bench npukit_vit npukit_dscnn npukit_command_lm

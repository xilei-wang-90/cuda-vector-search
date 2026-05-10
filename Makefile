NVCC      ?= nvcc
CUDA_ARCH ?= 86          # RTX 30xx (Ampere). Override e.g. CUDA_ARCH=89 for RTX 40xx.
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) --compiler-options -Wall,-Wextra

BUILD_DIR := build
SRC_DIR   := src

BINS := $(BUILD_DIR)/check_roundtrip

.PHONY: all clean
all: $(BINS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/check_roundtrip: $(SRC_DIR)/check_roundtrip.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

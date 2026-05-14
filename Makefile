NVCC      ?= nvcc
CUDA_ARCH ?= 86          # RTX 30xx (Ampere). Override e.g. CUDA_ARCH=89 for RTX 40xx.
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) --compiler-options -Wall,-Wextra

BUILD_DIR := build
SRC_DIR   := src

BINS := $(BUILD_DIR)/check_roundtrip $(BUILD_DIR)/compute_distances $(BUILD_DIR)/check_topk

.PHONY: all clean test
all: $(BINS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/check_roundtrip: $(SRC_DIR)/diagnostics/check_roundtrip.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/compute_distances: $(SRC_DIR)/compute_distances.cu $(SRC_DIR)/distance_kernel.cu $(SRC_DIR)/distance_kernel.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $(SRC_DIR)/compute_distances.cu $(SRC_DIR)/distance_kernel.cu -o $@

$(BUILD_DIR)/check_topk: $(SRC_DIR)/diagnostics/check_topk.cu $(SRC_DIR)/distance_kernel.cu $(SRC_DIR)/distance_kernel.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $(SRC_DIR)/diagnostics/check_topk.cu $(SRC_DIR)/distance_kernel.cu -o $@

test: $(BUILD_DIR)/check_roundtrip $(BUILD_DIR)/check_topk
	./$(BUILD_DIR)/check_roundtrip
	./$(BUILD_DIR)/check_topk

clean:
	rm -rf $(BUILD_DIR)

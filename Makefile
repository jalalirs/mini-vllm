# Mini-vLLM Build System
# Compiles CUDA kernels for H100 (SM90)
#
# Usage:
#   make          - Build CUDA kernels
#   make clean    - Clean build artifacts
#   make install  - Install Python package
#   make test     - Verify build

.PHONY: all build clean install test

CUDA_ARCH ?= 90
BUILD_DIR ?= build
NPROC ?= $(shell nproc 2>/dev/null || echo 8)

all: build

# =============================================================================
# Build CUDA kernels
# =============================================================================
build:
	@echo "======================================="
	@echo "Building CUDA kernels for SM$(CUDA_ARCH)"
	@echo "======================================="
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	@cd $(BUILD_DIR) && make VERBOSE=1 -j$(NPROC)
	@echo ""
	@echo "Build complete. Library:"
	@ls -la mini_vllm/_C*.so 2>/dev/null || echo "WARNING: .so not found"

# =============================================================================
# Build with debug symbols (for profiling)
# =============================================================================
debug:
	@echo "Building with debug symbols..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DCMAKE_CUDA_FLAGS="-g -lineinfo"
	@cd $(BUILD_DIR) && make VERBOSE=1 -j$(NPROC)

# =============================================================================
# Clean build artifacts
# =============================================================================
clean:
	rm -rf $(BUILD_DIR)
	rm -f mini_vllm/_C*.so
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Install Python package (after CUDA build)
# =============================================================================
install:
	pip install --no-build-isolation -e .

# =============================================================================
# Full build + install
# =============================================================================
all-install: build install

# =============================================================================
# Test that CUDA ops load correctly
# =============================================================================
test:
	@echo "Testing CUDA ops..."
	@python3 -c "\
from mini_vllm.ops import cuda_ops_available, get_library_path; \
print(f'CUDA ops available: {cuda_ops_available()}'); \
print(f'Library path: {get_library_path()}'); \
assert cuda_ops_available(), 'CUDA ops not loaded!'; \
print('SUCCESS: CUDA ops loaded correctly')"

# =============================================================================
# Show build configuration
# =============================================================================
info:
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo "BUILD_DIR: $(BUILD_DIR)"
	@echo "NPROC: $(NPROC)"
	@which nvcc && nvcc --version || echo "nvcc not found"
	@python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not found"


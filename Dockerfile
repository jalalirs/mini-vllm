# Mini-vLLM: H100-optimized inference with explicit CUDA kernel compilation
#
# Build:   docker build -t mini-vllm:latest .
# Run TP8: docker run --gpus all -p 8000:8000 -v /models:/models mini-vllm:latest \
#            --model /models/gpt-oss-120b --tensor-parallel-size 8

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.4.0

# =============================================================================
# Build Stage - Explicit CUDA Kernel Compilation
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ARG TORCH_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Python venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (needed for headers during CUDA compilation)
RUN pip install --upgrade pip wheel && \
    pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu124

# Copy source
WORKDIR /build
COPY . /build/

# =============================================================================
# EXPLICIT CUDA COMPILATION
# =============================================================================
RUN echo "=======================================" && \
    echo "COMPILING CUDA KERNELS FOR H100 (SM90)" && \
    echo "=======================================" && \
    echo ""

# Create build directory
RUN mkdir -p /build/build

# Run CMake configuration
RUN cd /build/build && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPYTHON_EXECUTABLE=$(which python3) \
    2>&1 | tee /build/cmake_config.log

# Compile CUDA kernels with verbose output
RUN cd /build/build && make VERBOSE=1 -j$(nproc) 2>&1 | tee /build/cuda_build.log

# Verify the compiled library
RUN echo "" && \
    echo "=======================================" && \
    echo "CUDA COMPILATION COMPLETE" && \
    echo "=======================================" && \
    ls -la /build/mini_vllm/_C*.so && \
    echo "" && \
    echo "Checking symbols:" && \
    nm -D /build/mini_vllm/_C*.so | grep -E "(paged_attention|rms_norm|silu_and_mul)" | head -20

# =============================================================================
# Install Python dependencies (package installed via copy, not pip)
# =============================================================================
RUN pip install \
        transformers>=4.40.0 \
        safetensors>=0.4.0 \
        fastapi>=0.100.0 \
        uvicorn>=0.23.0 \
        pydantic>=2.0.0


# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy venv with compiled CUDA library
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy build logs for debugging
COPY --from=builder /build/cmake_config.log /app/logs/
COPY --from=builder /build/cuda_build.log /app/logs/

WORKDIR /app

# =============================================================================
# H100 Runtime Optimizations
# =============================================================================
ENV CUDA_VISIBLE_DEVICES=all
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Verify CUDA ops loaded
# =============================================================================
RUN python3 -c "from mini_vllm.ops import cuda_ops_available; print(f'CUDA ops: {cuda_ops_available()}')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "-m", "mini_vllm"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

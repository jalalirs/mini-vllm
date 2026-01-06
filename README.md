# Mini-vLLM

Minimal LLM inference for GPT-OSS on H100 with full CUDA kernel source.

**Build requires H100 GPU.** Use the deployment scripts to build on cluster.

## Build

### On GPU Node (via deployment scripts)

```bash
# Build Docker image on cluster GPU pod, push to ECR
./deploy/scripts/build.sh latest
```

This runs `cmake` + `make` inside the container to explicitly compile CUDA for SM90.

### Manual Build (if you have H100 locally)

```bash
# 1. Compile CUDA kernels
make

# 2. Install Python package
make install

# 3. Verify
make test
```

## CUDA Kernels

All kernels are compiled for H100 (SM90). Build logs show every nvcc invocation.

```
mini_vllm/csrc/
├── torch_bindings.cpp           # PyTorch op registration
├── attention/
│   ├── paged_attention_v1.cu    # Single-pass paged attention
│   ├── paged_attention_v2.cu    # Two-pass (memory efficient)
│   └── merge_attn_states.cu     # Split-KV merging
├── layernorm_kernels.cu         # RMSNorm, fused add+norm
├── activation_kernels.cu        # SiLU, GELU, SwiGLU
├── pos_encoding_kernels.cu      # RoPE
├── cache_kernels.cu             # KV cache ops
└── moe/
    ├── topk_softmax_kernels.cu  # Expert routing
    ├── moe_align_sum_kernels.cu # Token alignment
    └── grouped_topk_kernels.cu  # Grouped routing
```

## Run

```bash
# TP8
docker run --gpus all -p 8000:8000 -v /models:/models mini-vllm:latest \
    --model /models/gpt-oss-120b --tensor-parallel-size 8
```

Or via Kubernetes:

```bash
./deploy/scripts/deploy.sh tp8
```

## Profiling

Build logs are saved in `/app/logs/` inside the container:
- `cmake_config.log` - CMake configuration
- `cuda_build.log` - Full nvcc output

Profile with Nsight:

```bash
nsys profile -o profile python -m mini_vllm --model /models/gpt-oss-120b
```

## Architecture

```
mini_vllm/
├── _C.so                        # Compiled CUDA library (output of cmake)
├── ops/                         # Python bindings to _C.so
├── models/gpt_oss.py            # GPT-OSS model
├── layers/                      # Uses CUDA ops when available
├── attention/                   # PagedAttention + FlashAttention
├── engine/                      # LLMEngine
└── csrc/                        # CUDA source (input to cmake)
```

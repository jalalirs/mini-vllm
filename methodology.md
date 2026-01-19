# mini-vLLM Development Methodology

## Overview

mini-vLLM is a minimal fork of vLLM optimized for GPT-OSS models with MXFP4 quantization on H100 GPUs.

## Build & Deploy Process

### 1. Make Code Changes Locally

Edit code in the local repository. When removing functionality:
- Delete unused files/directories
- Trace and remove all code that imports/uses the deleted modules
- Remove the imports, function calls, and dead code paths entirely
- Add comments like `# mini-vLLM: <feature> removed` where code was removed

### 2. Build on AWS

Build happens on a builder pod in the Kubernetes cluster (not locally):

```bash
# Build server image only (fast - uses cached essentials/cuda)
./deploy/scripts/build.sh server

# Build all stages (slow - only when dependencies change)
./deploy/scripts/build.sh all
```

**3-stage Docker build:**
1. `essentials` - Python, PyTorch, dependencies (~8GB, rarely rebuilt)
2. `cuda` - Flash Attention v3 + CUDA kernels (~2GB, rebuild on kernel changes)
3. `server` - Python code + runtime (~10GB, fast rebuild on code changes)

### 3. Deploy to Kubernetes

```bash
# Deploy with tensor parallel = 2
./deploy/scripts/deploy.sh tp2

# Deploy with tensor parallel = 8
./deploy/scripts/deploy.sh tp8

# Delete deployment
./deploy/scripts/deploy.sh tp2 delete
```

### 4. Monitor

```bash
# Watch logs
./deploy/scripts/deploy.sh tp2 logs

# Check status
./deploy/scripts/deploy.sh tp2 status

# Or directly with kubectl
kubectl logs -n hyperpod-ns-inference -l app=mini-vllm-gptoss-tp2 -f
kubectl get pods -n hyperpod-ns-inference -l app=mini-vllm-gptoss-tp2
```

### 5. Verify Health

- Server starts without import errors
- Health checks pass (`/health` endpoint)
- Pod shows `Running` status with `READY 1/1`

### 6. Iterate on Errors

If deployment fails:
- Check logs: `./deploy/scripts/deploy.sh tp2 logs`
- Identify missing imports/modules
- Trace back and remove the code that references them
- Rebuild: `./deploy/scripts/build.sh server`
- Redeploy: `./deploy/scripts/deploy.sh tp2`

### 7. Commit and Push

Once verified working:
```bash
git add -A
git commit -m "Description of changes"
git push
```

## Environment Setup

Before running scripts, ensure AWS credentials are set:

```bash
source deploy/aws/credentials.sh
source deploy/aws/env.sh
```

## Target Configuration

- **Model**: GPT-OSS (`/mnt/fsx/inference/models/gpt-oss-20b`)
- **Platform**: CUDA only (H100 GPUs on ml.p5.48xlarge)
- **Registry**: ECR `972488948509.dkr.ecr.eu-west-2.amazonaws.com/engineering/inference`
- **Namespace**: `hyperpod-ns-inference`
- **Quantization**: MXFP4 with Marlin backend
- **Engine**: V1 only
- **Attention**: FLASH_ATTN only

## Code Surgery Principles

1. **Delete aggressively** - Remove entire modules/directories when possible
2. **Trace and remove** - Follow imports and remove all code that uses deleted modules
3. **No stubs** - Don't create stub files, remove the references entirely
4. **Keep it simple** - No backwards compatibility hacks
5. **Document removals** - Add `# mini-vLLM: <feature> removed` comments

---

## Removal TODO List

Track progress of code removal. For each item:
- [ ] = Not started
- [~] = In progress (tracing imports)
- [x] = Completed and verified

### Confirmed for Removal

#### Directories
- [x] `vllm/ray/` - Ray distributed framework (V1 engine doesn't use it) ✓ REMOVED
- [ ] `vllm/multimodal/` - Multimodal input processing (GPT-OSS is text-only)

#### Quantization (keep only: mxfp4, gptq_marlin, fp8, base classes)
- [ ] `vllm/model_executor/layers/quantization/compressed_tensors/` - entire directory
- [ ] `vllm/model_executor/layers/quantization/awq.py`
- [ ] `vllm/model_executor/layers/quantization/awq_marlin.py`
- [ ] `vllm/model_executor/layers/quantization/awq_triton.py`
- [ ] `vllm/model_executor/layers/quantization/auto_round.py`
- [ ] `vllm/model_executor/layers/quantization/bitblas.py`
- [ ] `vllm/model_executor/layers/quantization/bitsandbytes.py`
- [ ] `vllm/model_executor/layers/quantization/cpu_wna16.py`
- [ ] `vllm/model_executor/layers/quantization/deepspeedfp.py`
- [ ] `vllm/model_executor/layers/quantization/experts_int8.py`
- [ ] `vllm/model_executor/layers/quantization/fbgemm_fp8.py`
- [ ] `vllm/model_executor/layers/quantization/fp_quant.py`
- [ ] `vllm/model_executor/layers/quantization/gguf.py`
- [ ] `vllm/model_executor/layers/quantization/gptq.py`
- [ ] `vllm/model_executor/layers/quantization/gptq_bitblas.py`
- [ ] `vllm/model_executor/layers/quantization/gptq_marlin_24.py`
- [ ] `vllm/model_executor/layers/quantization/hqq_marlin.py`
- [ ] `vllm/model_executor/layers/quantization/inc.py`
- [ ] `vllm/model_executor/layers/quantization/input_quant_fp8.py`
- [ ] `vllm/model_executor/layers/quantization/ipex_quant.py`
- [ ] `vllm/model_executor/layers/quantization/kv_cache.py`
- [ ] `vllm/model_executor/layers/quantization/modelopt.py`
- [ ] `vllm/model_executor/layers/quantization/moe_wna16.py`
- [ ] `vllm/model_executor/layers/quantization/petit.py`
- [ ] `vllm/model_executor/layers/quantization/ptpc_fp8.py`
- [ ] `vllm/model_executor/layers/quantization/rtn.py`
- [ ] `vllm/model_executor/layers/quantization/torchao.py`
- [ ] `vllm/model_executor/layers/quantization/tpu_int8.py`

#### Quantization Utils (keep only: marlin_utils.py, marlin_utils_fp4.py, w8a8_utils.py)
- [ ] `vllm/model_executor/layers/quantization/utils/allspark_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/bitblas_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
- [ ] `vllm/model_executor/layers/quantization/utils/gptq_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/machete_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- [ ] `vllm/model_executor/layers/quantization/utils/marlin_utils_test.py`
- [ ] `vllm/model_executor/layers/quantization/utils/marlin_utils_test_24.py`
- [ ] `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py`
- [ ] `vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py`
- [ ] `vllm/model_executor/layers/quantization/utils/petit_utils.py`

#### V1 Attention Backends (keep only: flash_attn.py)
- [ ] `vllm/v1/attention/backends/cpu_attn.py`
- [ ] `vllm/v1/attention/backends/flashinfer.py`
- [ ] `vllm/v1/attention/backends/flex_attention.py`
- [ ] `vllm/v1/attention/backends/gdn_attn.py`
- [ ] `vllm/v1/attention/backends/linear_attn.py`
- [ ] `vllm/v1/attention/backends/mamba_attn.py`
- [ ] `vllm/v1/attention/backends/mamba1_attn.py`
- [ ] `vllm/v1/attention/backends/mamba2_attn.py`
- [ ] `vllm/v1/attention/backends/mla/` - entire directory
- [ ] `vllm/v1/attention/backends/pallas.py`
- [ ] `vllm/v1/attention/backends/rocm_aiter_fa.py`
- [ ] `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`
- [ ] `vllm/v1/attention/backends/rocm_attn.py`
- [ ] `vllm/v1/attention/backends/short_conv_attn.py`
- [ ] `vllm/v1/attention/backends/tree_attn.py`
- [ ] `vllm/v1/attention/backends/triton_attn.py`

#### Platform/Distributed (keep only: CUDA)
- [ ] `vllm/distributed/tpu_distributed_utils.py`
- [ ] `vllm/distributed/device_communicators/cpu_communicator.py`
- [ ] `vllm/distributed/device_communicators/xpu_communicator.py`
- [ ] `vllm/model_executor/model_loader/tpu.py`

#### Config Files
- [ ] `vllm/config/pooler.py`
- [ ] `vllm/config/speech_to_text.py`
- [ ] `vllm/config/speculative.py`

#### CSRC Kernels
- [ ] `csrc/quantization/w8a8/fp8/amd/` - AMD/ROCm specific
- [ ] `csrc/attention/mla/` - MLA attention (not used by GPT-OSS)
- [ ] `csrc/sparse/` - Sparse attention

### Needs Investigation

These may or may not be removable - need to check usage:

- [ ] `vllm/beam_search.py` - Check if sampling uses this
- [ ] `vllm/pooling_params.py` - Check if any code path uses this
- [ ] `vllm/usage/` - Usage tracking (may be needed for telemetry)
- [ ] `vllm/triton_utils/` - Check if any CUDA code uses Triton kernels
- [ ] `vllm/compilation/` - CUDA graph compilation (likely needed)
- [ ] `vllm/third_party/` - Check what's actually used
- [ ] `vllm/lora/` - Already stubbed, check if stubs can be removed entirely

### Completed Removals

- [x] Non-CUDA platforms (commit 44ed2c7)
- [x] LoRA support - stubbed (commit 44ed2c7)
- [x] Unused entrypoints: anthropic, sagemaker, pooling, speech-to-text (commit e22e9ff)
- [x] Unused model files, kept only GptOss (commit 60c1878)

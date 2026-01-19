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

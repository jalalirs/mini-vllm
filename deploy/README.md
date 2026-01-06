# Mini-vLLM Deployment

Deployment scripts for running mini-vLLM on AWS EKS with H100 GPUs.

## Quick Start

```bash
# 1. Setup AWS credentials
cp deploy/aws/credentials.example.sh deploy/aws/credentials.sh
# Edit credentials.sh with your AWS SSO credentials

# 2. Source environment
source deploy/aws/env.sh

# 3. Create ECR repository (one-time)
./deploy/scripts/setup-ecr.sh

# 4. Build and push image (run on building pod in cluster)
./deploy/scripts/build.sh

# 5. Deploy
./deploy/scripts/deploy.sh tp2  # TP=2 (2 GPUs)
./deploy/scripts/deploy.sh tp4  # TP=4 (4 GPUs)
./deploy/scripts/deploy.sh tp8  # TP=8 (8 GPUs, full node)
```

## Directory Structure

```
deploy/
├── aws/
│   ├── env.sh                    # Environment configuration
│   ├── credentials.sh            # AWS credentials (gitignored)
│   └── credentials.example.sh    # Credentials template
├── k8s/
│   ├── mini-vllm-tp2.yaml       # Kubernetes manifest for TP=2
│   ├── mini-vllm-tp4.yaml       # Kubernetes manifest for TP=4
│   └── mini-vllm-tp8.yaml       # Kubernetes manifest for TP=8
├── scripts/
│   ├── build.sh                 # Build image in cluster
│   ├── deploy.sh                # Deploy/manage deployment
│   ├── setup-ecr.sh             # Create ECR repository
│   ├── benchmark.sh             # Benchmark throughput
│   ├── generate-manifests.sh    # Generate K8s manifests
│   └── local-test.sh            # Local Docker test
└── README.md
```

## Deployment Options

### TP2 (2 GPUs)
- Good for: Smaller models (7B-13B) or testing
- Memory: ~160GB GPU memory
- Throughput: Highest per-GPU efficiency

### TP4 (4 GPUs)
- Good for: Medium models (30B-70B)
- Memory: ~320GB GPU memory
- Throughput: Balanced

### TP8 (8 GPUs - Full p5.48xlarge)
- Good for: Large models (120B+)
- Memory: ~640GB GPU memory
- Throughput: Maximum batch sizes

## Scripts Reference

### build.sh
Builds the Docker image inside the Kubernetes cluster using a privileged builder pod, then pushes to ECR.

```bash
./deploy/scripts/build.sh [tag]
./deploy/scripts/build.sh v0.1.0
```

### deploy.sh
Manages deployments.

```bash
# Deploy
./deploy/scripts/deploy.sh tp2

# Delete
./deploy/scripts/deploy.sh tp2 delete

# View logs
./deploy/scripts/deploy.sh tp2 logs

# Check status
./deploy/scripts/deploy.sh tp2 status

# Port forward to localhost
./deploy/scripts/deploy.sh tp2 port-forward
```

### benchmark.sh
Runs throughput benchmark against a deployed instance.

```bash
# After port-forward
./deploy/scripts/benchmark.sh http://localhost:8000 100 10
```

### local-test.sh
Builds and runs locally with Docker for testing.

```bash
./deploy/scripts/local-test.sh /path/to/model
```

## Environment Variables

Key variables in `deploy/aws/env.sh`:

| Variable | Description | Default |
|----------|-------------|---------|
| `MINI_VLLM_NAMESPACE` | K8s namespace | `hyperpod-ns-inference` |
| `MINI_VLLM_REGISTRY` | ECR registry URL | `972488948509.dkr.ecr...` |
| `MINI_VLLM_MODEL_PATH` | Model path on FSx | `/mnt/fsx/inference/models/gpt-oss-120b` |
| `MINI_VLLM_NODE_TYPE` | Node instance type | `ml.p5.48xlarge` |
| `MINI_VLLM_FSX_PVC` | FSx PVC name | `fsx-pvc-inference` |

## Monitoring

### View Logs
```bash
kubectl logs -n hyperpod-ns-inference -l app=mini-vllm-gptoss-tp2 -f
```

### Check GPU Usage
```bash
kubectl exec -n hyperpod-ns-inference -it <pod-name> -- nvidia-smi
```

### Port Forward for Local Access
```bash
kubectl port-forward -n hyperpod-ns-inference svc/mini-vllm-gptoss-tp2 8000:8000
```

## Troubleshooting

### Image Build Fails
1. Check builder pod logs: `kubectl logs mini-vllm-image-builder -n hyperpod-ns-inference`
2. Ensure Docker daemon is running in builder pod
3. Check ECR login: `aws ecr get-login-password --region eu-west-2`

### Pod Won't Start
1. Check events: `kubectl describe pod <pod-name> -n hyperpod-ns-inference`
2. Check GPU availability: `kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'`
3. Check PVC: `kubectl get pvc -n hyperpod-ns-inference`

### Model Loading Fails
1. Verify model path exists on FSx
2. Check model format (safetensors required)
3. Check GPU memory (may need larger TP size)

## Comparing with Standard vLLM

To benchmark against the official vLLM:

```bash
# Deploy standard vLLM
kubectl apply -f aws/configs/gptoss.yaml

# Run same benchmark against both
./deploy/scripts/benchmark.sh http://localhost:8000  # mini-vllm
./deploy/scripts/benchmark.sh http://localhost:8001  # standard vllm
```

Key metrics to compare:
- **Tokens/second**: Should be within 5% of standard vLLM
- **Time to First Token (TTFT)**: Should be comparable
- **P99 Latency**: Should be comparable
- **Memory Usage**: Should be similar or lower


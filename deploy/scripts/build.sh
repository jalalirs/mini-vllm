#!/bin/bash
# Build mini-vllm image in cluster and push to ECR
# Usage: ./build.sh [tag]
# Example: ./build.sh latest
#          ./build.sh v0.1.0

set -e

# Disable MSYS path conversion (Windows Git Bash)
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment
if [[ -f "$PROJECT_ROOT/deploy/aws/credentials.sh" ]]; then
    source "$PROJECT_ROOT/deploy/aws/credentials.sh"
fi
if [[ -f "$PROJECT_ROOT/deploy/aws/env.sh" ]]; then
    source "$PROJECT_ROOT/deploy/aws/env.sh"
fi

IMAGE_TAG="${1:-latest}"
IMAGE_NAME="${MINI_VLLM_PREFIX}server"
FULL_IMAGE="${MINI_VLLM_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
BUILDER_POD="${MINI_VLLM_PREFIX}image-builder"

echo "=========================================="
echo "Building mini-vLLM (CUDA kernels for H100)"
echo "Image: $FULL_IMAGE"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=========================================="
echo ""
echo "This build compiles CUDA kernels. Expected time: 15-30 minutes."
echo ""

# Step 1: Ensure builder pod exists with enough resources for CUDA compilation
echo "[1/6] Checking builder pod..."
if ! kubectl get pod "$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" &>/dev/null; then
    echo "Creating builder pod for CUDA compilation..."
    cat <<EOF | kubectl apply -n "$MINI_VLLM_NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: $BUILDER_POD
  labels:
    app: mini-vllm-builder
    kueue.x-k8s.io/queue-name: $MINI_VLLM_KUEUE_QUEUE
spec:
  containers:
  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
    command: ["dockerd-entrypoint.sh"]
    resources:
      # CUDA compilation needs significant CPU/memory
      requests:
        cpu: "8"
        memory: "32Gi"
        ephemeral-storage: "50Gi"
      limits:
        cpu: "16"
        memory: "64Gi"
        ephemeral-storage: "100Gi"
    volumeMounts:
    - name: docker-storage
      mountPath: /var/lib/docker
  volumes:
  - name: docker-storage
    emptyDir:
      sizeLimit: 100Gi
  restartPolicy: Never
EOF
    
    echo "Waiting for builder pod (may take 1-2 minutes)..."
    kubectl wait --for=condition=Ready pod/"$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" --timeout=300s
else
    echo "Builder pod already exists."
fi

# Step 2: Copy source code
echo ""
echo "[2/6] Copying source code to builder..."
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- rm -rf /build 2>/dev/null || true
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- mkdir -p /build

# Tar and copy (excluding unnecessary files)
echo "  Packaging source..."
(cd "$PROJECT_ROOT" && tar cf - \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='.pytest_cache' \
    --exclude='*.so' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='.env' \
    --exclude='*.log' \
    --exclude='logs' \
    --exclude='deploy/aws/credentials.sh' \
    --exclude='*.tar' \
    --exclude='*.tar.gz' \
    --exclude='*.whl' \
    .) | kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- tar xf - -C /build

echo "  Source copied."

# Step 3: Login to ECR
echo ""
echo "[3/6] Logging into ECR..."
aws ecr get-login-password --region "$MINI_VLLM_REGISTRY_REGION" | \
    kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker login --username AWS --password-stdin "$MINI_VLLM_REGISTRY"

# Step 4: Build image (CUDA compilation happens here)
echo ""
echo "[4/6] Building Docker image with CUDA kernels..."
echo "  This runs cmake + make inside the container."
echo "  Watch for nvcc compilation output..."
echo ""

kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker build \
    --progress=plain \
    --build-arg CUDA_VERSION=12.4.1 \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg TORCH_VERSION=2.4.0 \
    -t "$FULL_IMAGE" \
    /build

# Step 5: Push to ECR
echo ""
echo "[5/6] Pushing to ECR..."
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker push "$FULL_IMAGE"

# Step 6: Cleanup
echo ""
echo "[6/6] Cleaning up..."
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker image prune -f

echo ""
echo "=========================================="
echo "SUCCESS: $FULL_IMAGE"
echo ""
echo "Deploy with:"
echo "  ./deploy/scripts/deploy.sh tp2"
echo "  ./deploy/scripts/deploy.sh tp4"
echo "  ./deploy/scripts/deploy.sh tp8"
echo "=========================================="

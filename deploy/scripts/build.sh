#!/bin/bash
# =============================================================================
# Mini-vLLM Docker Image Builder
# =============================================================================
# Builds 3-stage Docker images with optimal caching:
#   1. essentials - Python, PyTorch, dependencies (~8GB, rarely rebuilt)
#   2. cuda - Flash Attention v3 + CUDA kernels (~2GB, rebuild on kernel changes)
#   3. server - Python code + runtime (~10GB, fast rebuild on code changes)
#
# Usage:
#   ./build.sh [essentials|cuda|server|all]
#
# Examples:
#   ./build.sh all          # Build all stages
#   ./build.sh server       # Build only server (uses cached essentials/cuda)
#   ./build.sh essentials   # Build base image
# =============================================================================

set -e

# Disable MSYS path conversion (Windows Git Bash)
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Auto-source environment files
if [[ -f "$PROJECT_ROOT/deploy/aws/credentials.sh" ]]; then
    source "$PROJECT_ROOT/deploy/aws/credentials.sh"
fi
if [[ -f "$PROJECT_ROOT/deploy/aws/env.sh" ]]; then
    source "$PROJECT_ROOT/deploy/aws/env.sh" 2>/dev/null
fi

BUILD_TARGET="${1:-server}"

# Ensure environment is loaded
if [[ -z "$MINI_VLLM_NAMESPACE" ]]; then
    echo "ERROR: Environment not loaded. Run:"
    echo "  source deploy/aws/credentials.sh"
    echo "  source deploy/aws/env.sh"
    exit 1
fi

BUILDER_POD="${MINI_VLLM_BUILDER_POD:-${MINI_VLLM_PREFIX}image-builder}"

echo "=============================================="
echo "Mini-vLLM Docker Builder"
echo "=============================================="
echo "Target: $BUILD_TARGET"
echo "Registry: $MINI_VLLM_REGISTRY"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=============================================="

# =============================================================================
# Step 1: Create builder pod if not exists
# =============================================================================
echo "[1/5] Checking builder pod..."
if ! kubectl get pod "$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" &>/dev/null; then
    echo "Creating builder pod..."
    cat <<EOF | kubectl apply -n "$MINI_VLLM_NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: $BUILDER_POD
  labels:
    app: image-builder
    kueue.x-k8s.io/queue-name: $MINI_VLLM_KUEUE_QUEUE
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: $MINI_VLLM_NODE_TYPE
  containers:
  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
    command: ["dockerd-entrypoint.sh"]
    resources:
      requests:
        memory: "64Gi"
        cpu: "32"
      limits:
        memory: "128Gi"
        cpu: "64"
  restartPolicy: Never
EOF

    echo "Waiting for builder pod to be ready..."
    kubectl wait --for=condition=Ready pod/"$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" --timeout=300s
fi
echo "Builder pod ready: $BUILDER_POD"

# =============================================================================
# Step 2: Copy source code to builder
# =============================================================================
echo "[2/5] Copying source code to builder..."
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- rm -rf /build 2>/dev/null || true
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- mkdir -p /build

(cd "$PROJECT_ROOT" && tar cf - \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.egg-info' \
    --exclude='.deps' \
    vllm csrc cmake requirements \
    CMakeLists.txt pyproject.toml setup.py \
    Dockerfile.essentials Dockerfile.cuda Dockerfile.server) | \
kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- tar xf - -C /build

echo "Source copied successfully"

# =============================================================================
# Step 3: Login to ECR
# =============================================================================
echo "[3/5] Logging into ECR..."
aws ecr get-login-password --region "$MINI_VLLM_REGISTRY_REGION" | \
    kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker login --username AWS --password-stdin "${MINI_VLLM_REGISTRY%%/*}"

# =============================================================================
# Build Functions
# =============================================================================
build_image() {
    local dockerfile=$1
    local image_name=$2
    local build_args="${3:-}"

    local image_tag="${MINI_VLLM_REGISTRY}/${image_name}:latest"

    echo ""
    echo "Building: $image_name"
    echo "Tag: $image_tag"
    echo "Dockerfile: $dockerfile"

    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker build $build_args -t "$image_tag" -f "/build/$dockerfile" /build

    echo "Pushing: $image_tag"
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker push "$image_tag"

    echo "SUCCESS: $image_tag"
}

# =============================================================================
# Step 4: Build images
# =============================================================================
echo "[4/5] Building images..."

case "$BUILD_TARGET" in
    essentials)
        build_image "Dockerfile.essentials" "${MINI_VLLM_PREFIX}essentials"
        ;;
    cuda)
        ESSENTIALS_TAG="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}essentials:latest"
        build_image "Dockerfile.cuda" "${MINI_VLLM_PREFIX}cuda" \
            "--build-arg ESSENTIALS_IMAGE=$ESSENTIALS_TAG"
        ;;
    server)
        CUDA_TAG="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}cuda:latest"
        build_image "Dockerfile.server" "${MINI_VLLM_PREFIX}server" \
            "--build-arg CUDA_IMAGE=$CUDA_TAG"
        ;;
    all)
        build_image "Dockerfile.essentials" "${MINI_VLLM_PREFIX}essentials"

        ESSENTIALS_TAG="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}essentials:latest"
        build_image "Dockerfile.cuda" "${MINI_VLLM_PREFIX}cuda" \
            "--build-arg ESSENTIALS_IMAGE=$ESSENTIALS_TAG"

        CUDA_TAG="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}cuda:latest"
        build_image "Dockerfile.server" "${MINI_VLLM_PREFIX}server" \
            "--build-arg CUDA_IMAGE=$CUDA_TAG"
        ;;
    *)
        echo "ERROR: Unknown target: $BUILD_TARGET"
        echo ""
        echo "Usage: $0 [essentials|cuda|server|all]"
        echo ""
        echo "Targets:"
        echo "  essentials  Build base image with Python, PyTorch, dependencies"
        echo "  cuda        Build CUDA kernels and Flash Attention"
        echo "  server      Build final runtime image"
        echo "  all         Build all stages in order"
        exit 1
        ;;
esac

# =============================================================================
# Step 5: Cleanup
# =============================================================================
echo "[5/5] Cleaning up dangling images..."
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
    docker image prune -f 2>/dev/null || true

echo ""
echo "=============================================="
echo "BUILD COMPLETE"
echo "=============================================="

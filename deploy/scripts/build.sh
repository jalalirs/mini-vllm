#!/bin/bash
# Build mini-vllm image in cluster and push to ECR
# Usage: ./build.sh [mode] [tag]
#
# Modes:
#   full  - Full build including CUDA (15-30 min) - default
#   base  - Build only base image with CUDA (15-30 min, rarely needed)
#   fast  - Quick build using pre-built base (~1-2 min, Python changes only)
#
# Examples:
#   ./build.sh              # Full build with tag 'latest'
#   ./build.sh fast         # Fast build (Python only) with tag 'latest'  
#   ./build.sh base v1.0    # Build base image with tag 'v1.0'
#   ./build.sh full v0.2    # Full build with tag 'v0.2'

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

# Parse arguments
MODE="full"
IMAGE_TAG="latest"

if [[ "$1" == "base" || "$1" == "fast" || "$1" == "full" ]]; then
    MODE="$1"
    IMAGE_TAG="${2:-latest}"
else
    IMAGE_TAG="${1:-latest}"
fi

# Image names
SERVER_IMAGE="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}server:${IMAGE_TAG}"
BASE_IMAGE="${MINI_VLLM_REGISTRY}/${MINI_VLLM_PREFIX}base:${IMAGE_TAG}"
BUILDER_POD="${MINI_VLLM_PREFIX}image-builder"

echo "=========================================="
echo "Building mini-vLLM"
echo "Mode: $MODE"
echo "Tag: $IMAGE_TAG"
if [[ "$MODE" == "base" ]]; then
    echo "Image: $BASE_IMAGE"
    echo "Expected time: 15-30 minutes (CUDA compilation)"
elif [[ "$MODE" == "fast" ]]; then
    echo "Image: $SERVER_IMAGE"
    echo "Base: $BASE_IMAGE"
    echo "Expected time: 1-2 minutes (Python only)"
else
    echo "Image: $SERVER_IMAGE"
    echo "Expected time: 15-30 minutes (CUDA compilation)"
fi
echo "=========================================="
echo ""

# Ensure builder pod exists
ensure_builder_pod() {
    echo "[1/6] Checking builder pod..."
    if ! kubectl get pod "$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" &>/dev/null; then
        echo "Creating builder pod..."
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
      requests:
        cpu: "8"
        memory: "32Gi"
      limits:
        cpu: "16"
        memory: "64Gi"
    volumeMounts:
    - name: docker-data
      mountPath: /var/lib/docker
  volumes:
  - name: docker-data
    emptyDir: {}
  restartPolicy: Never
  nodeSelector:
    node.kubernetes.io/instance-type: ml.m5.4xlarge
EOF
        echo "Waiting for builder pod to be ready..."
        kubectl wait --for=condition=Ready pod/"$BUILDER_POD" -n "$MINI_VLLM_NAMESPACE" --timeout=120s
    else
        echo "Builder pod already exists."
    fi
}

# Copy source to builder
copy_source() {
    echo "[2/6] Copying source code to builder..."
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
        .) | kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- tar xf - -C /build
    echo "  Source copied."
}

# ECR login
ecr_login() {
    echo "[3/6] Logging into ECR..."
    local ECR_PASSWORD=$(aws ecr get-login-password --region "${MINI_VLLM_REGISTRY_REGION:-eu-west-2}")
    echo "$ECR_PASSWORD" | kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker login --username AWS --password-stdin "${MINI_VLLM_REGISTRY%%/*}"
}

# Build functions
build_full() {
    echo "[4/6] Building Docker image (full - with CUDA)..."
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker build -t "$SERVER_IMAGE" -f /build/Dockerfile /build
}

build_base() {
    echo "[4/6] Building base image (CUDA only)..."
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker build -t "$BASE_IMAGE" -f /build/Dockerfile.base /build
}

build_fast() {
    echo "[4/6] Building fast image (Python only)..."
    echo "  Using base: $BASE_IMAGE"
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- \
        docker build -t "$SERVER_IMAGE" -f /build/Dockerfile.fast \
        --build-arg BASE_IMAGE="$BASE_IMAGE" /build
}

# Push to ECR
push_image() {
    local IMAGE="$1"
    echo "[5/6] Pushing to ECR: $IMAGE"
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- docker push "$IMAGE"
}

# Cleanup
cleanup() {
    echo "[6/6] Cleaning up..."
    kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- docker image prune -f || true
}

# Main execution
ensure_builder_pod
kubectl exec -n "$MINI_VLLM_NAMESPACE" "$BUILDER_POD" -- mkdir -p /build
copy_source
ecr_login

case "$MODE" in
    base)
        build_base
        push_image "$BASE_IMAGE"
        ;;
    fast)
        build_fast
        push_image "$SERVER_IMAGE"
        ;;
    full)
        build_full
        push_image "$SERVER_IMAGE"
        ;;
esac

cleanup

echo ""
echo "=========================================="
if [[ "$MODE" == "base" ]]; then
    echo "SUCCESS: $BASE_IMAGE"
    echo ""
    echo "Now use fast builds for Python changes:"
    echo "  ./build.sh fast"
else
    echo "SUCCESS: $SERVER_IMAGE"
    echo ""
    echo "Deploy with:"
    echo "  ./deploy/scripts/deploy.sh tp2"
fi
echo "=========================================="

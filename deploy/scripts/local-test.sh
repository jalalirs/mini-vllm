#!/bin/bash
# Test mini-vLLM locally with Docker
# Usage: ./local-test.sh [model_path]
# Example: ./local-test.sh /path/to/models/gpt-oss-120b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:-/mnt/fsx/inference/models/gpt-oss-120b}"
IMAGE_NAME="mini-vllm:local"
CONTAINER_NAME="mini-vllm-test"

echo "=========================================="
echo "Local Mini-vLLM Test"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "=========================================="

# Step 1: Build image locally
echo "[1/3] Building Docker image..."
cd "$PROJECT_ROOT"
docker build -t "$IMAGE_NAME" .

# Step 2: Stop any existing container
echo "[2/3] Cleaning up..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Step 3: Run container
echo "[3/3] Starting container..."

# Check if model path exists
if [[ -d "$MODEL_PATH" ]]; then
    MODEL_MOUNT="-v ${MODEL_PATH}:/models/gpt-oss-120b:ro"
    MODEL_ARG="/models/gpt-oss-120b"
else
    echo "WARNING: Model path doesn't exist locally"
    echo "Container will start but may fail to load model"
    MODEL_MOUNT=""
    MODEL_ARG="$MODEL_PATH"
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -p 8000:8000 \
    $MODEL_MOUNT \
    "$IMAGE_NAME" \
    --model "$MODEL_ARG" \
    --tensor-parallel-size 2 \
    --dtype bfloat16

echo ""
echo "Container started!"
echo ""
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

echo ""
echo "=========================================="
echo "Test Commands:"
echo "=========================================="
echo ""
echo "# Check health"
echo "curl http://localhost:8000/health"
echo ""
echo "# List models"
echo "curl http://localhost:8000/v1/models"
echo ""
echo "# Test completion"
echo 'curl -X POST http://localhost:8000/v1/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '\''{"model": "gpt-oss-120b", "prompt": "Hello, world!", "max_tokens": 50}'\'''
echo ""
echo "# View logs"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "# Stop container"
echo "docker rm -f $CONTAINER_NAME"
echo "=========================================="


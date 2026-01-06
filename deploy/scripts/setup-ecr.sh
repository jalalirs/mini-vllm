#!/bin/bash
# Create ECR repository for mini-vLLM
# Usage: ./setup-ecr.sh

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

echo "=========================================="
echo "Setting up ECR Repository for mini-vLLM"
echo "Registry: $MINI_VLLM_REGISTRY"
echo "Region: $MINI_VLLM_REGISTRY_REGION"
echo "=========================================="

# Extract repo prefix from registry URL
REPO_PREFIX=$(echo "$MINI_VLLM_REGISTRY" | sed 's|.*\.amazonaws\.com/||')

REPO_NAME="${REPO_PREFIX}/${MINI_VLLM_PREFIX}server"

echo ""
echo "Creating repository: $REPO_NAME"

if aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$MINI_VLLM_REGISTRY_REGION" &>/dev/null; then
    echo "  ✓ Repository already exists"
else
    aws ecr create-repository \
        --repository-name "$REPO_NAME" \
        --region "$MINI_VLLM_REGISTRY_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        2>/dev/null && echo "  ✓ Repository created" || echo "  ✗ Failed to create repository"
fi

echo ""
echo "=========================================="
echo "ECR setup complete"
echo "Full image path: $MINI_VLLM_REGISTRY/${MINI_VLLM_PREFIX}server:latest"
echo "=========================================="


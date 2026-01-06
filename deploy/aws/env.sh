#!/bin/bash
# Mini-vLLM AWS Environment Configuration
# Source this file before running deployment scripts: source deploy/aws/env.sh

# Disable MSYS path conversion (Windows Git Bash)
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

# =============================================================================
# CLUSTER
# =============================================================================
export MINI_VLLM_NAMESPACE="hyperpod-ns-inference"
export MINI_VLLM_PREFIX="mini-vllm-"
export MINI_VLLM_KUEUE_QUEUE="${MINI_VLLM_NAMESPACE}-localqueue"

# =============================================================================
# CONTAINER REGISTRY
# =============================================================================
export MINI_VLLM_REGISTRY="972488948509.dkr.ecr.eu-west-2.amazonaws.com/engineering/inference"
export MINI_VLLM_REGISTRY_REGION="eu-west-2"

# =============================================================================
# STORAGE (FSx Lustre)
# =============================================================================
export MINI_VLLM_FSX_PVC="fsx-pvc-inference"
export MINI_VLLM_FSX_MOUNT="/mnt/fsx"
export MINI_VLLM_MODELS_PATH="/mnt/fsx/inference/models"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
export MINI_VLLM_MODEL_NAME="gpt-oss-120b"
export MINI_VLLM_MODEL_PATH="${MINI_VLLM_MODELS_PATH}/${MINI_VLLM_MODEL_NAME}"

# =============================================================================
# GPU CONFIGURATION (p5.48xlarge = 8x H100)
# =============================================================================
export MINI_VLLM_NODE_TYPE="ml.p5.48xlarge"
export MINI_VLLM_GPU_PER_NODE=8

# =============================================================================
# DEFAULT TENSOR PARALLEL SIZE
# =============================================================================
export MINI_VLLM_TP_SIZE=2

# =============================================================================
# IMAGE BUILDER
# =============================================================================
export MINI_VLLM_BUILDER_IMAGE="docker:24-dind"
export MINI_VLLM_BUILDER_POD="${MINI_VLLM_PREFIX}image-builder"

# =============================================================================
# AWS SSO Login URL
# =============================================================================
export AWS_SSO_LOGIN_URL="https://d-9c675fdb02.awsapps.com/start/#/?tab=accounts"

# =============================================================================
# Credential Check
# =============================================================================
_check_aws_credentials() {
    if ! aws sts get-caller-identity &>/dev/null; then
        echo ""
        echo -e "\033[0;31m========================================\033[0m"
        echo -e "\033[0;31m  AWS CREDENTIALS EXPIRED OR INVALID\033[0m"
        echo -e "\033[0;31m========================================\033[0m"
        echo ""
        echo "Opening AWS SSO login page..."
        echo "Please copy new credentials to: deploy/aws/credentials.sh"
        echo ""
        
        if command -v cmd.exe &>/dev/null; then
            cmd.exe /c start chrome "$AWS_SSO_LOGIN_URL" 2>/dev/null || \
            cmd.exe /c start "$AWS_SSO_LOGIN_URL" 2>/dev/null
        elif command -v open &>/dev/null; then
            open "$AWS_SSO_LOGIN_URL"
        elif command -v xdg-open &>/dev/null; then
            xdg-open "$AWS_SSO_LOGIN_URL"
        else
            echo "Open this URL manually: $AWS_SSO_LOGIN_URL"
        fi
        
        return 1
    fi
    return 0
}

# Check credentials
if ! _check_aws_credentials; then
    echo ""
    echo "Re-source this file after updating credentials:"
    echo "  source deploy/aws/credentials.sh && source deploy/aws/env.sh"
fi

echo "Mini-vLLM AWS environment loaded."
echo "  Namespace: $MINI_VLLM_NAMESPACE"
echo "  Registry: $MINI_VLLM_REGISTRY"
echo "  Model: $MINI_VLLM_MODEL_PATH"
echo "  TP Size: $MINI_VLLM_TP_SIZE"


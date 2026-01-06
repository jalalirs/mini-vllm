#!/bin/bash
# Check mini-vLLM deployment status
# Usage: ./status.sh [tp-config]
# Examples:
#   ./status.sh           # Show all mini-vllm deployments
#   ./status.sh tp2       # Show TP=2 deployment details

set -e

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

TP_CONFIG="${1:-all}"

echo "=========================================="
echo "Mini-vLLM Status"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=========================================="

if [[ "$TP_CONFIG" == "all" ]]; then
    echo ""
    echo "Deployments:"
    kubectl get deployments -n "$MINI_VLLM_NAMESPACE" -l "app in (mini-vllm-gptoss-tp2,mini-vllm-gptoss-tp4,mini-vllm-gptoss-tp8)" 2>/dev/null || echo "  No mini-vllm deployments found"
    
    echo ""
    echo "Pods:"
    kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l "app in (mini-vllm-gptoss-tp2,mini-vllm-gptoss-tp4,mini-vllm-gptoss-tp8)" -o wide 2>/dev/null || echo "  No mini-vllm pods found"
    
    echo ""
    echo "Services:"
    kubectl get services -n "$MINI_VLLM_NAMESPACE" -l "app in (mini-vllm-gptoss-tp2,mini-vllm-gptoss-tp4,mini-vllm-gptoss-tp8)" 2>/dev/null || echo "  No mini-vllm services found"
else
    DEPLOYMENT_NAME="${MINI_VLLM_PREFIX}gptoss-${TP_CONFIG}"
    
    echo ""
    echo "Deployment: $DEPLOYMENT_NAME"
    kubectl get deployment "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" -o wide 2>/dev/null || echo "  Not found"
    
    echo ""
    echo "Pod details:"
    kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l "app=$DEPLOYMENT_NAME" -o wide 2>/dev/null || echo "  No pods found"
    
    echo ""
    echo "Pod describe:"
    POD=$(kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l "app=$DEPLOYMENT_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$POD" ]]; then
        kubectl describe pod "$POD" -n "$MINI_VLLM_NAMESPACE" | tail -30
    fi
    
    echo ""
    echo "Service:"
    kubectl get service "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" 2>/dev/null || echo "  Not found"
fi

echo ""
echo "=========================================="


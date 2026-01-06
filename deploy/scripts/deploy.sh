#!/bin/bash
# Deploy mini-vLLM with specified tensor parallel size
# Usage: ./deploy.sh <tp-config> [action]
# Examples:
#   ./deploy.sh tp2         # Deploy with TP=2 (uses 2 GPUs)
#   ./deploy.sh tp4         # Deploy with TP=4 (uses 4 GPUs)
#   ./deploy.sh tp8         # Deploy with TP=8 (uses 8 GPUs)
#   ./deploy.sh tp2 delete  # Remove TP=2 deployment

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

TP_CONFIG="${1:-tp2}"
ACTION="${2:-apply}"

# Parse TP size from config name
case "$TP_CONFIG" in
    tp2) TP_SIZE=2 ;;
    tp4) TP_SIZE=4 ;;
    tp8) TP_SIZE=8 ;;
    *)
        echo "ERROR: Invalid config. Use: tp2, tp4, or tp8"
        exit 1
        ;;
esac

export MINI_VLLM_TP_SIZE=$TP_SIZE

DEPLOYMENT_NAME="${MINI_VLLM_PREFIX}gptoss-${TP_CONFIG}"
MANIFEST="$PROJECT_ROOT/deploy/k8s/mini-vllm-${TP_CONFIG}.yaml"

echo "=========================================="
echo "Deployment: $DEPLOYMENT_NAME"
echo "Tensor Parallel: $TP_SIZE"
echo "Action: $ACTION"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=========================================="

case "$ACTION" in
    apply)
        if [[ ! -f "$MANIFEST" ]]; then
            echo "ERROR: Manifest not found: $MANIFEST"
            echo "Generate it with: ./deploy/scripts/generate-manifests.sh"
            exit 1
        fi
        
        echo "Applying manifest..."
        envsubst < "$MANIFEST" | kubectl apply -n "$MINI_VLLM_NAMESPACE" -f -
        
        echo "Waiting for deployment..."
        kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" --timeout=600s || true
        
        # Get pod status
        echo ""
        echo "Pod status:"
        kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME"
        ;;
    
    delete)
        echo "Deleting deployment..."
        envsubst < "$MANIFEST" | kubectl delete -n "$MINI_VLLM_NAMESPACE" -f - --ignore-not-found
        ;;
    
    logs)
        echo "Getting logs..."
        kubectl logs -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" --tail=100 -f
        ;;
    
    status)
        echo "Deployment status:"
        kubectl get deployment "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" -o wide
        echo ""
        echo "Pod status:"
        kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" -o wide
        ;;
    
    port-forward)
        echo "Port forwarding to localhost:8000..."
        kubectl port-forward -n "$MINI_VLLM_NAMESPACE" svc/"$DEPLOYMENT_NAME" 8000:8000
        ;;
    
    *)
        echo "ERROR: Unknown action: $ACTION"
        echo "Valid actions: apply, delete, logs, status, port-forward"
        exit 1
        ;;
esac

echo "=========================================="
echo "DONE"
echo "=========================================="


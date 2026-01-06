#!/bin/bash
# Monitor mini-vLLM pod logs
# Usage: ./monitor.sh <tp-config>
# Examples:
#   ./monitor.sh tp2         # Stream logs for TP=2 deployment
#   ./monitor.sh tp2 prev    # Show previous container logs (after crash)

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

TP_CONFIG="${1:-tp2}"
MODE="${2:-follow}"

case "$TP_CONFIG" in
    tp2|tp4|tp8) ;;
    *)
        echo "ERROR: Invalid config. Use: tp2, tp4, or tp8"
        exit 1
        ;;
esac

DEPLOYMENT_NAME="${MINI_VLLM_PREFIX}gptoss-${TP_CONFIG}"

echo "=========================================="
echo "Monitoring: $DEPLOYMENT_NAME"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=========================================="

# Get pod name
POD=$(kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l "app=$DEPLOYMENT_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$POD" ]]; then
    echo "ERROR: No pod found for $DEPLOYMENT_NAME"
    echo ""
    echo "Current pods in namespace:"
    kubectl get pods -n "$MINI_VLLM_NAMESPACE"
    exit 1
fi

echo "Pod: $POD"
echo "=========================================="

case "$MODE" in
    prev|previous)
        echo "Showing previous container logs..."
        kubectl logs -n "$MINI_VLLM_NAMESPACE" "$POD" --previous --tail=500
        ;;
    all)
        echo "Showing all logs..."
        kubectl logs -n "$MINI_VLLM_NAMESPACE" "$POD" --tail=1000
        ;;
    *)
        echo "Streaming logs (Ctrl+C to stop)..."
        kubectl logs -n "$MINI_VLLM_NAMESPACE" "$POD" -f --tail=100
        ;;
esac


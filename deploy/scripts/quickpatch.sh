#!/bin/bash
# Quick patch Python files in running pod (no rebuild)
# Usage: ./quickpatch.sh <tp-config>
#
# This copies local Python files directly to the running pod and restarts it.
# Fastest iteration - takes ~10-20 seconds. Use for quick Python fixes.
#
# Note: Changes are lost on pod restart. For permanent changes, use:
#   ./build.sh fast && ./deploy.sh <tp-config>

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

case "$TP_CONFIG" in
    tp2|tp4|tp8) ;;
    *)
        echo "ERROR: Invalid config. Use: tp2, tp4, or tp8"
        exit 1
        ;;
esac

DEPLOYMENT_NAME="${MINI_VLLM_PREFIX}gptoss-${TP_CONFIG}"

echo "=========================================="
echo "Quick Patch: $DEPLOYMENT_NAME"
echo "=========================================="

# Get pod name
POD=$(kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [[ -z "$POD" ]]; then
    echo "ERROR: No pod found for $DEPLOYMENT_NAME"
    echo "Deploy first with: ./deploy.sh $TP_CONFIG"
    exit 1
fi

echo "Pod: $POD"
echo ""

# Copy Python files (exclude __pycache__, .pyc)
echo "[1/2] Copying Python files..."
cd "$PROJECT_ROOT"
tar cf - --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' mini_vllm/ | \
    kubectl exec -i -n "$MINI_VLLM_NAMESPACE" "$POD" -- tar xf - -C /app/

echo "[2/2] Restarting pod..."
kubectl delete pod -n "$MINI_VLLM_NAMESPACE" "$POD"

echo ""
echo "=========================================="
echo "Patch applied. New pod starting..."
echo "Monitor with: ./monitor.sh $TP_CONFIG"
echo "=========================================="


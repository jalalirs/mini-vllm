#!/bin/bash
# Deploy mini-vLLM to Kubernetes
# Usage: ./deploy.sh <tp-config> [action]
# Examples:
#   ./deploy.sh tp2         # Deploy with TP=2
#   ./deploy.sh tp8         # Deploy with TP=8
#   ./deploy.sh tp2 delete  # Remove deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment
source "$PROJECT_ROOT/deploy/aws/env.sh"
[[ -f "$PROJECT_ROOT/deploy/aws/credentials.sh" ]] && source "$PROJECT_ROOT/deploy/aws/credentials.sh"

TP_CONFIG="${1:-tp2}"
ACTION="${2:-apply}"

# Parse TP size
case "$TP_CONFIG" in
    tp1) TP_SIZE=1 ;;
    tp2) TP_SIZE=2 ;;
    tp4) TP_SIZE=4 ;;
    tp8) TP_SIZE=8 ;;
    *) echo "ERROR: Invalid config. Use: tp1, tp2, tp4, or tp8"; exit 1 ;;
esac

DEPLOYMENT_NAME="${MINI_VLLM_PREFIX}gptoss-${TP_CONFIG}"

echo "=============================================="
echo "Deployment: $DEPLOYMENT_NAME"
echo "Tensor Parallel: $TP_SIZE"
echo "Action: $ACTION"
echo "Namespace: $MINI_VLLM_NAMESPACE"
echo "=============================================="

generate_manifest() {
    cat << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $DEPLOYMENT_NAME
  namespace: $MINI_VLLM_NAMESPACE
  labels:
    app: $DEPLOYMENT_NAME
    kueue.x-k8s.io/queue-name: $MINI_VLLM_KUEUE_QUEUE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $DEPLOYMENT_NAME
  template:
    metadata:
      labels:
        app: $DEPLOYMENT_NAME
        kueue.x-k8s.io/queue-name: $MINI_VLLM_KUEUE_QUEUE
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: $MINI_VLLM_NODE_TYPE
      volumes:
        - name: fsx-models
          persistentVolumeClaim:
            claimName: $MINI_VLLM_PVC
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 64Gi
      containers:
        - name: mini-vllm
          image: $MINI_VLLM_REGISTRY/${MINI_VLLM_PREFIX}server:latest
          imagePullPolicy: Always
          args:
            - "--model"
            - "$MINI_VLLM_MODEL_PATH"
            - "--tensor-parallel-size"
            - "$TP_SIZE"
            - "--port"
            - "8000"
            - "--host"
            - "0.0.0.0"
          env:
            - name: HF_HOME
              value: $MINI_VLLM_MOUNT/inference/huggingface
            - name: VLLM_LOGGING_LEVEL
              value: INFO
          resources:
            limits:
              nvidia.com/gpu: "$TP_SIZE"
              memory: "256Gi"
            requests:
              nvidia.com/gpu: "$TP_SIZE"
              memory: "128Gi"
          volumeMounts:
            - name: fsx-models
              mountPath: $MINI_VLLM_MOUNT
            - name: shm
              mountPath: /dev/shm
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: $DEPLOYMENT_NAME
  namespace: $MINI_VLLM_NAMESPACE
spec:
  selector:
    app: $DEPLOYMENT_NAME
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
  type: ClusterIP
EOF
}

case "$ACTION" in
    apply)
        # Delete existing first
        echo "Removing existing deployment..."
        kubectl delete deployment "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" --ignore-not-found
        kubectl wait --for=delete pod -l app="$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" --timeout=60s 2>/dev/null || true
        
        echo "Applying manifest..."
        generate_manifest | kubectl apply -f -
        
        echo ""
        echo "Pod status:"
        sleep 2
        kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME"
        ;;
    
    delete)
        echo "Deleting deployment..."
        kubectl delete deployment "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" --ignore-not-found
        kubectl delete service "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" --ignore-not-found
        ;;
    
    logs)
        kubectl logs -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" --tail=100 -f
        ;;
    
    status)
        kubectl get deployment "$DEPLOYMENT_NAME" -n "$MINI_VLLM_NAMESPACE" -o wide
        echo ""
        kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" -o wide
        ;;

    port-forward)
        echo "Port forwarding to $DEPLOYMENT_NAME:8000..."
        kubectl port-forward svc/"$DEPLOYMENT_NAME" 8000:8000 -n "$MINI_VLLM_NAMESPACE"
        ;;

    test)
        # Get pod name
        POD_NAME=$(kubectl get pods -n "$MINI_VLLM_NAMESPACE" -l app="$DEPLOYMENT_NAME" -o jsonpath='{.items[0].metadata.name}')
        MAX_TOKENS="${3:-100}"
        echo "Testing with max_tokens=$MAX_TOKENS on pod $POD_NAME..."

        # Run curl inside the pod with timing
        START_TIME=$(date +%s%3N)
        RESPONSE=$(kubectl exec -n "$MINI_VLLM_NAMESPACE" "$POD_NAME" -- curl -s http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MINI_VLLM_MODEL_PATH\",
                \"prompt\": \"Write a detailed explanation of how neural networks work:\",
                \"max_tokens\": $MAX_TOKENS,
                \"ignore_eos\": true,
                \"temperature\": 0.7
            }")
        END_TIME=$(date +%s%3N)

        echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
        echo ""

        # Calculate throughput using awk
        ELAPSED_MS=$((END_TIME - START_TIME))
        THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", $MAX_TOKENS / ($ELAPSED_MS / 1000)}")
        echo "=============================================="
        echo "Time: ${ELAPSED_MS}ms ($(awk "BEGIN {printf \"%.2f\", $ELAPSED_MS / 1000}")s)"
        echo "Throughput: ${THROUGHPUT} tokens/sec"
        echo "=============================================="
        ;;

    *)
        echo "ERROR: Unknown action. Use: apply, delete, logs, status, port-forward, test"
        exit 1
        ;;
esac

echo "=============================================="
echo "DONE"
echo "=============================================="

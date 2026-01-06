#!/bin/bash
# Launcher for mini-vLLM that handles tensor parallelism
# For TP > 1, uses torchrun to spawn multiple processes

set -e

# Parse tensor-parallel-size from arguments
TP_SIZE=1
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --tensor-parallel-size|-tp)
            TP_SIZE="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

echo "[launcher] Tensor Parallel Size: $TP_SIZE"

if [[ "$TP_SIZE" -eq 1 ]]; then
    # Single GPU - run directly
    echo "[launcher] Running single-process mode"
    exec python -m mini_vllm "${ARGS[@]}"
else
    # Multi-GPU - use torchrun
    echo "[launcher] Running multi-process mode with torchrun"
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$TP_SIZE" \
        -m mini_vllm "${ARGS[@]}"
fi


# SPDX-License-Identifier: Apache-2.0
"""Fused MoE forward pass with CUDA kernel support."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Check for CUDA ops
_HAS_CUDA_OPS = False
try:
    from mini_vllm.ops import silu_and_mul, cuda_ops_available
    if cuda_ops_available():
        _silu_and_mul = silu_and_mul
        _HAS_CUDA_OPS = True
except ImportError:
    pass


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-k routing weights.
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        gating_output: [num_tokens, num_experts] router logits
        top_k: Number of experts per token
        renormalize: Whether to renormalize weights to sum to 1
        
    Returns:
        topk_weights: [num_tokens, top_k]
        topk_ids: [num_tokens, top_k]
        token_expert_indices: [num_tokens * top_k]
    """
    # Softmax over experts
    scores = F.softmax(gating_output, dim=-1)
    
    # Top-k selection
    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1)
    
    # Renormalize weights
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    # Flatten for expert processing
    token_expert_indices = topk_ids.view(-1)
    
    return topk_weights, topk_ids, token_expert_indices


def _apply_swiglu(gate_up: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply SwiGLU activation: silu(gate) * up.
    
    Uses CUDA kernel when available for better performance.
    """
    if out is None:
        d = gate_up.shape[-1] // 2
        out = torch.empty(gate_up.shape[:-1] + (d,), device=gate_up.device, dtype=gate_up.dtype)
    
    if _HAS_CUDA_OPS and gate_up.is_cuda:
        _silu_and_mul(out, gate_up)
    else:
        # PyTorch fallback
        d = gate_up.shape[-1] // 2
        gate = gate_up[..., :d]
        up = gate_up[..., d:]
        out.copy_(F.silu(gate) * up)
    
    return out


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool = True,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MoE forward pass.
    
    Uses CUDA kernels for activation when available.
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: [num_experts, 2 * intermediate_size, hidden_size] gate+up weights
        w2: [num_experts, hidden_size, intermediate_size] down weights
        router_logits: [num_tokens, num_experts]
        top_k: Number of experts per token
        renormalize: Renormalize routing weights
        w1_bias: Optional bias for gate+up
        w2_bias: Optional bias for down proj
        
    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[2]
    
    # Get routing
    topk_weights, topk_ids, _ = fused_topk(hidden_states, router_logits, top_k, renormalize)
    
    # Initialize output
    output = torch.zeros_like(hidden_states)
    
    # Pre-allocate intermediate buffer for SwiGLU
    swiglu_out = torch.empty(1, intermediate_size, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Process each token-expert pair
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_idx = topk_ids[token_idx, k].item()
            weight = topk_weights[token_idx, k]
            
            # Get input for this token
            x = hidden_states[token_idx:token_idx+1]  # [1, hidden]
            
            # Gate + Up projection (fused in w1)
            gate_up = F.linear(x, w1[expert_idx])  # [1, 2 * intermediate]
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_idx]
            
            # SwiGLU activation (uses CUDA kernel when available)
            intermediate = _apply_swiglu(gate_up, swiglu_out)  # [1, intermediate]
            
            # Down projection
            expert_output = F.linear(intermediate, w2[expert_idx])  # [1, hidden]
            if w2_bias is not None:
                expert_output = expert_output + w2_bias[expert_idx]
            
            # Weighted sum
            output[token_idx] += weight * expert_output.squeeze(0)
    
    return output

# SPDX-License-Identifier: Apache-2.0
"""Fused MoE layer for GPT-OSS."""

import torch
import torch.nn as nn
from typing import Optional

from mini_vllm.distributed import get_tensor_parallel_world_size


class FusedMoE(nn.Module):
    """Fused Mixture of Experts layer.
    
    Implements efficient MoE with:
    - Fused gate+up projection (SwiGLU)
    - Top-k expert routing
    - Tensor parallelism support
    
    For GPT-OSS: 16 experts, top-2 routing, SwiGLU activation.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        reduce_results: bool = True,
        renormalize: bool = True,
        has_bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        tp_size = get_tensor_parallel_world_size()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        
        # Per-partition intermediate size
        assert intermediate_size % tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // tp_size
        
        # Expert weights: [num_experts, 2 * intermediate_size_per_partition, hidden_size]
        # Contains fused gate and up projections
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_experts,
                2 * self.intermediate_size_per_partition,
                hidden_size,
                dtype=dtype,
            )
        )
        
        # Down projection: [num_experts, hidden_size, intermediate_size_per_partition]
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                self.intermediate_size_per_partition,
                dtype=dtype,
            )
        )
        
        # Biases
        if has_bias:
            self.w13_bias = nn.Parameter(
                torch.zeros(num_experts, 2 * self.intermediate_size_per_partition, dtype=dtype)
            )
            self.w2_bias = nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=dtype)
            )
        else:
            self.register_parameter("w13_bias", None)
            self.register_parameter("w2_bias", None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            router_logits: [num_tokens, num_experts]
            
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        from mini_vllm.layers.fused_moe.fused_moe import fused_moe
        
        return fused_moe(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            w1_bias=self.w13_bias,
            w2_bias=self.w2_bias,
        )

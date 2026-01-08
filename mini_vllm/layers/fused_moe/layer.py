# SPDX-License-Identifier: Apache-2.0
"""Fused MoE layer for GPT-OSS with MXFP4 support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from mini_vllm.distributed import get_tensor_parallel_world_size, get_tensor_parallel_rank

logger = logging.getLogger(__name__)


class FusedMoE(nn.Module):
    """Fused Mixture of Experts layer with MXFP4 quantization.
    
    Implements efficient MoE with:
    - MXFP4 quantized weights (4-bit with FP8 scales)
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
        use_mxfp4: bool = True,
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
        self.use_mxfp4 = use_mxfp4
        self.dtype = dtype
        self.mxfp4_block = 32  # MXFP4 block size
        
        # Per-partition intermediate size
        assert intermediate_size % tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // tp_size
        N = self.intermediate_size_per_partition
        K = hidden_size
        E = num_experts
        
        if use_mxfp4:
            # MXFP4 quantized weights
            # w13: [E, 2*N, K/2] uint8 (packed FP4, 2 values per byte)
            self.w13_weight = nn.Parameter(
                torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            
            # w13 scales: [E, 2*N, K/block] uint8 (FP8 E4M3)
            self.w13_weight_scale = nn.Parameter(
                torch.zeros(E, 2 * N, K // self.mxfp4_block, dtype=torch.uint8),
                requires_grad=False,
            )
            
            # w2: [E, K, N/2] uint8 (packed FP4)
            self.w2_weight = nn.Parameter(
                torch.zeros(E, K, N // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            
            # w2 scales: [E, K, N/block] uint8 (FP8 E4M3)
            self.w2_weight_scale = nn.Parameter(
                torch.zeros(E, K, N // self.mxfp4_block, dtype=torch.uint8),
                requires_grad=False,
            )
            
            logger.debug(
                f"Created MXFP4 MoE: E={E}, N={N}, K={K}, "
                f"w13=[{E}, {2*N}, {K//2}], w2=[{E}, {K}, {N//2}]"
            )
        else:
            # Full precision weights
            self.w13_weight = nn.Parameter(
                torch.empty(E, 2 * N, K, dtype=dtype)
            )
            self.w2_weight = nn.Parameter(
                torch.empty(E, K, N, dtype=dtype)
            )
            self.register_parameter("w13_weight_scale", None)
            self.register_parameter("w2_weight_scale", None)
        
        # Biases (always full precision)
        if has_bias:
            self.w13_bias = nn.Parameter(
                torch.zeros(E, 2 * N, dtype=torch.float32 if use_mxfp4 else dtype)
            )
            self.w2_bias = nn.Parameter(
                torch.zeros(E, K, dtype=torch.float32 if use_mxfp4 else dtype)
            )
        else:
            self.register_parameter("w13_bias", None)
            self.register_parameter("w2_bias", None)
    
    def _dequantize_mxfp4(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize MXFP4 weights.
        
        Args:
            weight: [E, M, N/2] uint8 packed weights
            scale: [E, M, N/block] uint8 FP8 scales
            
        Returns:
            [E, M, N] dequantized weights in bfloat16
        """
        E, M, N_half = weight.shape
        N = N_half * 2
        block = self.mxfp4_block
        device = weight.device
        
        # Unpack FP4 from uint8 (2 values per byte)
        low = (weight & 0x0F).to(torch.int8)
        high = ((weight >> 4) & 0x0F).to(torch.int8)
        
        # Interleave to get original order
        unpacked = torch.stack([low, high], dim=-1).reshape(E, M, N)
        
        # FP4 E2M1 format: sign(1) + exp(2) + mantissa(1)
        # Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        # Use lookup table for efficiency
        fp4_table = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=torch.float32, device=device)
        
        # Convert to float using table
        dequant = fp4_table[unpacked.long()]
        
        # Convert FP8 E4M3 scales to float
        # FP8 E4M3: sign(1) + exp(4) + mantissa(3)
        scale_sign = ((scale >> 7) & 1).float() * -2.0 + 1.0
        scale_exp = ((scale >> 3) & 0xF).int()
        scale_mant = (scale & 0x7).float() / 8.0
        
        # 2^(exp-7) * (1 + mant), with subnormal handling
        scale_float = torch.where(
            scale_exp == 0,
            scale_mant / 64.0,
            torch.pow(2.0, scale_exp.float() - 7.0) * (1.0 + scale_mant)
        ) * scale_sign
        
        # Expand scales to match weight dimensions
        scale_expanded = scale_float.repeat_interleave(block, dim=-1)
        if scale_expanded.shape[-1] > N:
            scale_expanded = scale_expanded[..., :N]
        elif scale_expanded.shape[-1] < N:
            # Pad if needed
            pad_size = N - scale_expanded.shape[-1]
            scale_expanded = F.pad(scale_expanded, (0, pad_size), value=1.0)
        
        # Apply scales
        result = dequant * scale_expanded
        
        return result.to(torch.bfloat16)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with MXFP4 dequantization.
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            router_logits: [num_tokens, num_experts]
            
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Get routing weights and expert assignments
        topk_weights, topk_ids = self._select_experts(router_logits)
        
        if self.use_mxfp4:
            return self._forward_mxfp4(hidden_states, topk_weights, topk_ids)
        else:
            return self._forward_fp16(hidden_states, topk_weights, topk_ids)
    
    def _select_experts(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts based on router logits.
        
        Args:
            router_logits: [num_tokens, num_experts]
            
        Returns:
            topk_weights: [num_tokens, top_k]
            topk_ids: [num_tokens, top_k]
        """
        # Softmax over experts
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-k
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Renormalize if needed
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights.to(router_logits.dtype), topk_ids
    
    def _forward_mxfp4(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with MXFP4 dequantization."""
        batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Dequantize weights (this is expensive, ideally use Triton kernels)
        w13 = self._dequantize_mxfp4(self.w13_weight, self.w13_weight_scale)  # [E, 2N, K]
        w2 = self._dequantize_mxfp4(self.w2_weight, self.w2_weight_scale)    # [E, K, N]
        
        E, N2, K = w13.shape
        N = N2 // 2
        
        # Get biases
        w13_bias = self.w13_bias.to(dtype) if self.w13_bias is not None else None
        w2_bias = self.w2_bias.to(dtype) if self.w2_bias is not None else None
        
        # Process with grouped operations for efficiency
        output = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]
            weights = topk_weights[:, k:k+1]
            
            for e in range(E):
                mask = (expert_ids == e)
                if not mask.any():
                    continue
                
                x_e = hidden_states[mask]
                
                # Gate-up projection: x @ w13^T
                w13_e = w13[e]  # [2N, K]
                h = F.linear(x_e, w13_e)  # [tokens, 2N]
                if w13_bias is not None:
                    h = h + w13_bias[e]
                
                # SwiGLU: silu(gate) * up
                gate, up = h.chunk(2, dim=-1)
                h = F.silu(gate) * up
                
                # Down projection: h @ w2^T
                w2_e = w2[e]  # [K, N]
                out = F.linear(h, w2_e)  # [tokens, K]
                if w2_bias is not None:
                    out = out + w2_bias[e]
                
                output[mask] += weights[mask] * out
        
        return output
    
    def _forward_fp16(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with full precision weights."""
        batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        E = self.num_experts
        output = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]
            weights = topk_weights[:, k:k+1]
            
            for e in range(E):
                mask = (expert_ids == e)
                if not mask.any():
                    continue
                
                x_e = hidden_states[mask]
                
                # Gate-up projection
                w13_e = self.w13_weight[e]
                h = F.linear(x_e, w13_e)
                if self.w13_bias is not None:
                    h = h + self.w13_bias[e]
                
                # SwiGLU
                gate, up = h.chunk(2, dim=-1)
                h = F.silu(gate) * up
                
                # Down projection
                w2_e = self.w2_weight[e]
                out = F.linear(h, w2_e)
                if self.w2_bias is not None:
                    out = out + self.w2_bias[e]
                
                output[mask] += weights[mask] * out
        
        return output

# SPDX-License-Identifier: Apache-2.0
"""Fused MoE layer for GPT-OSS with MXFP4 Triton backend."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from mini_vllm.distributed import get_tensor_parallel_world_size, get_tensor_parallel_rank

logger = logging.getLogger(__name__)


class FusedMoE(nn.Module):
    """Fused Mixture of Experts layer with MXFP4 quantization.
    
    Uses Triton backend for MXFP4 MoE on H100:
    - Weights stored as uint8 packed FP4
    - Scales stored as uint8 FP8
    - Triton matmul_ogs kernel for fused GEMM
    
    For GPT-OSS: 16 experts, top-2 routing, SwiGLU activation.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        reduce_results: bool = True,
        renormalize: bool = True,
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
        self.mxfp4_block = 32
        
        # Pad intermediate size per partition
        intermediate_size_per_partition = intermediate_size // tp_size
        self.intermediate_size_padded = (
            (intermediate_size_per_partition + 63) // 64 * 64
        )
        
        # MXFP4 method (handles weight creation and swizzling)
        self.mxfp4_method: Optional["Mxfp4MoEMethod"] = None
        
        if use_mxfp4:
            from mini_vllm.layers.quantization.mxfp4 import Mxfp4MoEMethod
            self.mxfp4_method = Mxfp4MoEMethod(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size_padded * tp_size,
            )
            self.mxfp4_method.create_weights(self)
        else:
            # Non-quantized weights (bfloat16)
            E = num_experts
            N = self.intermediate_size_padded
            K = hidden_size
            
            self.w13_weight = nn.Parameter(
                torch.empty(E, 2 * N, K, dtype=dtype), requires_grad=False
            )
            self.w2_weight = nn.Parameter(
                torch.empty(E, K, N, dtype=dtype), requires_grad=False
            )
            self.w13_bias = nn.Parameter(
                torch.zeros(E, 2 * N, dtype=dtype), requires_grad=False
            )
            self.w2_bias = nn.Parameter(
                torch.zeros(E, K, dtype=dtype), requires_grad=False
            )
    
    def process_weights_after_loading(self):
        """Process weights after loading - swizzle for Triton."""
        if self.use_mxfp4 and self.mxfp4_method is not None:
            self.mxfp4_method.process_weights_after_loading(self)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with MXFP4 Triton backend.
        
        Args:
            hidden_states: [num_tokens, hidden_size] bfloat16
            router_logits: [num_tokens, num_experts] router scores
            
        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        if self.use_mxfp4:
            return self._forward_triton_mxfp4(hidden_states, router_logits)
        else:
            return self._forward_fp16(hidden_states, router_logits)
    
    def _forward_triton_mxfp4(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with Triton MXFP4 backend (matches vLLM logs)."""
        from mini_vllm.layers.fused_moe.triton_moe import triton_kernel_moe_forward
        
        # Get quantization config
        quant_config = self.mxfp4_method.get_quant_config(self)
        
        return triton_kernel_moe_forward(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            quant_config=quant_config,
            global_num_experts=self.num_experts,
            expert_map=None,
        )
    
    def _forward_fp16(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback FP16 forward (non-quantized)."""
        batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Softmax routing
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)
        
        # Process experts
        output = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        E = self.num_experts
        N = self.intermediate_size_padded
        
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]
            weights = topk_weights[:, k:k+1]
            
            for e in range(E):
                mask = (expert_ids == e)
                if not mask.any():
                    continue
                
                x_e = hidden_states[mask]
                
                # Gate+up projection
                h = F.linear(x_e, self.w13_weight[e])
                if self.w13_bias is not None:
                    h = h + self.w13_bias[e]
                
                # SwiGLU
                gate, up = h.chunk(2, dim=-1)
                h = F.silu(gate) * up
                
                # Down projection
                out = F.linear(h, self.w2_weight[e])
                if self.w2_bias is not None:
                    out = out + self.w2_bias[e]
                
                output[mask] += weights[mask] * out
        
        return output

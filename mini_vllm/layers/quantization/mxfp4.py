# SPDX-License-Identifier: Apache-2.0
# Mini-vLLM MXFP4 Quantization for GPT-OSS
"""
MXFP4 (Microscaling FP4) quantization support.
This provides 4-bit weight quantization with FP8 scales.
"""
from typing import Optional
import torch
from torch.nn.parameter import Parameter

import logging

logger = logging.getLogger(__name__)


class Mxfp4Config:
    """Configuration for MXFP4 quantization."""
    
    def __init__(self):
        self.quant_method = "mxfp4"
        self.block_size = 32  # MXFP4 uses 32-element blocks
    
    @classmethod
    def from_config(cls, config: dict) -> "Mxfp4Config":
        """Create from model config."""
        return cls()
    
    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"


class Mxfp4MoEMethod:
    """MXFP4 quantization method for MoE layers."""
    
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.block_size = 32  # MXFP4 block size
        
        # Pad intermediate size to multiple of block size
        self.intermediate_size_padded = (
            (intermediate_size + self.block_size - 1) // self.block_size * self.block_size
        )
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        tp_size: int = 1,
    ):
        """Create quantized weight buffers."""
        E = self.num_experts
        N = self.intermediate_size_padded // tp_size  # Per-partition intermediate
        K = self.hidden_size
        block = self.block_size
        
        # w13 (gate + up, column parallel): [E, 2*N, K/2] uint8 (packed FP4)
        # Each uint8 holds 2 FP4 values
        w13_weight = Parameter(
            torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        
        # w13 scales: [E, 2*N, K/block] uint8 (FP8)
        w13_weight_scale = Parameter(
            torch.zeros(E, 2 * N, K // block, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        
        # w13 bias: [E, 2*N] bfloat16
        w13_bias = Parameter(
            torch.zeros(E, 2 * N, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        
        # w2 (down, row parallel): [E, K, N/2] uint8 (packed FP4)
        w2_weight = Parameter(
            torch.zeros(E, K, N // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        
        # w2 scales: [E, K, N/block] uint8 (FP8)
        w2_weight_scale = Parameter(
            torch.zeros(E, K, N // block, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        
        # w2 bias: [E, K] bfloat16
        w2_bias = Parameter(
            torch.zeros(E, K, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        
        logger.info(
            f"Created MXFP4 MoE weights: E={E}, N={N}, K={K}, "
            f"w13=[{E}, {2*N}, {K//2}], w2=[{E}, {K}, {N//2}]"
        )
    
    def dequantize_mxfp4(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize MXFP4 weights to floating point.
        
        MXFP4 format:
        - Each uint8 contains 2 FP4 values (4 bits each)
        - FP4 = sign(1) + exponent(2) + mantissa(1)
        - Scale is FP8 per block of 32 elements
        
        Args:
            weight: [E, M, N/2] uint8 packed weights
            scale: [E, M, N/block] uint8 FP8 scales
            dtype: Output dtype (bfloat16)
        
        Returns:
            Dequantized weights [E, M, N]
        """
        E, M, N_half = weight.shape
        N = N_half * 2
        block = self.block_size
        
        # Unpack FP4 values from uint8
        # Lower 4 bits and upper 4 bits
        low = (weight & 0x0F).to(torch.int8)
        high = ((weight >> 4) & 0x0F).to(torch.int8)
        
        # Interleave low and high
        unpacked = torch.stack([low, high], dim=-1).view(E, M, N)
        
        # Convert FP4 to float
        # FP4: sign(1) + exp(2) + mantissa(1)
        # exp=0: subnormal, value = (-1)^s * 2^(-1) * (0.m)
        # exp=1,2,3: normal, value = (-1)^s * 2^(exp-1) * (1.m)
        sign = ((unpacked >> 3) & 1).float() * -2.0 + 1.0  # -1 or 1
        exp = (unpacked >> 1) & 0x3
        mantissa = unpacked & 0x1
        
        # Compute value
        # Subnormal (exp=0): 0.5 * mantissa
        # Normal: 2^(exp-1) * (1 + 0.5*mantissa)
        is_subnormal = (exp == 0)
        exp_float = exp.float()
        mant_float = mantissa.float()
        
        value = torch.where(
            is_subnormal,
            0.5 * mant_float,
            torch.pow(2.0, exp_float - 1.0) * (1.0 + 0.5 * mant_float)
        )
        dequant = sign * value
        
        # Apply scales (FP8 E4M3)
        # Expand scale to match weight shape
        scale_expanded = scale.repeat_interleave(block, dim=-1)
        if scale_expanded.shape[-1] > N:
            scale_expanded = scale_expanded[..., :N]
        
        # Convert FP8 scale to float
        scale_float = self._fp8_to_float(scale_expanded)
        
        # Apply scale
        result = dequant * scale_float
        
        return result.to(dtype)
    
    def _fp8_to_float(self, x: torch.Tensor) -> torch.Tensor:
        """Convert FP8 E4M3 to float."""
        # FP8 E4M3: sign(1) + exp(4) + mantissa(3)
        sign = ((x >> 7) & 1).float() * -2.0 + 1.0
        exp = ((x >> 3) & 0xF).int()
        mant = (x & 0x7).float() / 8.0
        
        # Compute value: 2^(exp-7) * (1 + mant) for exp > 0
        # For exp=0: subnormal
        is_subnormal = (exp == 0)
        result = torch.where(
            is_subnormal,
            mant / 64.0,  # 2^(-6) * mant
            torch.pow(2.0, exp.float() - 7.0) * (1.0 + mant)
        )
        return sign * result
    
    def forward(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with MXFP4 dequantization.
        
        This is a fallback implementation that dequantizes on-the-fly.
        For production, use Triton kernels for fused MXFP4 matmul.
        """
        batch_size, hidden_size = x.shape
        device = x.device
        dtype = x.dtype
        
        # Dequantize weights
        w13 = self.dequantize_mxfp4(layer.w13_weight, layer.w13_weight_scale, dtype)
        w2 = self.dequantize_mxfp4(layer.w2_weight, layer.w2_weight_scale, dtype)
        
        # Get dimensions
        E, N2, K = w13.shape  # N2 = 2*N
        N = N2 // 2
        
        # Initialize output
        output = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Process each expert
        topk = topk_weights.shape[1]
        for k in range(topk):
            expert_ids = topk_ids[:, k]  # [batch]
            weights = topk_weights[:, k:k+1]  # [batch, 1]
            
            # Gather expert weights for this batch
            for e in range(E):
                mask = (expert_ids == e)
                if not mask.any():
                    continue
                
                x_e = x[mask]  # [num_tokens, K]
                
                # First matmul: x @ w13^T -> [num_tokens, 2*N]
                w13_e = w13[e].T  # [K, 2*N]
                h = x_e @ w13_e + layer.w13_bias[e]
                
                # SwiGLU activation
                gate = h[:, :N]
                up = h[:, N:]
                h = torch.nn.functional.silu(gate) * up
                
                # Second matmul: h @ w2^T -> [num_tokens, K]
                w2_e = w2[e].T  # [N, K]
                out = h @ w2_e + layer.w2_bias[e]
                
                # Weighted sum
                output[mask] += weights[mask] * out
        
        return output


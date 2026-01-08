# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM for mini-vllm GPT-OSS MXFP4 support
"""MXFP4 quantization support for GPT-OSS on H100."""

import logging
from typing import Optional, Any

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

# Import triton_kernels for weight swizzling
try:
    import mini_vllm.third_party.triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from mini_vllm.third_party.triton_kernels.numerics import InFlexData
    from mini_vllm.third_party.triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from mini_vllm.third_party.triton_kernels.tensor_details import layout
    from mini_vllm.third_party.triton_kernels.tensor_details.layout import StridedLayout
    from mini_vllm.third_party.triton_kernels.matmul_ogs import PrecisionConfig, FlexCtx
    TRITON_KERNELS_AVAILABLE = True
except ImportError as e:
    TRITON_KERNELS_AVAILABLE = False
    logger.warning(f"triton_kernels not available: {e}")


def swizzle_mxfp4_weights(
    quant_tensor: torch.Tensor,
    scale: torch.Tensor,
    num_warps: int = 8,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    """
    Swizzle MXFP4 weights for optimal Triton kernel performance.
    
    Args:
        quant_tensor: [E, N, K/2] uint8 packed FP4 weights
        scale: [E, N, K/32] uint8 FP8 scales
        num_warps: Number of warps for kernel tuning
    
    Returns:
        Tuple of (swizzled_weight, flex_ctx, swizzled_scale)
    """
    if not TRITON_KERNELS_AVAILABLE:
        raise RuntimeError("triton_kernels required for MXFP4 weight swizzling")
    
    # For H100 (SM90), use default matmul layouts
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )
    
    # Set SM90 constraints
    constraints = {"split_k": 1}
    opt_flags.update_opt_flags_constraints(constraints)
    
    # Transpose for quantization axis on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    
    # Convert layout for optimized memory access
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    
    return quant_tensor, InFlexData(), scale


class Mxfp4MoEMethod:
    """MXFP4 MoE method using Triton backend for H100."""
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mxfp4_block = 32
        
        # Pad intermediate size to multiple of 64 for Triton
        self.intermediate_size_padded = (
            (intermediate_size + 63) // 64 * 64
        )
        
        # Will be set after weight processing
        self.w13_precision_config: Optional[PrecisionConfig] = None
        self.w2_precision_config: Optional[PrecisionConfig] = None
        self.w13_weight: Optional[torch.Tensor] = None
        self.w2_weight: Optional[torch.Tensor] = None
    
    def create_weights(self, layer: torch.nn.Module):
        """Create MXFP4 weight buffers."""
        E = self.num_experts
        N = self.intermediate_size_padded
        K = self.hidden_size
        block = self.mxfp4_block
        
        # w13 (gate + up, column parallel): [E, 2*N, K/2] uint8
        w13_weight = Parameter(
            torch.zeros(E, 2 * N, K // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        
        # w13 scales: [E, 2*N, K/block] uint8
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
        
        # w2 (down, row parallel): [E, K, N/2] uint8
        w2_weight = Parameter(
            torch.zeros(E, K, N // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        
        # w2 scales: [E, K, N/block] uint8
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
        
        logger.info("Using Triton backend")
        logger.info(
            f"Created MXFP4 MoE weights: E={E}, N={N}, K={K}, "
            f"w13=[{E}, {2*N}, {K//2}], w2=[{E}, {K}, {N//2}]"
        )
    
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Swizzle weights after loading for optimal Triton kernel performance."""
        logger.info("Processing MXFP4 weights with Triton backend...")
        
        # Convert biases to float32 (required by Triton kernels)
        w13_bias = layer.w13_bias.to(torch.float32)
        w2_bias = layer.w2_bias.to(torch.float32)
        
        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)
        
        # Determine num_warps based on batch size
        num_warps = 8  # Default for H100
        
        # Swizzle w13 weights
        w13_weight, w13_flex, w13_scale = swizzle_mxfp4_weights(
            layer.w13_weight, layer.w13_weight_scale, num_warps
        )
        
        # Swizzle w2 weights
        w2_weight, w2_flex, w2_scale = swizzle_mxfp4_weights(
            layer.w2_weight, layer.w2_weight_scale, num_warps
        )
        
        # Create precision configs
        self.w13_precision_config = PrecisionConfig(
            weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
        )
        self.w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
        )
        
        # Store swizzled weights
        self.w13_weight = w13_weight
        self.w2_weight = w2_weight
        
        # Update layer weights
        del layer.w13_weight
        del layer.w2_weight
        layer.w13_weight = w13_weight
        layer.w2_weight = w2_weight
        
        logger.info("MXFP4 weight processing complete (Triton backend)")
    
    def get_quant_config(self, layer: torch.nn.Module):
        """Get quantization config for Triton MoE forward."""
        from mini_vllm.layers.fused_moe.triton_moe import Mxfp4QuantConfig
        
        return Mxfp4QuantConfig(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_precision=self.w13_precision_config,
            w2_precision=self.w2_precision_config,
            use_mxfp4_w4a16=True,
        )

# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM for mini-vllm GPT-OSS MXFP4 support
"""Triton-based MXFP4 MoE kernels for GPT-OSS."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Import vendored triton_kernels
try:
    import mini_vllm.third_party.triton_kernels.swiglu as triton_swiglu
    from mini_vllm.third_party.triton_kernels.matmul_ogs import (
        FnSpecs, FusedActivation, matmul_ogs, PrecisionConfig
    )
    from mini_vllm.third_party.triton_kernels.routing import (
        RoutingData, routing, routing_from_bitmatrix
    )
    from mini_vllm.third_party.triton_kernels.tensor import Bitmatrix
    TRITON_KERNELS_AVAILABLE = True
    logger.info("Loaded triton_kernels from mini_vllm.third_party")
except (AttributeError, ImportError) as e:
    TRITON_KERNELS_AVAILABLE = False
    logger.warning(f"Failed to import triton_kernels: {e}")


@dataclass
class Mxfp4QuantConfig:
    """MXFP4 quantization config for MoE."""
    w1_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None
    w1_precision: Optional["PrecisionConfig"] = None
    w2_precision: Optional["PrecisionConfig"] = None
    use_mxfp4_w4a16: bool = True


def _resize_cache(cache: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Resize cache tensor to target shape."""
    numel = 1
    for s in shape:
        numel *= s
    if cache.numel() < numel:
        return cache.new_empty(shape)
    return cache.view(-1)[:numel].view(shape)


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Packs topk_ids into a bitmatrix for routing."""
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple:
    """Create routing data structures for matmul_ogs."""
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix_obj = Bitmatrix(
        bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None
    )

    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
        bitmatrix_obj, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    quant_config: Optional[Mxfp4QuantConfig] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Main entry point for Triton MXFP4 MoE forward pass.
    
    Args:
        hidden_states: [num_tokens, hidden_size] bfloat16
        w1: MXFP4 gate+up weights (swizzled)
        w2: MXFP4 down weights (swizzled)
        gating_output: [num_tokens, num_experts] router logits
        topk: Number of experts per token
        renormalize: Whether to renormalize routing weights
        quant_config: MXFP4 quantization config with bias and precision
        global_num_experts: Total number of experts
        expert_map: Optional mapping for expert parallelism
    
    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    if not TRITON_KERNELS_AVAILABLE:
        raise RuntimeError("triton_kernels not available")
    
    routing_data, gather_idx, scatter_idx = routing(
        gating_output, topk, sm_first=not renormalize
    )

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        quant_config=quant_config,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    quant_config: Optional[Mxfp4QuantConfig] = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    intermediate_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused MoE experts using Triton matmul_ogs kernel.
    """
    if quant_config is None:
        quant_config = Mxfp4QuantConfig()

    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.ndim == 2

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    # SwiGLU activation for GPT-OSS
    act = FusedActivation(
        FnSpecs("swiglu", triton_swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    # First GEMM: hidden @ w1 (gate+up)
    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    # Second GEMM: intermediate @ w2 (down)
    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    
    output_tensor = output_tensor.view(M, K)
    return output_tensor

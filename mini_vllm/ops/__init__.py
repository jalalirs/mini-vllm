# SPDX-License-Identifier: Apache-2.0
"""
Mini-vLLM CUDA Operations

Loads the compiled CUDA kernel library (_C.so) and exposes ops to Python.
The library is built via CMake (not pip):
    mkdir build && cd build
    cmake ..
    make -j
"""

import os
import torch
from pathlib import Path
from typing import Optional, List

# =============================================================================
# Load the compiled CUDA library
# =============================================================================
_LIB_PATH = Path(__file__).parent.parent / "_C.so"
_LIB = None
HAS_CUDA_OPS = False

def _load_library():
    """Load the compiled CUDA library."""
    global _LIB, HAS_CUDA_OPS
    
    if _LIB is not None:
        return _LIB
    
    lib_paths = [
        _LIB_PATH,
        Path(__file__).parent.parent / "_C.cpython-311-x86_64-linux-gnu.so",
        Path(__file__).parent.parent / "_C.cpython-310-x86_64-linux-gnu.so",
    ]
    
    for path in lib_paths:
        if path.exists():
            try:
                torch.ops.load_library(str(path))
                _LIB = path
                HAS_CUDA_OPS = True
                print(f"[mini_vllm] Loaded CUDA ops from: {path}")
                return _LIB
            except Exception as e:
                print(f"[mini_vllm] Failed to load {path}: {e}")
    
    print("[mini_vllm] WARNING: CUDA ops not found. Using PyTorch fallbacks.")
    print(f"[mini_vllm] Searched: {[str(p) for p in lib_paths]}")
    return None

# Try to load on import
_load_library()


# =============================================================================
# Attention Operations
# =============================================================================

def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    """PagedAttention V1 - single-pass attention with paged KV cache."""
    if not HAS_CUDA_OPS:
        raise RuntimeError("CUDA ops not compiled. Run: mkdir build && cd build && cmake .. && make")
    
    if k_scale is None:
        k_scale = torch.tensor([1.0], device=query.device)
    if v_scale is None:
        v_scale = torch.tensor([1.0], device=query.device)
    
    torch.ops.mini_vllm_ops.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale,
        block_tables, seq_lens, block_size, max_seq_len, alibi_slopes,
        kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step
    )


def paged_attention_v2(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    """PagedAttention V2 - two-pass attention for memory efficiency."""
    if not HAS_CUDA_OPS:
        raise RuntimeError("CUDA ops not compiled")
    
    if k_scale is None:
        k_scale = torch.tensor([1.0], device=query.device)
    if v_scale is None:
        v_scale = torch.tensor([1.0], device=query.device)
    
    torch.ops.mini_vllm_ops.paged_attention_v2(
        out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step
    )


# =============================================================================
# LayerNorm Operations
# =============================================================================

def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """RMS Normalization using CUDA kernel."""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.rms_norm(out, input, weight, epsilon)
    else:
        # PyTorch fallback
        variance = input.pow(2).mean(-1, keepdim=True)
        input_norm = input * torch.rsqrt(variance + epsilon)
        out.copy_(input_norm * weight)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Fused Add + RMS Normalization (in-place)."""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.fused_add_rms_norm(input, residual, weight, epsilon)
    else:
        # PyTorch fallback
        input.add_(residual)
        residual.copy_(input)
        variance = input.pow(2).mean(-1, keepdim=True)
        input.mul_(torch.rsqrt(variance + epsilon))
        input.mul_(weight)


# =============================================================================
# Activation Operations (SwiGLU for MoE)
# =============================================================================

def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """SiLU and Multiply: silu(x[..., :d]) * x[..., d:]"""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.silu_and_mul(out, input)
    else:
        # PyTorch fallback
        d = input.shape[-1] // 2
        gate = input[..., :d]
        up = input[..., d:]
        out.copy_(torch.nn.functional.silu(gate) * up)


def gelu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
    """GELU and Multiply: gelu(x[..., :d]) * x[..., d:]"""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.gelu_and_mul(out, input)
    else:
        # PyTorch fallback
        d = input.shape[-1] // 2
        gate = input[..., :d]
        up = input[..., d:]
        out.copy_(torch.nn.functional.gelu(gate) * up)


# =============================================================================
# Positional Encoding
# =============================================================================

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Apply Rotary Position Embedding (RoPE)."""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    else:
        # PyTorch fallback
        seq_len = positions.max().item() + 1
        cos = cos_sin_cache[:seq_len, :head_size // 2]
        sin = cos_sin_cache[:seq_len, head_size // 2:]
        
        cos = cos[positions].unsqueeze(-2)
        sin = sin[positions].unsqueeze(-2)
        
        q1, q2 = query[..., :head_size // 2], query[..., head_size // 2:]
        query[..., :head_size // 2] = q1 * cos - q2 * sin
        query[..., head_size // 2:] = q1 * sin + q2 * cos
        
        if key is not None:
            k1, k2 = key[..., :head_size // 2], key[..., head_size // 2:]
            key[..., :head_size // 2] = k1 * cos - k2 * sin
            key[..., head_size // 2:] = k1 * sin + k2 * cos


# =============================================================================
# KV Cache Operations
# =============================================================================

def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor,
) -> None:
    """Swap KV cache blocks between locations."""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.swap_blocks(src, dst, block_mapping)
    else:
        for src_idx, dst_idx in block_mapping:
            dst[dst_idx].copy_(src[src_idx])


def copy_blocks(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """Copy KV cache blocks."""
    if HAS_CUDA_OPS:
        torch.ops.mini_vllm_ops.copy_blocks(key_caches, value_caches, block_mapping)
    else:
        for src_idx, dst_idx in block_mapping:
            for key_cache in key_caches:
                key_cache[dst_idx].copy_(key_cache[src_idx])
            for value_cache in value_caches:
                value_cache[dst_idx].copy_(value_cache[src_idx])


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    """Reshape key/value and store in cache."""
    if HAS_CUDA_OPS:
        if k_scale is None:
            k_scale = torch.tensor([1.0], device=key.device)
        if v_scale is None:
            v_scale = torch.tensor([1.0], device=value.device)
        torch.ops.mini_vllm_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping, 
            kv_cache_dtype, k_scale, v_scale
        )
    else:
        num_tokens = key.shape[0]
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot >= 0:
                block_idx = slot // key_cache.shape[2]
                block_offset = slot % key_cache.shape[2]
                key_cache[block_idx, :, block_offset] = key[i]
                value_cache[block_idx, :, block_offset] = value[i]


# =============================================================================
# Utility
# =============================================================================

def cuda_ops_available() -> bool:
    """Check if CUDA ops are compiled and loaded."""
    return HAS_CUDA_OPS


def get_library_path() -> Optional[str]:
    """Get path to loaded CUDA library."""
    return str(_LIB) if _LIB else None


__all__ = [
    # Attention
    "paged_attention_v1",
    "paged_attention_v2",
    # LayerNorm
    "rms_norm",
    "fused_add_rms_norm",
    # Activation
    "silu_and_mul",
    "gelu_and_mul",
    # Positional
    "rotary_embedding",
    # Cache
    "swap_blocks",
    "copy_blocks",
    "reshape_and_cache",
    # Utils
    "cuda_ops_available",
    "get_library_path",
]

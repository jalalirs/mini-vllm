# SPDX-License-Identifier: Apache-2.0
"""Attention layer with CUDA paged attention kernels."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    """Multi-head attention with paged KV cache.
    
    Uses our compiled CUDA paged attention kernels (paged_attention_v1/v2).
    Falls back to native PyTorch for prefill or when CUDA ops unavailable.
    
    Supports:
    - Grouped Query Attention (GQA)
    - Sliding window attention
    - Paged KV cache for decode
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        num_kv_heads: Optional[int] = None,
        sliding_window: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.scale = scale or (head_dim ** -0.5)
        self.sliding_window = sliding_window
        
        # GQA: number of query heads per KV head
        assert num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Check for compiled CUDA paged attention
        self._has_paged_attn = False
        try:
            from mini_vllm.ops import paged_attention_v1, cuda_ops_available
            if cuda_ops_available():
                self._paged_attention_v1 = paged_attention_v1
                self._has_paged_attn = True
        except ImportError:
            pass
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: [num_tokens, num_heads * head_dim]
            key: [num_tokens, num_kv_heads * head_dim]
            value: [num_tokens, num_kv_heads * head_dim]
            kv_cache: Optional KV cache tensor
            attn_metadata: Optional metadata (block_tables, seq_lens, etc.)
            
        Returns:
            output: [num_tokens, num_heads * head_dim]
        """
        num_tokens = query.shape[0]
        
        # Reshape to [num_tokens, num_heads, head_dim]
        query = query.view(num_tokens, self.num_heads, self.head_dim)
        key = key.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = value.view(num_tokens, self.num_kv_heads, self.head_dim)
        
        # Decode with paged KV cache - use CUDA kernel
        if kv_cache is not None and attn_metadata is not None and self._has_paged_attn:
            output = self._forward_paged(query, kv_cache, attn_metadata)
        # Prefill or fallback - use native PyTorch
        else:
            output = self._forward_native(query, key, value)
        
        # Reshape back to [num_tokens, num_heads * head_dim]
        return output.view(num_tokens, self.num_heads * self.head_dim)
    
    def _forward_paged(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """Paged attention forward using our compiled CUDA kernel.
        
        Args:
            query: [num_tokens, num_heads, head_dim]
            kv_cache: [num_blocks, 2, num_kv_heads, block_size, head_dim]
            attn_metadata: dict with block_tables, seq_lens, etc.
        """
        num_tokens = query.shape[0]
        
        # Split KV cache
        key_cache = kv_cache[:, 0]  # [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache = kv_cache[:, 1]
        
        # Prepare output
        output = torch.empty(
            num_tokens, self.num_heads, self.head_dim,
            device=query.device, dtype=query.dtype
        )
        
        # Call our CUDA kernel
        self._paged_attention_v1(
            out=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            block_tables=attn_metadata["block_tables"],
            seq_lens=attn_metadata["seq_lens"],
            block_size=attn_metadata.get("block_size", 16),
            max_seq_len=attn_metadata.get("max_seq_len", 2048),
        )
        
        return output
    
    def _forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Native PyTorch attention (for prefill or fallback)."""
        num_tokens = query.shape[0]
        
        # Expand KV heads for GQA
        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # [num_tokens, num_heads, head_dim] -> [num_heads, num_tokens, head_dim]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply sliding window mask
        if self.sliding_window is not None:
            window_mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=query.device, dtype=torch.bool),
                diagonal=-self.sliding_window,
            )
            window_mask = ~window_mask
            scores = scores.masked_fill(window_mask, float('-inf'))
        
        # Softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        # [num_heads, num_tokens, head_dim] -> [num_tokens, num_heads, head_dim]
        return output.transpose(0, 1)

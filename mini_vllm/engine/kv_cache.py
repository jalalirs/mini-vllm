# SPDX-License-Identifier: Apache-2.0
"""KV cache management."""

import torch
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class KVCache:
    """KV cache for a single layer."""
    
    key_cache: torch.Tensor  # [max_seq_len, num_kv_heads, head_dim]
    value_cache: torch.Tensor  # [max_seq_len, num_kv_heads, head_dim]
    seq_len: int = 0  # Current sequence length
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new keys/values.
        
        Args:
            key: [batch_size, num_kv_heads, head_dim]
            value: [batch_size, num_kv_heads, head_dim]
            positions: [batch_size] position indices
            
        Returns:
            Full key and value tensors including cached values
        """
        # Store new values
        for i, pos in enumerate(positions):
            self.key_cache[pos] = key[i]
            self.value_cache[pos] = value[i]
        
        # Update sequence length
        max_pos = positions.max().item() + 1
        self.seq_len = max(self.seq_len, max_pos)
        
        # Return full cached tensors up to current length
        return self.key_cache[:self.seq_len], self.value_cache[:self.seq_len]
    
    def reset(self):
        """Reset cache for new sequence."""
        self.seq_len = 0


class KVCacheManager:
    """Manages KV cache across all layers."""
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        gpu_memory_utilization: float = 0.90,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        # Calculate memory and allocate
        self._calculate_cache_size(gpu_memory_utilization)
        self.caches: List[KVCache] = []
        
        # Pre-allocate caches for all layers
        self._allocate_caches()
    
    def _calculate_cache_size(self, gpu_memory_utilization: float):
        """Calculate KV cache size based on available memory."""
        if not torch.cuda.is_available():
            return
        
        # Get available memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        available_memory = total_memory - allocated_memory
        
        # Memory for KV cache (conservative estimate)
        kv_cache_memory = available_memory * gpu_memory_utilization
        
        # Each layer needs 2 (K+V) * max_seq_len * num_kv_heads * head_dim * dtype_size
        dtype_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        per_layer_memory = 2 * self.max_seq_len * self.num_kv_heads * self.head_dim * dtype_size
        total_cache_memory = per_layer_memory * self.num_layers
        
        if total_cache_memory > kv_cache_memory:
            # Reduce max_seq_len to fit
            ratio = kv_cache_memory / total_cache_memory
            self.max_seq_len = int(self.max_seq_len * ratio)
            print(f"Reduced max_seq_len to {self.max_seq_len} to fit GPU memory")
    
    def _allocate_caches(self):
        """Allocate KV caches for all layers."""
        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                self.max_seq_len, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            value_cache = torch.zeros(
                self.max_seq_len, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            self.caches.append(KVCache(key_cache, value_cache))
    
    def get_cache(self, layer_idx: int) -> KVCache:
        """Get KV cache for a specific layer."""
        return self.caches[layer_idx]
    
    def reset_all(self):
        """Reset all caches for new batch."""
        for cache in self.caches:
            cache.reset()
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage stats."""
        total_cache_size = 0
        for cache in self.caches:
            total_cache_size += cache.key_cache.numel() * cache.key_cache.element_size()
            total_cache_size += cache.value_cache.numel() * cache.value_cache.element_size()
        
        return {
            "total_cache_bytes": total_cache_size,
            "total_cache_mb": total_cache_size / (1024 * 1024),
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
        }


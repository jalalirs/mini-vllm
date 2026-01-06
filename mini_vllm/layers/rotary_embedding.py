# SPDX-License-Identifier: Apache-2.0
"""Rotary Position Embeddings (RoPE) with YARN scaling."""

import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn


def _compute_inv_freq(
    base: float,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse frequencies for RoPE."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    return inv_freq


def _compute_yarn_inv_freq(
    base: float,
    dim: int,
    device: torch.device,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Compute YARN-scaled inverse frequencies.
    
    YARN (Yet Another RoPE extensioN) allows extending context length
    while maintaining quality on shorter sequences.
    """
    # Base frequencies
    inv_freq = _compute_inv_freq(base, dim, device)
    
    # Compute low and high frequency cutoffs
    low_freq_wavelen = original_max_position_embeddings / beta_slow
    high_freq_wavelen = original_max_position_embeddings / beta_fast
    
    # Wavelengths for each frequency
    old_context_len = original_max_position_embeddings
    wavelen = 2 * math.pi / inv_freq
    
    # Smooth ramp between low and high frequencies
    smooth = (old_context_len / wavelen - beta_slow) / (beta_fast - beta_slow)
    smooth = torch.clamp(smooth, 0.0, 1.0)
    
    # Interpolate between original and scaled frequencies
    inv_freq_scaled = inv_freq / factor
    inv_freq = torch.lerp(inv_freq_scaled, inv_freq, smooth)
    
    return inv_freq


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.
    
    Implements RoPE which rotates query/key vectors based on position.
    Supports YARN scaling for extended context lengths.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling
        self.dtype = dtype
        
        # Compute inverse frequencies
        if rope_scaling and rope_scaling.get("rope_type") == "yarn":
            inv_freq = _compute_yarn_inv_freq(
                base=rope_scaling.get("rope_theta", base),
                dim=head_dim,
                device=torch.device("cpu"),
                factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                beta_fast=rope_scaling["beta_fast"],
                beta_slow=rope_scaling["beta_slow"],
            )
        else:
            inv_freq = _compute_inv_freq(base, head_dim, torch.device("cpu"))
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin values
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0
    
    def _update_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Update cos/sin cache if needed."""
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        
        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        
        # Compute position indices
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        
        # Compute freqs: [seq_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Compute cos and sin: [seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.
        
        Args:
            positions: Position indices [batch, seq_len] or [seq_len]
            query: Query tensor [..., head_dim]
            key: Key tensor [..., head_dim]
            
        Returns:
            Rotated query and key tensors
        """
        seq_len = positions.max().item() + 1
        self._update_cache(int(seq_len), positions.device, query.dtype)
        
        # Get cos/sin for positions
        cos = self._cos_cached[positions]
        sin = self._sin_cached[positions]
        
        # Reshape for broadcasting
        while cos.dim() < query.dim():
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        
        # Apply rotation
        query_rot = self._rotate(query, cos, sin)
        key_rot = self._rotate(key, cos, sin)
        
        return query_rot, key_rot
    
    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary transformation."""
        # Split into halves
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        # Rotate
        rotated = torch.cat((-x2, x1), dim=-1)
        
        return x * cos + rotated * sin


def get_rope(
    head_dim: int,
    max_position: int,
    base: float = 10000.0,
    rope_scaling: Optional[dict] = None,
    **kwargs,
) -> RotaryEmbedding:
    """Factory function for rotary embeddings."""
    return RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_position,
        base=base,
        rope_scaling=rope_scaling,
    )


# SPDX-License-Identifier: Apache-2.0
"""Layer normalization implementations with CUDA kernel support."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    
    Uses optimized CUDA kernel when available, falls back to PyTorch.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        
        # Check for CUDA ops
        self._has_cuda_ops = False
        try:
            from mini_vllm.ops import rms_norm, fused_add_rms_norm, cuda_ops_available
            if cuda_ops_available():
                self._rms_norm = rms_norm
                self._fused_add_rms_norm = fused_add_rms_norm
                self._has_cuda_ops = True
        except ImportError:
            pass
    
    def _norm_native(self, x: torch.Tensor) -> torch.Tensor:
        """Native PyTorch RMS normalization."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(orig_dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        If residual is provided, computes fused add + norm:
            residual = x + residual
            output = norm(residual) * weight
            return output, residual
            
        Otherwise:
            return norm(x) * weight
        """
        if residual is not None:
            # Fused residual add + norm
            if self._has_cuda_ops and x.is_cuda:
                # Use CUDA kernel (in-place)
                x = x + residual
                residual = x.clone()
                out = torch.empty_like(x)
                self._rms_norm(out, x, self.weight, self.eps)
                return out, residual
            else:
                # Native PyTorch
                x = x + residual
                residual = x
                output = self._norm_native(x) * self.weight
                return output, residual
        else:
            # Standalone norm
            if self._has_cuda_ops and x.is_cuda:
                out = torch.empty_like(x)
                self._rms_norm(out, x, self.weight, self.eps)
                return out
            else:
                return self._norm_native(x) * self.weight

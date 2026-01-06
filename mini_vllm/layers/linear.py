# SPDX-License-Identifier: Apache-2.0
"""Linear layers with tensor parallelism support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mini_vllm.distributed import (
    get_tensor_parallel_world_size,
    tensor_parallel_all_reduce,
)


class RowParallelLinear(nn.Module):
    """Linear with row parallelism.
    
    Splits input features across tensor parallel ranks.
    Y = XA where A is partitioned row-wise, requires all-reduce.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        tp_size = get_tensor_parallel_world_size()
        
        self.input_size = input_size
        self.output_size = output_size
        
        assert input_size % tp_size == 0
        self.input_size_per_partition = input_size // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = F.linear(x, self.weight)
        
        # All-reduce across tensor parallel group
        if get_tensor_parallel_world_size() > 1:
            output = tensor_parallel_all_reduce(output)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output, None


class QKVParallelLinear(nn.Module):
    """Fused QKV projection with tensor parallelism.
    
    Computes Q, K, V in one kernel. Supports GQA where num_kv_heads < num_heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        tp_size = get_tensor_parallel_world_size()
        
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # Per-partition sizes
        assert num_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0
        
        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = num_kv_heads // tp_size
        
        self.q_size = self.num_heads_per_partition * head_size
        self.kv_size = self.num_kv_heads_per_partition * head_size
        
        total_output_size = self.q_size + 2 * self.kv_size
        
        self.weight = nn.Parameter(
            torch.empty(total_output_size, hidden_size, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(total_output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        qkv = F.linear(x, self.weight, self.bias)
        return qkv, None

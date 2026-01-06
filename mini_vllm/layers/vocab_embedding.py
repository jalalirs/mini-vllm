# SPDX-License-Identifier: Apache-2.0
"""Vocabulary embeddings with tensor parallelism."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.distributed import (
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
    tensor_parallel_all_reduce,
)


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized across vocabulary dimension.
    
    Each rank holds vocab_size // tp_size embeddings.
    Uses masking for out-of-range tokens and all-reduce to combine.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Partition vocabulary
        assert num_embeddings % tp_size == 0
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.vocab_start_idx = tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, dtype=dtype)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            # Mask tokens outside this partition
            mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
            masked_input = input_ids - self.vocab_start_idx
            masked_input = torch.where(mask, masked_input, torch.zeros_like(masked_input))
            
            # Lookup
            output = F.embedding(masked_input, self.weight)
            
            # Zero out masked positions
            output = output * mask.unsqueeze(-1).to(output.dtype)
            
            # All-reduce to combine partitions
            output = tensor_parallel_all_reduce(output)
        else:
            output = F.embedding(input_ids, self.weight)
        
        return output


class ParallelLMHead(nn.Module):
    """Language model head parallelized across vocabulary.
    
    Linear projection from hidden_size to vocab_size, partitioned.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Partition vocabulary
        assert num_embeddings % tp_size == 0
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.vocab_start_idx = tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.num_embeddings_per_partition, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits.
        
        Returns partial logits (needs all-gather for full vocab).
        """
        logits = F.linear(hidden_states, self.weight, self.bias)
        return logits


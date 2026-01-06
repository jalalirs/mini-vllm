# SPDX-License-Identifier: Apache-2.0
"""Distributed parallel state management for tensor parallelism."""

import os
from typing import Optional

import torch
import torch.distributed as dist


# Global state
_TENSOR_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_TENSOR_PARALLEL_RANK: int = 0
_TENSOR_PARALLEL_WORLD_SIZE: int = 1
_LOCAL_RANK: int = 0
_INITIALIZED: bool = False


def is_initialized() -> bool:
    """Check if distributed is initialized."""
    return _INITIALIZED


def init_distributed(
    tensor_parallel_size: int = 1,
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> None:
    """Initialize distributed process groups.
    
    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        backend: PyTorch distributed backend (nccl for GPU)
        init_method: Process group initialization method
    """
    global _TENSOR_PARALLEL_GROUP, _TENSOR_PARALLEL_RANK
    global _TENSOR_PARALLEL_WORLD_SIZE, _LOCAL_RANK, _INITIALIZED
    
    if _INITIALIZED:
        return
    
    if tensor_parallel_size == 1:
        # Single GPU mode
        _TENSOR_PARALLEL_RANK = 0
        _TENSOR_PARALLEL_WORLD_SIZE = 1
        _LOCAL_RANK = 0
        _INITIALIZED = True
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return
    
    # Multi-GPU mode
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", tensor_parallel_size)))
    
    # Verify world size matches tensor parallel size
    assert world_size == tensor_parallel_size, (
        f"World size ({world_size}) must equal tensor_parallel_size ({tensor_parallel_size})"
    )
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    
    # Create tensor parallel group (all ranks in same group for now)
    ranks = list(range(world_size))
    _TENSOR_PARALLEL_GROUP = dist.new_group(ranks)
    _TENSOR_PARALLEL_RANK = rank
    _TENSOR_PARALLEL_WORLD_SIZE = world_size
    _LOCAL_RANK = local_rank
    _INITIALIZED = True
    
    print(f"Initialized distributed: rank={rank}/{world_size}, local_rank={local_rank}")


def get_tensor_parallel_rank() -> int:
    """Get rank within tensor parallel group."""
    return _TENSOR_PARALLEL_RANK


def get_tensor_parallel_world_size() -> int:
    """Get size of tensor parallel group."""
    return _TENSOR_PARALLEL_WORLD_SIZE


def get_tensor_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get tensor parallel process group."""
    return _TENSOR_PARALLEL_GROUP


def get_local_rank() -> int:
    """Get local rank (GPU index on this node)."""
    return _LOCAL_RANK


def tensor_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across tensor parallel group.
    
    Used after row-parallel linear layers.
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    dist.all_reduce(tensor, group=_TENSOR_PARALLEL_GROUP)
    return tensor


def tensor_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across tensor parallel group.
    
    Used to reconstruct full tensors from partitions.
    
    Args:
        tensor: Local partition
        dim: Dimension to gather along
        
    Returns:
        Full tensor concatenated from all ranks
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    # Prepare output
    tensor_list = [torch.empty_like(tensor) for _ in range(_TENSOR_PARALLEL_WORLD_SIZE)]
    dist.all_gather(tensor_list, tensor, group=_TENSOR_PARALLEL_GROUP)
    
    return torch.cat(tensor_list, dim=dim)


def tensor_parallel_scatter(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Scatter tensor to get local partition.
    
    Args:
        tensor: Full tensor (used on rank 0)
        dim: Dimension to scatter along
        
    Returns:
        Local partition for this rank
    """
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    chunks = tensor.chunk(_TENSOR_PARALLEL_WORLD_SIZE, dim=dim)
    return chunks[_TENSOR_PARALLEL_RANK].contiguous()


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank."""
    if _TENSOR_PARALLEL_WORLD_SIZE == 1:
        return tensor
    
    dist.broadcast(tensor, src=src, group=_TENSOR_PARALLEL_GROUP)
    return tensor


def barrier() -> None:
    """Synchronize all processes."""
    if _TENSOR_PARALLEL_WORLD_SIZE > 1:
        dist.barrier(group=_TENSOR_PARALLEL_GROUP)


def destroy_distributed() -> None:
    """Clean up distributed state."""
    global _INITIALIZED, _TENSOR_PARALLEL_GROUP
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    _TENSOR_PARALLEL_GROUP = None
    _INITIALIZED = False


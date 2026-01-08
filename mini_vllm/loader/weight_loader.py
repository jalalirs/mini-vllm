# SPDX-License-Identifier: Apache-2.0
"""Weight loading utilities for GPT-OSS with MXFP4 support."""

import json
import os
from pathlib import Path
from typing import Optional, Iterator
import logging

import torch
from safetensors.torch import safe_open

from mini_vllm.models.gpt_oss import GptOssConfig
from mini_vllm.distributed import (
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
)

logger = logging.getLogger(__name__)


def load_config(model_path: str) -> GptOssConfig:
    """Load model configuration from path."""
    config_path = Path(model_path) / "config.json"
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"  vocab_size: {config_dict.get('vocab_size')}")
    logger.info(f"  hidden_size: {config_dict.get('hidden_size')}")
    logger.info(f"  num_hidden_layers: {config_dict.get('num_hidden_layers')}")
    logger.info(f"  num_local_experts: {config_dict.get('num_local_experts')}")
    
    quant_config = config_dict.get("quantization_config")
    if quant_config:
        logger.info(f"  quantization: {quant_config.get('quant_method', 'none')}")
    
    return GptOssConfig(**config_dict)


def iter_safetensors(model_path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate over all safetensors files in model directory."""
    model_path = Path(model_path)
    
    # Find all safetensor files
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    logger.info(f"Found {len(safetensor_files)} safetensor files")
    
    for file_path in safetensor_files:
        logger.debug(f"Loading {file_path.name}")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def load_model_weights(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Load weights into model with MXFP4 support.
    
    Args:
        model: The model to load weights into
        model_path: Path to the model directory
        device: Target device
        dtype: Target dtype for non-quantized weights
    """
    tp_rank = get_tensor_parallel_rank()
    tp_size = get_tensor_parallel_world_size()
    
    config = model.config
    use_mxfp4 = config.use_mxfp4
    
    logger.info(f"Loading weights for rank {tp_rank}/{tp_size}")
    logger.info(f"  MXFP4 quantization: {use_mxfp4}")
    
    # Build parameter mapping
    params_dict = dict(model.named_parameters())
    loaded_params = set()
    
    # MXFP4 specific values
    mxfp4_block = 32
    num_experts = config.num_local_experts
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    
    # Calculate TP slicing for MXFP4
    intermediate_size_block = intermediate_size // mxfp4_block
    per_rank_intermediate_size_block = (intermediate_size_block + tp_size - 1) // tp_size
    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block
    tp_rank_start = tp_rank * per_rank_intermediate_size
    tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)
    
    logger.info(f"  intermediate_size: {intermediate_size}")
    logger.info(f"  per_rank_intermediate_size: {per_rank_intermediate_size}")
    logger.info(f"  tp_rank_start: {tp_rank_start}, tp_rank_end: {tp_rank_end}")
    
    for name, weight in iter_safetensors(model_path):
        # Map checkpoint name to model parameter name
        param_name = _map_weight_name(name)
        
        if param_name not in params_dict:
            logger.debug(f"Skipping {name} -> {param_name} (not in model)")
            continue
        
        param = params_dict[param_name]
        
        if use_mxfp4 and _is_mxfp4_weight(name):
            # Handle MXFP4 quantized weights
            weight = _shard_mxfp4_weight(
                name=name,
                weight=weight,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_rank_start=tp_rank_start,
                tp_rank_end=tp_rank_end,
                num_experts=num_experts,
                intermediate_size=intermediate_size,
                mxfp4_block=mxfp4_block,
            )
            # Don't convert dtype for quantized weights
            param.data.copy_(weight)
        else:
            # Handle regular weights with TP sharding
            weight = _shard_weight(
                name=param_name,
                weight=weight,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            param.data.copy_(weight.to(dtype))
        
        loaded_params.add(param_name)
    
    # Move model to device
    model.to(device)
    
    logger.info(f"Loaded {len(loaded_params)} parameters")
    
    # Check for missing parameters
    missing = set(params_dict.keys()) - loaded_params
    if missing:
        logger.warning(f"Missing {len(missing)} parameters: {list(missing)[:10]}...")


def _map_weight_name(ckpt_name: str) -> str:
    """Map checkpoint weight name to model parameter name.
    
    GPT-OSS checkpoint format:
        model.layers.{i}.self_attn.q_proj.weight -> model.layers.{i}.attn.qkv_proj.weight
        model.layers.{i}.mlp.experts.w13_weight -> model.layers.{i}.mlp.experts.w13_weight
    """
    # Handle QKV projection mapping
    name = ckpt_name
    
    # Map self_attn -> attn
    name = name.replace("self_attn", "attn")
    
    # Map q_proj/k_proj/v_proj -> qkv_proj (handled specially)
    # We'll stack these during loading
    
    return name


def _is_mxfp4_weight(name: str) -> bool:
    """Check if weight is MXFP4 quantized."""
    mxfp4_patterns = [
        ".w13_weight",
        ".w2_weight", 
        ".w13_weight_scale",
        ".w2_weight_scale",
    ]
    return any(p in name for p in mxfp4_patterns)


def _shard_mxfp4_weight(
    name: str,
    weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    tp_rank_start: int,
    tp_rank_end: int,
    num_experts: int,
    intermediate_size: int,
    mxfp4_block: int,
) -> torch.Tensor:
    """Shard MXFP4 weight for tensor parallelism.
    
    MXFP4 weights have special sharding patterns:
    - w13_weight: [E, 2*N, K/2] -> shard on dim 1 (2*N)
    - w2_weight: [E, K, N/2] -> shard on dim 2 (N/2)
    - Scales follow the same patterns
    """
    if ".w13_weight_scale" in name:
        # [E, 2*N, K/block] -> shard on dim 1
        narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
        return narrow_weight.contiguous()
        
    elif ".w2_weight_scale" in name:
        # [E, K, N/block] -> shard on dim 2
        tp_start_block = tp_rank_start // mxfp4_block
        tp_end_block = tp_rank_end // mxfp4_block
        narrow_weight = weight[..., tp_start_block : tp_end_block]
        return narrow_weight.contiguous()
        
    elif ".w13_weight" in name:
        # [E, 2*N, block_size, entries] -> flatten and shard
        weight = weight.view(num_experts, 2 * intermediate_size, -1).contiguous()
        narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
        return narrow_weight.contiguous()
        
    elif ".w2_weight" in name:
        # [E, K, N/2] -> shard on last dim
        weight = weight.view(num_experts, -1, intermediate_size // 2).contiguous()
        narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]
        return narrow_weight.contiguous()
    
    return weight


def _shard_weight(
    name: str,
    weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Shard regular weight for tensor parallelism."""
    if tp_size == 1:
        return weight
    
    # Column-parallel (shard dim 0): q_proj, k_proj, v_proj, gate_proj, up_proj, embed_tokens, lm_head
    column_parallel = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head"]
    
    # Row-parallel (shard dim 1): o_proj, down_proj
    row_parallel = ["o_proj", "down_proj"]
    
    for pattern in column_parallel:
        if pattern in name:
            shard_size = weight.shape[0] // tp_size
            return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]
    
    for pattern in row_parallel:
        if pattern in name:
            shard_size = weight.shape[-1] // tp_size
            return weight[..., tp_rank * shard_size : (tp_rank + 1) * shard_size]
    
    # Router and other weights are not sharded
    return weight


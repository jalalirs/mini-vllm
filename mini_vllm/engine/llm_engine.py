# SPDX-License-Identifier: Apache-2.0
"""LLM inference engine for GPT-OSS."""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Iterator
import uuid

import torch
from transformers import AutoTokenizer

from mini_vllm.models.gpt_oss import GptOssForCausalLM, GptOssConfig
from mini_vllm.distributed import init_distributed, get_tensor_parallel_rank
from mini_vllm.engine.kv_cache import KVCacheManager


@dataclass
class SamplingParams:
    """Parameters for text generation."""
    
    max_tokens: int = 256
    """Maximum tokens to generate."""
    
    temperature: float = 1.0
    """Sampling temperature. 0 = greedy."""
    
    top_p: float = 1.0
    """Nucleus sampling probability."""
    
    top_k: int = -1
    """Top-k sampling. -1 = disabled."""
    
    stop: Optional[List[str]] = None
    """Stop strings."""
    
    stop_token_ids: Optional[List[int]] = None
    """Stop token IDs."""


@dataclass
class GenerationOutput:
    """Output from text generation."""
    
    request_id: str
    """Unique request identifier."""
    
    prompt: str
    """Original prompt."""
    
    text: str
    """Generated text."""
    
    token_ids: List[int] = field(default_factory=list)
    """Generated token IDs."""
    
    finished: bool = False
    """Whether generation is complete."""
    
    prompt_tokens: int = 0
    """Number of prompt tokens."""
    
    completion_tokens: int = 0
    """Number of generated tokens."""
    
    finish_reason: Optional[str] = None
    """Why generation stopped: 'stop', 'length', 'eos'"""


class LLMEngine:
    """Main inference engine.
    
    Handles:
    - Model loading with tensor parallelism
    - KV cache management
    - Continuous batching (simplified)
    - Token generation
    
    Usage:
        engine = LLMEngine.from_pretrained(
            "/models/gpt-oss-120b",
            tensor_parallel_size=2,
        )
        
        outputs = engine.generate(["Hello world"], SamplingParams(max_tokens=100))
    """
    
    def __init__(
        self,
        model: GptOssForCausalLM,
        tokenizer,
        config: GptOssConfig,
        kv_cache_manager: KVCacheManager,
        tensor_parallel_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        self.tensor_parallel_size = tensor_parallel_size
        self.device = next(model.parameters()).device
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        **kwargs,
    ) -> "LLMEngine":
        """Load model and create engine.
        
        Args:
            model_path: Path to model directory
            tensor_parallel_size: Number of GPUs
            dtype: Model dtype
            gpu_memory_utilization: Fraction of GPU memory for KV cache
            max_model_len: Maximum sequence length
            
        Returns:
            Initialized LLMEngine
        """
        # Initialize distributed
        init_distributed(tensor_parallel_size=tensor_parallel_size)
        rank = get_tensor_parallel_rank()
        device = torch.device(f"cuda:{rank}")
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        print(f"[Rank {rank}] Loading model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load config
        config = cls._load_config(model_path)
        if max_model_len:
            config.max_position_embeddings = max_model_len
        
        # Create model
        model = GptOssForCausalLM(config)
        
        # Load weights - use simple loader for debugging
        cls._load_weights_simple(model, model_path, device, torch_dtype, rank, tensor_parallel_size)
        
        model = model.to(device).to(torch_dtype)
        model.eval()
        
        # Process MXFP4 weights after loading (dequantize once)
        if config.use_mxfp4:
            print(f"[Rank {rank}] Processing MXFP4 weights...")
            cls._process_mxfp4_weights(model)
            print(f"[Rank {rank}] MXFP4 weights processed")
        
        # Create KV cache manager
        kv_cache_manager = KVCacheManager(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads // tensor_parallel_size,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            dtype=torch_dtype,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        print(f"[Rank {rank}] Model loaded successfully")
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            kv_cache_manager=kv_cache_manager,
            tensor_parallel_size=tensor_parallel_size,
        )
    
    @staticmethod
    def _process_mxfp4_weights(model: GptOssForCausalLM):
        """Process MXFP4 weights after loading - dequantize once."""
        from mini_vllm.layers.fused_moe import FusedMoE
        
        for name, module in model.named_modules():
            if isinstance(module, FusedMoE):
                module.process_weights_after_loading()
    
    @staticmethod
    def _load_config(model_path: str) -> GptOssConfig:
        """Load model config from path."""
        import json
        import os
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        
        return GptOssConfig(**config_dict)
    
    @staticmethod
    def _load_weights(
        model: GptOssForCausalLM,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        tp_rank: int,
        tp_size: int,
    ):
        """Load model weights with MXFP4 and tensor parallel support.
        
        Uses streaming loading to minimize memory usage.
        """
        import os
        import glob
        from safetensors import safe_open
        
        # Find weight files
        safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        config = model.config
        use_mxfp4 = getattr(config, 'use_mxfp4', False)
        
        print(f"[Rank {tp_rank}] Loading weights, MXFP4={use_mxfp4}")
        
        # MXFP4 parameters
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
        
        # Patterns for column-parallel (shard dim 0)
        column_parallel = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head"]
        # Patterns for row-parallel (shard dim 1)
        row_parallel = ["o_proj", "down_proj"]
        
        # MXFP4 weight patterns (do NOT convert dtype)
        mxfp4_patterns = [".w13_weight", ".w2_weight", ".w13_weight_scale", ".w2_weight_scale"]
        
        # Build name mapping from checkpoint to model params
        params_dict = dict(model.named_parameters())
        
        loaded_count = 0
        file_count = len(safetensors_files)
        
        for file_idx, file_path in enumerate(safetensors_files):
            print(f"[Rank {tp_rank}] Loading file {file_idx+1}/{file_count}: {os.path.basename(file_path)}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    # Find corresponding parameter
                    param_name = LLMEngine._map_ckpt_to_param_name(name)
                    if param_name not in params_dict:
                        continue
                    
                    param = params_dict[param_name]
                    tensor = f.get_tensor(name)
                    
                    # Check if this is an MXFP4 weight
                    is_mxfp4 = use_mxfp4 and any(p in name for p in mxfp4_patterns)
                    
                    if is_mxfp4:
                        # Handle MXFP4 weights - shard but don't convert dtype
                        tensor = LLMEngine._shard_mxfp4_weight(
                            tensor=tensor,
                            name=name,
                            tp_rank=tp_rank,
                            tp_size=tp_size,
                            tp_rank_start=tp_rank_start,
                            tp_rank_end=tp_rank_end,
                            num_experts=num_experts,
                            intermediate_size=intermediate_size,
                            mxfp4_block=mxfp4_block,
                        )
                        # Direct copy into parameter
                        param.data.copy_(tensor)
                    else:
                        # Handle regular weights
                        if tp_size > 1:
                            tensor = LLMEngine._shard_weight(
                                tensor, name, tp_rank, tp_size, column_parallel, row_parallel
                            )
                        # Direct copy into parameter
                        param.data.copy_(tensor.to(dtype))
                    
                    loaded_count += 1
                    
                    # Free memory immediately
                    del tensor
        
        print(f"[Rank {tp_rank}] Loaded {loaded_count} weight tensors")
    
    @staticmethod
    def _map_ckpt_to_param_name(name: str) -> str:
        """Map checkpoint tensor name to model parameter name."""
        # Most names map directly, but some need adjustment
        # self_attn -> attn
        name = name.replace("self_attn", "attn")
        return name
    
    @staticmethod
    def _load_weights_simple(
        model: GptOssForCausalLM,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        tp_rank: int,
        tp_size: int,
    ):
        """Load weights with proper sharding and name mapping.
        
        Handles:
        - Tensor parallel sharding for column/row parallel layers
        - QKV weight stacking
        - MXFP4 quantized weights
        - MoE expert weights
        """
        import os
        import glob
        from safetensors import safe_open
        
        safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        config = model.config
        use_mxfp4 = getattr(config, 'use_mxfp4', False)
        
        print(f"[Rank {tp_rank}] Loading weights with full sharding, MXFP4={use_mxfp4}, TP={tp_size}")
        
        params_dict = dict(model.named_parameters())
        loaded = 0
        loaded_names = set()
        
        # MXFP4 parameters for sharding
        mxfp4_block = 32
        num_experts = config.num_local_experts
        intermediate_size = config.intermediate_size
        
        # Calculate TP slicing for MXFP4 
        per_rank_intermediate = intermediate_size // tp_size
        tp_rank_start = tp_rank * per_rank_intermediate
        tp_rank_end = (tp_rank + 1) * per_rank_intermediate
        
        # Patterns for TP sharding
        column_parallel = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "lm_head", "embed_tokens"]
        row_parallel = ["o_proj", "down_proj"]
        head_parallel = ["sinks"]  # Attention sinks are sharded by head
        
        # QKV buffer to stack q/k/v weights
        qkv_buffers = {}  # layer_idx -> {q, k, v}
        
        for file_path in safetensors_files:
            fname = os.path.basename(file_path)
            print(f"[Rank {tp_rank}] Processing {fname}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for ckpt_name in f.keys():
                    tensor = f.get_tensor(ckpt_name)
                    
                    # Handle QKV projection stacking
                    if any(x in ckpt_name for x in ["q_proj", "k_proj", "v_proj"]):
                        loaded += LLMEngine._handle_qkv_weight(
                            ckpt_name, tensor, params_dict, qkv_buffers, 
                            tp_rank, tp_size, dtype, loaded_names
                        )
                        del tensor
                        continue
                    
                    # Handle MoE expert weights
                    if ".mlp.experts." in ckpt_name:
                        loaded += LLMEngine._handle_moe_weight(
                            ckpt_name, tensor, params_dict, 
                            tp_rank, tp_size, dtype, loaded_names,
                            num_experts, intermediate_size, mxfp4_block, use_mxfp4
                        )
                        del tensor
                        continue
                    
                    # Map checkpoint name to param name
                    param_name = LLMEngine._map_ckpt_to_param_name(ckpt_name)
                    if param_name not in params_dict:
                        continue
                    
                    param = params_dict[param_name]
                    
                    # Apply TP sharding
                    if tp_size > 1:
                        tensor = LLMEngine._apply_tp_sharding(
                            tensor, ckpt_name, tp_rank, tp_size,
                            column_parallel, row_parallel, head_parallel
                        )
                    
                    # Copy to parameter
                    if tensor.shape == param.shape:
                        param.data.copy_(tensor.to(dtype))
                        loaded += 1
                        loaded_names.add(param_name)
                    
                    del tensor
        
        print(f"[Rank {tp_rank}] Loaded {loaded} parameters")
        
        # Report missing
        missing = set(params_dict.keys()) - loaded_names
        if missing:
            print(f"[Rank {tp_rank}] Missing {len(missing)} params")
    
    @staticmethod
    def _handle_qkv_weight(
        ckpt_name: str,
        tensor: torch.Tensor,
        params_dict: dict,
        qkv_buffers: dict,
        tp_rank: int,
        tp_size: int,
        dtype: torch.dtype,
        loaded_names: set,
    ) -> int:
        """Handle Q/K/V weight stacking into qkv_proj."""
        import re
        
        # Extract layer index
        match = re.search(r'layers\.(\d+)', ckpt_name)
        if not match:
            return 0
        layer_idx = int(match.group(1))
        
        # Determine which weight (q, k, or v)
        if "q_proj" in ckpt_name:
            weight_type = "q"
        elif "k_proj" in ckpt_name:
            weight_type = "k"
        elif "v_proj" in ckpt_name:
            weight_type = "v"
        else:
            return 0
        
        # Shard along output dimension (column parallel)
        if tp_size > 1:
            shard_size = tensor.shape[0] // tp_size
            tensor = tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
        
        # Store in buffer
        if layer_idx not in qkv_buffers:
            qkv_buffers[layer_idx] = {}
        qkv_buffers[layer_idx][weight_type] = tensor
        
        # Check if we have all three
        if len(qkv_buffers[layer_idx]) == 3:
            q = qkv_buffers[layer_idx]["q"]
            k = qkv_buffers[layer_idx]["k"]
            v = qkv_buffers[layer_idx]["v"]
            
            # Stack QKV (concat along dim 0)
            qkv = torch.cat([q, k, v], dim=0)
            
            # Find target parameter
            param_name = f"model.layers.{layer_idx}.attn.qkv_proj.weight"
            if param_name in params_dict:
                param = params_dict[param_name]
                if qkv.shape == param.shape:
                    param.data.copy_(qkv.to(dtype))
                    loaded_names.add(param_name)
                    del qkv_buffers[layer_idx]
                    return 1
        
        return 0
    
    @staticmethod
    def _handle_moe_weight(
        ckpt_name: str,
        tensor: torch.Tensor,
        params_dict: dict,
        tp_rank: int,
        tp_size: int,
        dtype: torch.dtype,
        loaded_names: set,
        num_experts: int,
        intermediate_size: int,
        mxfp4_block: int,
        use_mxfp4: bool,
    ) -> int:
        """Handle MoE expert weight loading with sharding."""
        # Map checkpoint name to param name
        param_name = ckpt_name.replace("self_attn", "attn")
        
        if param_name not in params_dict:
            return 0
        
        param = params_dict[param_name]
        
        # Apply sharding for MXFP4 weights
        if use_mxfp4 and ("w13_weight" in ckpt_name or "w2_weight" in ckpt_name):
            per_rank = intermediate_size // tp_size
            tp_start = tp_rank * per_rank
            tp_end = (tp_rank + 1) * per_rank
            
            if ".w13_weight_scale" in ckpt_name:
                tensor = tensor[:, 2 * tp_start : 2 * tp_end, ...]
            elif ".w2_weight_scale" in ckpt_name:
                tp_start_block = tp_start // mxfp4_block
                tp_end_block = tp_end // mxfp4_block
                tensor = tensor[..., tp_start_block : tp_end_block]
            elif ".w13_weight" in ckpt_name:
                tensor = tensor.view(num_experts, 2 * intermediate_size, -1)
                tensor = tensor[:, 2 * tp_start : 2 * tp_end, ...]
            elif ".w2_weight" in ckpt_name:
                tensor = tensor.view(num_experts, -1, intermediate_size // 2)
                tensor = tensor[..., tp_start // 2 : tp_end // 2]
            
            tensor = tensor.contiguous()
            
            # Don't convert dtype for quantized weights
            if tensor.shape == param.shape:
                param.data.copy_(tensor)
                loaded_names.add(param_name)
                return 1
        else:
            # Non-MXFP4 or bias
            if ".w13_bias" in ckpt_name:
                per_rank = intermediate_size // tp_size
                tensor = tensor[:, 2 * tp_rank * per_rank : 2 * (tp_rank + 1) * per_rank]
            
            if tensor.shape == param.shape:
                param.data.copy_(tensor.to(dtype))
                loaded_names.add(param_name)
                return 1
        
        return 0
    
    @staticmethod
    def _apply_tp_sharding(
        tensor: torch.Tensor,
        name: str,
        tp_rank: int,
        tp_size: int,
        column_parallel: list,
        row_parallel: list,
        head_parallel: list,
    ) -> torch.Tensor:
        """Apply tensor parallel sharding."""
        for pattern in column_parallel:
            if pattern in name:
                shard_size = tensor.shape[0] // tp_size
                return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
        
        for pattern in row_parallel:
            if pattern in name:
                shard_size = tensor.shape[-1] // tp_size
                return tensor[..., tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
        
        for pattern in head_parallel:
            if pattern in name:
                shard_size = tensor.shape[0] // tp_size
                return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
        
        return tensor
    
    @staticmethod
    def _shard_mxfp4_weight(
        tensor: torch.Tensor,
        name: str,
        tp_rank: int,
        tp_size: int,
        tp_rank_start: int,
        tp_rank_end: int,
        num_experts: int,
        intermediate_size: int,
        mxfp4_block: int,
    ) -> torch.Tensor:
        """Shard MXFP4 weight for tensor parallelism."""
        if tp_size == 1:
            return tensor
        
        if ".w13_weight_scale" in name:
            # [E, 2*N, K/block] -> shard on dim 1
            narrow = tensor[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
            return narrow.contiguous()
            
        elif ".w2_weight_scale" in name:
            # [E, K, N/block] -> shard on dim 2
            tp_start_block = tp_rank_start // mxfp4_block
            tp_end_block = tp_rank_end // mxfp4_block
            narrow = tensor[..., tp_start_block : tp_end_block]
            return narrow.contiguous()
            
        elif ".w13_weight" in name:
            # [E, 2*N, block_size, entries] -> flatten and shard
            tensor = tensor.view(num_experts, 2 * intermediate_size, -1).contiguous()
            narrow = tensor[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
            return narrow.contiguous()
            
        elif ".w2_weight" in name:
            # [E, K, N/2] -> shard on last dim
            tensor = tensor.view(num_experts, -1, intermediate_size // 2).contiguous()
            narrow = tensor[..., tp_rank_start // 2 : tp_rank_end // 2]
            return narrow.contiguous()
        
        return tensor
    
    @staticmethod
    def _shard_weight(
        tensor: torch.Tensor,
        name: str,
        tp_rank: int,
        tp_size: int,
        column_parallel: List[str],
        row_parallel: List[str],
    ) -> torch.Tensor:
        """Shard a weight tensor for tensor parallelism."""
        for pattern in column_parallel:
            if pattern in name and tensor.dim() >= 2:
                # Shard along output dimension
                shard_size = tensor.size(0) // tp_size
                return tensor[tp_rank * shard_size:(tp_rank + 1) * shard_size].contiguous()
        
        for pattern in row_parallel:
            if pattern in name and tensor.dim() >= 2:
                # Shard along input dimension
                shard_size = tensor.size(1) // tp_size
                return tensor[:, tp_rank * shard_size:(tp_rank + 1) * shard_size].contiguous()
        
        return tensor
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[GenerationOutput]:
        """Generate completions for prompts.
        
        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters
            
        Returns:
            List of GenerationOutput, one per prompt
        """
        outputs = []
        
        for prompt in prompts:
            output = self._generate_single(prompt, sampling_params)
            outputs.append(output)
        
        return outputs
    
    def _generate_single(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> GenerationOutput:
        """Generate completion for a single prompt."""
        request_id = str(uuid.uuid4())[:8]
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        
        # Generate
        generated_ids = []
        
        with torch.no_grad():
            for step in range(params.max_tokens):
                # Get positions
                positions = torch.arange(
                    input_ids.shape[1], device=self.device
                ).unsqueeze(0)
                
                # Forward pass
                hidden_states = self.model(input_ids, positions)
                
                # Get logits for last token
                logits = self.model.compute_logits(hidden_states[:, -1:, :])
                
                # Sample next token
                next_token = self.model.sample(
                    logits.squeeze(1),
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                )
                
                generated_ids.append(next_token.item())
                
                # Check stop conditions
                if next_token.item() == self.tokenizer.eos_token_id:
                    finish_reason = "eos"
                    break
                
                if params.stop_token_ids and next_token.item() in params.stop_token_ids:
                    finish_reason = "stop"
                    break
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            else:
                finish_reason = "length"
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check string stop conditions
        if params.stop:
            for stop_str in params.stop:
                if stop_str in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_str)]
                    finish_reason = "stop"
                    break
        
        return GenerationOutput(
            request_id=request_id,
            prompt=prompt,
            text=generated_text,
            token_ids=generated_ids,
            finished=True,
            prompt_tokens=prompt_len,
            completion_tokens=len(generated_ids),
            finish_reason=finish_reason,
        )
    
    def generate_stream(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> Iterator[GenerationOutput]:
        """Stream generation token by token.
        
        Yields GenerationOutput for each token.
        """
        request_id = str(uuid.uuid4())[:8]
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        
        generated_ids = []
        generated_text = ""
        
        with torch.no_grad():
            for step in range(params.max_tokens):
                positions = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
                
                hidden_states = self.model(input_ids, positions)
                logits = self.model.compute_logits(hidden_states[:, -1:, :])
                
                next_token = self.model.sample(
                    logits.squeeze(1),
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                )
                
                generated_ids.append(next_token.item())
                
                # Decode just the new token
                new_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                generated_text += new_text
                
                # Check stop
                finished = False
                finish_reason = None
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    finished = True
                    finish_reason = "eos"
                elif params.stop_token_ids and next_token.item() in params.stop_token_ids:
                    finished = True
                    finish_reason = "stop"
                elif params.stop:
                    for stop_str in params.stop:
                        if stop_str in generated_text:
                            finished = True
                            finish_reason = "stop"
                            break
                
                yield GenerationOutput(
                    request_id=request_id,
                    prompt=prompt,
                    text=new_text,  # Just the new token
                    token_ids=[next_token.item()],
                    finished=finished,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(generated_ids),
                    finish_reason=finish_reason,
                )
                
                if finished:
                    break
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)


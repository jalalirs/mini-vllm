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
        
        # Load weights
        cls._load_weights(model, model_path, device, torch_dtype, rank, tensor_parallel_size)
        
        model = model.to(device).to(torch_dtype)
        model.eval()
        
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
        """Load model weights with tensor parallel sharding."""
        import os
        import glob
        from safetensors import safe_open
        
        # Find weight files
        safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        # Patterns for column-parallel (shard dim 0)
        column_parallel = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head"]
        # Patterns for row-parallel (shard dim 1)
        row_parallel = ["o_proj", "down_proj"]
        
        state_dict = {}
        
        for file_path in safetensors_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    
                    # Apply sharding if needed
                    if tp_size > 1:
                        tensor = LLMEngine._shard_weight(
                            tensor, name, tp_rank, tp_size, column_parallel, row_parallel
                        )
                    
                    # Convert dtype and store
                    state_dict[name] = tensor.to(dtype)
        
        # Load into model
        model.load_state_dict(state_dict, strict=False)
    
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


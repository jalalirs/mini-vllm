# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS model implementation.

GPT-OSS-120B architecture:
- Decoder-only transformer
- Mixture of Experts (MoE) with top-2 routing
- Grouped Query Attention (GQA)
- YARN RoPE for extended context
- Alternating sliding window attention
- Attention sinks
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.layers.linear import QKVParallelLinear, RowParallelLinear
from mini_vllm.layers.layernorm import RMSNorm
from mini_vllm.layers.rotary_embedding import get_rope
from mini_vllm.layers.vocab_embedding import VocabParallelEmbedding, ParallelLMHead
from mini_vllm.layers.fused_moe import FusedMoE
from mini_vllm.attention import Attention
from mini_vllm.distributed import (
    get_tensor_parallel_rank,
    get_tensor_parallel_world_size,
)


class GptOssConfig:
    """Configuration for GPT-OSS model."""
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 8192,
        intermediate_size: int = 28672,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        num_local_experts: int = 16,
        num_experts_per_tok: int = 2,
        max_position_embeddings: int = 131072,
        rope_theta: float = 500000.0,
        rope_scaling: Optional[dict] = None,
        sliding_window: Optional[int] = 4096,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.tie_word_embeddings = tie_word_embeddings
        
        # YARN RoPE parameters
        self.rope_parameters = rope_scaling or {
            "rope_theta": rope_theta,
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "truncate": True,
        }


class OAIAttention(nn.Module):
    """GPT-OSS attention with GQA and sliding window."""
    
    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        
        tp_size = get_tensor_parallel_world_size()
        tp_rank = get_tensor_parallel_rank()
        
        # Rotary embedding with YARN scaling
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_parameters["rope_theta"],
            rope_scaling=config.rope_parameters,
        )
        
        # Attention head counts per partition
        self.num_local_heads = config.num_attention_heads // tp_size
        self.num_local_kv_heads = config.num_key_value_heads // tp_size
        
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # Attention sinks (learnable per-head parameters)
        self.sinks = nn.Parameter(
            torch.zeros(self.num_local_heads), requires_grad=False
        )
        
        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
        )
        
        # Sliding window (only on even layers)
        sliding_window = config.sliding_window if layer_idx % 2 == 0 else None
        
        # Attention computation
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            sliding_window=sliding_window,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention
        attn_output = self.attn(q, k, v, kv_cache)
        
        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


class MLPBlock(nn.Module):
    """GPT-OSS MoE block with FusedMoE."""
    
    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # Router
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        
        # Fused MoE experts
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="swiglu",
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute router logits
        router_logits = self.router(x)
        
        # FusedMoE forward
        return self.experts(hidden_states=x, router_logits=router_logits)


class TransformerBlock(nn.Module):
    """Single transformer block for GPT-OSS."""
    
    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Attention
        self.attn = OAIAttention(config, layer_idx)
        
        # MoE MLP
        self.mlp = MLPBlock(config, layer_idx)
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm for attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self attention
        hidden_states = self.attn(hidden_states, positions, kv_cache)
        
        # Pre-norm for MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MoE MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class GptOssModel(nn.Module):
    """GPT-OSS transformer model (without LM head)."""
    
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.embedding(input_ids)
        residual = None
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            hidden_states, residual = layer(hidden_states, positions, residual, kv_cache)
        
        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class GptOssForCausalLM(nn.Module):
    """GPT-OSS model for causal language modeling."""
    
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        
        # Transformer
        self.model = GptOssModel(config)
        
        # LM head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches)
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        logits = self.lm_head(hidden_states)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if temperature == 0:
            # Greedy
            return logits.argmax(dim=-1)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.squeeze(-1)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from pretrained weights."""
        from mini_vllm.loader import load_model_weights, load_config
        
        # Load config
        config = load_config(model_path)
        
        # Create model
        model = cls(config)
        
        # Load weights
        load_model_weights(model, model_path, **kwargs)
        
        return model


// Mini-vLLM CUDA ops binding for PyTorch
// Registers custom CUDA kernels for H100-optimized inference

#include <torch/extension.h>
#include "ops.h"
#include "cache.h"

// ============================================================================
// TORCH_LIBRARY: Register operations with PyTorch dispatcher
// ============================================================================

TORCH_LIBRARY(mini_vllm, m) {
  // =========================================================================
  // Attention Operations
  // =========================================================================
  
  // PagedAttention V1 - Single-pass attention with paged KV cache
  m.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  m.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2 - Two-pass attention for better memory efficiency
  m.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  m.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Merge attention states from split-KV computation
  m.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse) -> ()");
  m.impl("merge_attn_states", torch::kCUDA, &merge_attn_states);

  // =========================================================================
  // LayerNorm Operations
  // =========================================================================
  
  // RMS Normalization
  m.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  m.impl("rms_norm", torch::kCUDA, &rms_norm);

  // Fused Add + RMS Normalization (residual connection + norm)
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // =========================================================================
  // Activation Operations (for SwiGLU in MoE)
  // =========================================================================
  
  // SiLU and Multiply (for SwiGLU: silu(gate) * up)
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Multiply then SiLU
  m.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  m.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);

  // GELU and Multiply
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // GELU Tanh and Multiply
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // =========================================================================
  // Positional Encoding Operations
  // =========================================================================
  
  // Rotary Position Embedding (RoPE)
  m.def("rotary_embedding(Tensor positions, Tensor! query, Tensor? key, "
        "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // =========================================================================
  // KV Cache Operations
  // =========================================================================
  
  // Swap KV cache blocks between GPU memory locations
  m.def("swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  m.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Reshape and cache KV
  m.def("reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
        "Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape and cache with flash attention layout
  m.def("reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, "
        "Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache_flash", torch::kCUDA, &reshape_and_cache_flash);
}

// ============================================================================
// PYBIND11_MODULE: Python module bindings  
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Mini-vLLM CUDA operations for H100-optimized LLM inference";
}

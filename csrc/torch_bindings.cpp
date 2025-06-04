#include <torch/library.h>
#include "registration.h"
#include "src/advance_step_flashattn.h"
#include "src/advance_step_xformers.h"
#include "src/awq_dequantize.h"
#include "src/awq_gemm_gcu.h"
#include "src/batched_rotary_embedding.h"
#include "src/cache_ops.h"
#include "src/concat_and_cache_mla.h"
#include "src/context_attention_forward.h"
#include "src/cutlass_scaled_mm.h"
#include "src/dispatch_bgmv.h"
#include "src/dispatch_bgmv_low_level.h"
#include "src/dot_bias_quant.h"
#include "src/dynamic_per_token_group_fp8_quant.h"
#include "src/dynamic_per_token_scaled_fp8_quant.h"
#include "src/dynamic_scaled_fp8_quant.h"
#include "src/dynamic_scaled_int8_quant.h"
#include "src/dynamic_split.h"
#include "src/ets_moe_align_block_size.h"
#include "src/fatrelu_and_mul.h"
#include "src/fused_add_rms_norm.h"
#include "src/fused_add_rms_norm_per_token_group_quant_fp8.h"
#include "src/fused_add_rms_norm_static_fp8_quant.h"
#include "src/fused_add_rms_norm_static_int8_quant.h"
#include "src/fused_dispatch_decode.h"
#include "src/fused_grouped_topk.h"
#include "src/fused_moe_kernel.h"
#include "src/fused_moe_quant_kernel.h"
#include "src/fused_qkv_gemm_quant.h"
#include "src/fused_qkv_proj.h"
#include "src/gelu_and_mul.h"
#include "src/gelu_asym_quant.h"
#include "src/gelu_fast.h"
#include "src/gelu_fast_asym_quant.h"
#include "src/gelu_fast_static_int8_quant.h"
#include "src/gelu_mul_quant.h"
#include "src/gelu_new.h"
#include "src/gelu_new_asym_quant.h"
#include "src/gelu_new_static_int8_quant.h"
#include "src/gelu_static_int8_quant.h"
#include "src/gelu_quick.h"
#include "src/gelu_tanh_and_mul.h"
#include "src/gelu_tanh_asym_quant.h"
#include "src/gelu_tanh_mul_quant.h"
#include "src/gelu_tanh_static_int8_quant.h"
#include "src/get_ep_indices.h"
#include "src/gptq_gemm_gcu.h"
#include "src/gptq_shuffle.h"
#include "src/layer_norm_static_int8_quant.h"
#include "src/linear_quant.h"
#include "src/memory_efficient_attention_alibi.h"
#include "src/merge_attn_states.h"
#include "src/gather_cache.h"
#include "src/moe_align_block_size.h"
#include "src/moe_align_block_size_pad.h"
#include "src/moe_sum.h"
#include "src/mul_and_silu.h"
#include "src/paged_attention_v1.h"
#include "src/paged_attention_v2.h"
#include "src/reshape_and_cache_flash.h"
#include "src/rms_norm.h"
#include "src/rms_norm_per_token_group_quant_fp8.h"
#include "src/rms_norm_static_fp8_quant.h"
#include "src/rms_norm_static_int8_quant.h"
#include "src/rotary_embedding.h"
#include "src/rotary_embedding_with_kv_cache.h"
#include "src/sgl_moe_align_block_size.h"
#include "src/silu_and_mul.h"
#include "src/silu_and_mul_pad.h"
#include "src/silu_asym_quant.h"
#include "src/silu_mul_per_token_group_quant.h"
#include "src/silu_mul_per_token_group_quant_with_size.h"
#include "src/silu_mul_static_int8_quant.h"
#include "src/silu_static_int8_quant.h"
#include "src/static_scaled_fp8_quant.h"
#include "src/static_scaled_int8_asym_dequant.h"
#include "src/static_scaled_int8_asym_quant.h"
#include "src/static_scaled_int8_dequant.h"
#include "src/static_scaled_int8_quant.h"
#include "src/topk_softmax.h"
#include "src/weak_ref_tensor.h"
#include "src/weight_only_quant.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

using namespace vllm_gcu::llm_ops;
TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  std::optional<c10::OperatorHandle> handle;

  handle = c10::Dispatcher::singleton().findSchema({"_C::weak_ref_tensor", ""});
  if (!handle.has_value()) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  }
  ops.impl("weak_ref_tensor", torch::kPrivateUse1, &weak_ref_tensor);

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  // TODO modify to tensor
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float k_scale, float v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step, float k_zero,"
      "    float v_zero, Tensor? out_scales) -> ()");
  ops.impl("paged_attention_v1", torch::kPrivateUse1, &paged_attention_v1);

  // PagedAttention V2.
  // TODO modify to tensor
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float k_scale, float v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step, float k_zero,"
      "    float v_zero, Tensor? out_scales) -> ()");
  ops.impl("paged_attention_v2", torch::kPrivateUse1, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kPrivateUse1, &silu_and_mul);

  ops.def("silu_and_mul_pad(Tensor(a!) out, Tensor input, Tensor size) -> ()");
  ops.impl("silu_and_mul_pad", torch::kPrivateUse1, &silu_and_mul_pad);

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kPrivateUse1, &mul_and_silu);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kPrivateUse1, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kPrivateUse1, &gelu_tanh_and_mul);

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kPrivateUse1, &fatrelu_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kPrivateUse1, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kPrivateUse1, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kPrivateUse1, &gelu_quick);

  // prepare_inputs advance_step
  ops.def(
      "advance_step_xformers(int num_seqs, int num_queries, int block_size, "
      "Tensor! input_tokens, Tensor sampled_token_ids, "
      "Tensor! input_positions, Tensor! seq_lens, Tensor! slot_mapping, "
      "Tensor block_tables) -> ()");
  ops.impl("advance_step_xformers", torch::kPrivateUse1,
           &advance_step_xformers);

  ops.def(
      "advance_step_flashattn(int num_seqs, int num_queries, int block_size, "
      "Tensor! input_tokens, Tensor sampled_token_ids, "
      "Tensor! input_positions, Tensor! seq_lens, Tensor! slot_mapping, "
      "Tensor block_tables) -> ()");
  ops.impl("advance_step_flashattn", torch::kPrivateUse1,
           &advance_step_flashattn);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kPrivateUse1, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kPrivateUse1, &fused_add_rms_norm);

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");
  ops.impl("rms_norm_static_fp8_quant", torch::kPrivateUse1,
           &rms_norm_static_fp8_quant);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kPrivateUse1,
           &fused_add_rms_norm_static_fp8_quant);

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kPrivateUse1, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def(
      "batched_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor! key, int head_size,"
      "                         Tensor cos_sin_cache, bool is_neox,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding", torch::kPrivateUse1,
           &batched_rotary_embedding);

  // Quantization ops

  // Quantized GEMM for AQLM.
  ops.def(
      "aqlm_gemm(Tensor input, Tensor codes, Tensor codebooks, "
      "Tensor scales, int[] codebook_partition_sizes, Tensor? bias) "
      "-> Tensor");
  // ops.impl("aqlm_gemm", torch::kPrivateUse1, &aqlm_gemm);

  // Decompression method for AQLM.
  ops.def(
      "aqlm_dequant(Tensor codes, Tensor codebooks, "
      "int[] codebook_partition_sizes) -> Tensor");
  // ops.impl("aqlm_dequant", torch::kPrivateUse1, &aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters) -> Tensor");
  // ops.impl("awq_gemm", c10::kPrivateUse1, &awq_gemm);

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor");
  // ops.impl("awq_dequantize", c10::kPrivateUse1, &awq_dequantize);

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, SymInt size_k) -> "
      "Tensor");
  // conditionally compiled so impl in source file

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k) -> Tensor");
  //  conditionally compiled so impl in source file

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor");
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // conditionally compiled so impl registration is in source file

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  // ops.impl("permute_cols", torch::kPrivateUse1, &permute_cols);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool has_zp, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // Dequantization for GGML.
  ops.def("ggml_dequantize(Tensor W, int type, SymInt m, SymInt n) -> Tensor");
  // ops.impl("ggml_dequantize", torch::kPrivateUse1, &ggml_dequantize);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  // ops.impl("ggml_mul_mat_vec_a8", torch::kPrivateUse1, &ggml_mul_mat_vec_a8);

  // mmq kernel for GGML.
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  // ops.impl("ggml_mul_mat_a8", torch::kPrivateUse1, &ggml_mul_mat_a8);

  // fp8_marlin Optimized Quantized GEMM for FP8 weight-only.
  ops.def(
      "fp8_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, int num_bits, SymInt size_m, SymInt size_n, "
      "SymInt size_k) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // marlin_qqq_gemm for QQQ.
  ops.def(
      "marlin_qqq_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor s_tok, Tensor s_ch, Tensor s_group, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, "
      "SymInt size_k) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  //   ops.def(
  //       "cutlass_scaled_mm(Tensor! out, Tensor a,"
  //       "                  Tensor b, Tensor a_scales,"
  //       "                  Tensor b_scales, Tensor? bias) -> ()");
  // ops.impl("cutlass_scaled_mm", torch::kPrivateUse1, &cutlass_scaled_mm);

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");
  // ops.impl("cutlass_scaled_mm_azp", torch::kPrivateUse1,
  // &cutlass_scaled_mm_azp);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  // ops.impl("cutlass_scaled_mm_supports_fp8",
  // &cutlass_scaled_mm_supports_fp8);

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");
  // ops.impl("cutlass_scaled_mm_supports_block_fp8",
  // &cutlass_scaled_mm_supports_fp8);

  // Check if cutlass sparse scaled_mm is supported for CUDA devices of the
  // given capability
  ops.def(
      "cutlass_sparse_scaled_mm_supported(int cuda_device_capability) -> bool");
  // ops.impl("cutlass_sparse_scaled_mm_supported",
  // &cutlass_sparse_scaled_mm_supported);

  // CUTLASS sparse GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_sparse_mm(Tensor! out, Tensor a,"
      "                         Tensor bt_nzs,"
      "                         Tensor bt_meta, Tensor a_scales,"
      "                         Tensor b_scales, Tensor? bias) -> ()");
  // ops.impl("cutlass_scaled_sparse_mm", torch::kPrivateUse1,
  // &cutlass_scaled_sparse_mm);

  // CUTLASS sparse matrix compressor
  ops.def(
      "cutlass_sparse_compress_entry(Tensor! a_nzs, Tensor! a_meta,"
      "                              Tensor a) -> bool");
  // ops.impl("cutlass_sparse_compress_entry", &cutlass_sparse_compress_entry);

  // Mamba selective scan kernel
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int pad_slot_id) -> ()");
  // ops.impl("selective_scan_fwd", torch::kPrivateUse1, &selective_scan_fwd);

  ops.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
  // ops.impl("causal_conv1d_update", torch::kPrivateUse1,
  // &causal_conv1d_update);

  ops.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
  // ops.impl("causal_conv1d_fwd", torch::kPrivateUse1, &causal_conv1d_fwd);

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, int bit) "
      "-> Tensor");
  // ops.impl("gptq_gemm", torch::kPrivateUse1, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  // ops.impl("gptq_shuffle", torch::kPrivateUse1, &gptq_shuffle);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_fp8_quant", torch::kPrivateUse1,
           &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> ()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kPrivateUse1,
           &dynamic_scaled_fp8_quant);

  // Compute dynamic scaled INT8 quantized tensor with azp
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! output, "
      "Tensor input, Tensor! scales, "
      "Tensor azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kPrivateUse1,
           &dynamic_scaled_int8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kPrivateUse1,
           &dynamic_per_token_scaled_fp8_quant);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kPrivateUse1,
           &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  //   ops.def(
  //       "dynamic_scaled_int8_quant(Tensor! result, Tensor input, "
  //   "Tensor! scale, "
  //       "Tensor!? azp) -> ()");
  // ops.impl("dynamic_scaled_int8_quant", torch::kPrivateUse1,
  // &dynamic_scaled_int8_quant);

  // gcu defined ops
  // TODO(?): function define
  ops.def(
      "awq_gemm_gcu(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, Tensor? bias, int group_size) -> "
      "Tensor");
  ops.impl("awq_gemm_gcu", torch::kPrivateUse1, &awq_gemm_gcu);

  ops.def(
      "gptq_gemm_gcu(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, int bit, "
      "Tensor? bias, int group_size) "
      "-> Tensor");
  ops.impl("gptq_gemm_gcu", c10::kPrivateUse1, &gptq_gemm_gcu);

  ops.def("fused_moe_kernel", &fused_moe_kernel);
  ops.impl("fused_moe_kernel", c10::kPrivateUse1, &fused_moe_kernel);

  ops.def("get_ep_indices", &get_ep_indices);
  ops.impl("get_ep_indices", c10::kPrivateUse1, &get_ep_indices);

  ops.def("moe_align_block_size_pad", &moe_align_block_size_pad);
  ops.impl("moe_align_block_size_pad", c10::kPrivateUse1,
           &moe_align_block_size_pad);

  ops.def("fused_moe_quant_kernel", &fused_moe_quant_kernel);
  ops.impl("fused_moe_quant_kernel", c10::kPrivateUse1,
           &fused_moe_quant_kernel);

  ops.def("context_attention_forward", &context_attention_forward);
  ops.impl("context_attention_forward", c10::kPrivateUse1,
           &context_attention_forward);

  ops.def("weight_only_quant", &weight_only_quant);
  ops.impl("weight_only_quant", c10::kPrivateUse1, &weight_only_quant);

  ops.def(
      "rms_norm_static_int8_quant(Tensor(a!) result, Tensor input, Tensor "
      "weight, Tensor "
      "scale, float epsilon) -> ()");
  ops.impl("rms_norm_static_int8_quant", c10::kPrivateUse1,
           &rms_norm_static_int8_quant);

  ops.def(
      "fused_add_rms_norm_static_int8_quant(Tensor(a!) result, Tensor input, "
      "Tensor(a!) "
      "residual, Tensor weight, float epsilon, Tensor scale) -> ()");
  ops.impl("fused_add_rms_norm_static_int8_quant",
           c10::kPrivateUse1,
           &fused_add_rms_norm_static_int8_quant);

  ops.def("gelu_tanh_static_int8_quant", &gelu_tanh_static_int8_quant);
  ops.impl("gelu_tanh_static_int8_quant", c10::kPrivateUse1,
            &gelu_tanh_static_int8_quant);

  ops.def("gelu_static_int8_quant", &gelu_static_int8_quant);
  ops.impl("gelu_static_int8_quant",
            c10::kPrivateUse1,
            &gelu_static_int8_quant);

  ops.def("gelu_new_static_int8_quant", &gelu_new_static_int8_quant);
  ops.impl("gelu_new_static_int8_quant",
            c10::kPrivateUse1,
            &gelu_new_static_int8_quant);

  ops.def("silu_static_int8_quant", &silu_static_int8_quant);
  ops.impl("silu_static_int8_quant", c10::kPrivateUse1,
            &silu_static_int8_quant);

  ops.def("gelu_fast_static_int8_quant", &gelu_fast_static_int8_quant);
  ops.impl("gelu_fast_static_int8_quant", c10::kPrivateUse1,
            &gelu_fast_static_int8_quant);

  ops.def("gelu_tanh_asym_quant", &gelu_tanh_asym_quant);
  ops.impl("gelu_tanh_asym_quant", c10::kPrivateUse1, &gelu_tanh_asym_quant);

  ops.def("gelu_asym_quant", &gelu_asym_quant);
  ops.impl("gelu_asym_quant", c10::kPrivateUse1, &gelu_asym_quant);

  ops.def("gelu_new_asym_quant", &gelu_new_asym_quant);
  ops.impl("gelu_new_asym_quant", c10::kPrivateUse1, &gelu_new_asym_quant);

  ops.def("silu_asym_quant", &silu_asym_quant);
  ops.impl("silu_asym_quant", c10::kPrivateUse1, &silu_asym_quant);

  ops.def("gelu_fast_asym_quant", &gelu_fast_asym_quant);
  ops.impl("gelu_fast_asym_quant", c10::kPrivateUse1, &gelu_fast_asym_quant);

  ops.def("static_scaled_int8_dequant", &static_scaled_int8_dequant);
  ops.impl("static_scaled_int8_dequant", c10::kPrivateUse1,
           &static_scaled_int8_dequant);

  ops.def("static_scaled_int8_asym_quant", &static_scaled_int8_asym_quant);
  ops.impl("static_scaled_int8_asym_quant", c10::kPrivateUse1,
           &static_scaled_int8_asym_quant);

  ops.def("static_scaled_int8_asym_dequant", &static_scaled_int8_asym_dequant);
  ops.impl("static_scaled_int8_asym_dequant", c10::kPrivateUse1,
           &static_scaled_int8_asym_dequant);

  ops.def("silu_mul_static_int8_quant(Tensor(a!) result, "
            "Tensor input, Tensor scale) -> ()");
  ops.impl("silu_mul_static_int8_quant", c10::kPrivateUse1,
            &silu_mul_static_int8_quant);

  ops.def("gelu_mul_quant", &gelu_mul_quant);
  ops.impl("gelu_mul_quant", c10::kPrivateUse1, &gelu_mul_quant);

  ops.def("gelu_tanh_mul_quant", &gelu_tanh_mul_quant);
  ops.impl("gelu_tanh_mul_quant", c10::kPrivateUse1, &gelu_tanh_mul_quant);

  ops.def("layer_norm_static_int8_quant", &layer_norm_static_int8_quant);
  ops.impl("layer_norm_static_int8_quant", c10::kPrivateUse1,
            &layer_norm_static_int8_quant);

  ops.def("dot_bias_quant", &dot_bias_quant);
  ops.impl("dot_bias_quant", c10::kPrivateUse1, &dot_bias_quant);

  ops.def(
      "cutlass_scaled_mm(Tensor! out, "
      "Tensor x, Tensor weight, Tensor x_scale, "
      "Tensor w_scale, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kPrivateUse1, &cutlass_scaled_mm);

  ops.def(
      "linear_quant(Tensor! out, Tensor lhs, Tensor rhs, Tensor? bias, "
      "Tensor lhs_scale, Tensor rhs_scale) -> ()");
  ops.impl("linear_quant", torch::kPrivateUse1, linear_quant);

  ops.def("memory_efficient_attention_alibi",
          &memory_efficient_attention_alibi);
  ops.impl("memory_efficient_attention_alibi", c10::kPrivateUse1,
           &memory_efficient_attention_alibi);

  ops.def("dispatch_bgmv", &dispatch_bgmv);
  ops.impl("dispatch_bgmv", c10::kPrivateUse1, &dispatch_bgmv);

  ops.def("dispatch_bgmv_low_level", &dispatch_bgmv_low_level);
  ops.impl("dispatch_bgmv_low_level", c10::kPrivateUse1,
           &dispatch_bgmv_low_level);

  ops.def(
      "fused_grouped_topk(Tensor! topk_weights, Tensor! topk_ids, Tensor "
      "gating_output, SymInt topk, bool renormalize, SymInt num_expert_group, "
      "SymInt topk_group, Tensor e_score_correction_bias, str scoring_func) -> "
      "()");
  ops.impl("fused_grouped_topk", c10::kPrivateUse1, &fused_grouped_topk);

  ops.def("dynamic_split", &dynamic_split);
  ops.impl("dynamic_split", c10::kPrivateUse1, &dynamic_split);

  ops.def(
      "dynamic_per_token_group_fp8_quant(Tensor! out, Tensor! scale, "
      "Tensor input, int group_size) -> ()");
  ops.impl("dynamic_per_token_group_fp8_quant", torch::kPrivateUse1,
           dynamic_per_token_group_fp8_quant);
  ops.def(
      "silu_mul_per_token_group_quant("
      "Tensor! out, Tensor! scale, Tensor input, "
      "int group_size) -> ()");
  ops.impl("silu_mul_per_token_group_quant", torch::kPrivateUse1,
           &silu_mul_per_token_group_quant);

  ops.def(
      "fused_add_rms_norm_per_token_group_quant_fp8("
      "Tensor! out, Tensor! residual, "
      "Tensor! scale, Tensor input, Tensor weight, "
      "float epsilon, int group_size) -> ()");
  ops.impl("fused_add_rms_norm_per_token_group_quant_fp8", torch::kPrivateUse1,
           &fused_add_rms_norm_per_token_group_quant_fp8);

  ops.def(
      "rms_norm_per_token_group_quant_fp8(Tensor! out, Tensor! scale, "
      "Tensor input, Tensor weight, float epsilon, int group_size) -> ()");
  ops.impl("rms_norm_per_token_group_quant_fp8", torch::kPrivateUse1,
           &rms_norm_per_token_group_quant_fp8);

  ops.def(
      "silu_mul_per_token_group_quant_with_size(Tensor(a!) out, Tensor(a!) "
      "scale, Tensor input, Tensor size, int group_size) -> ()");
  ops.impl("silu_mul_per_token_group_quant_with_size", torch::kPrivateUse1,
           &silu_mul_per_token_group_quant_with_size);

  ops.def(
      "rotary_embedding_with_kv_cache(Tensor! q_out, Tensor! kv_cache, "
      "Tensor q, Tensor kv, Tensor positions, Tensor cos_sin_cache, "
      "Tensor weight, Tensor slot_mapping, Tensor scale, "
      "float eps, int[] split_size, str kv_cache_dtype) -> ()");
  ops.impl("rotary_embedding_with_kv_cache", torch::kPrivateUse1,
           &rotary_embedding_with_kv_cache);

  ops.def(
      "fused_dispatch_decode(Tensor(a!)[] outputs, Tensor recv_packed, "
      "Tensor sp_split_size, int[] split_sizes) -> ()");
  ops.impl("fused_dispatch_decode", torch::kPrivateUse1,
           &fused_dispatch_decode);

  ops.def(
      "ets_moe_align_block_size(Tensor(a!) sorted_token_ids, "
      "Tensor(a!) experts_ids, "
      "Tensor(a!) num_tokens_post_pad, Tensor topk_ids, Tensor real_token_num, "
      "Tensor expert_map, int num_experts, int block_size) -> ()");
  ops.impl("ets_moe_align_block_size", torch::kPrivateUse1,
           &ets_moe_align_block_size);

  ops.def(
      "merge_attn_states(Tensor(a!) output, Tensor(a!) output_lse, "
      "Tensor prefix_output, Tensor prefix_lse, "
      "Tensor suffix_output, Tensor suffix_lse) -> ()");
  ops.impl("merge_attn_states", torch::kPrivateUse1, &merge_attn_states);

  ops.def(
      "gather_cache(Tensor(a!) src_cache, Tensor(a!) dst, "
      "Tensor(a!) block_table, Tensor(a!) cu_seq_lens, "
      "int batch_size, Tensor(a!) seq_starts) -> ()");
  ops.impl("gather_cache", torch::kPrivateUse1, &gather_cache);

  // 添加 fused_qkv_proj 算子
  ops.def(
      "fused_qkv_proj(Tensor(a!) q, Tensor(a!) kv, Tensor x, Tensor weight, "
      "Tensor x_scale, Tensor weight_scale, int group_size) -> ()");
  ops.impl("fused_qkv_proj", torch::kPrivateUse1, &fused_qkv_proj);

  ops.def(
      "fused_qkv_gemm_quant(Tensor(a!) q, Tensor(a!) kv, Tensor x, Tensor "
      "weight, Tensor scale, Tensor zeros, int group_size) -> ()");
  ops.impl("fused_qkv_gemm_quant", torch::kPrivateUse1, &fused_qkv_gemm_quant);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kPrivateUse1, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kPrivateUse1, &copy_blocks);

  cache_ops.def(
      "copy_blocks_mla(Tensor(a!)[] kv_caches, Tensor block_mapping) -> ()");
  // cache_ops.impl("copy_blocks_mla", torch::kPrivateUse1, &copy_blocks_mla);

  // Reshape the key and value tensors and cache them.
  // TODO change to Tensor
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  float k_scale, float v_scale, float k_zero, float "
      "v_zero) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kPrivateUse1, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "Tensor! key_cache,"
      "Tensor! value_cache,"
      "Tensor slot_mapping,"
      "str kv_cache_dtype,"
      "Tensor! k_scale,"
      "Tensor! v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kPrivateUse1,
                 &reshape_and_cache_flash);

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kPrivateUse1,
                 &concat_and_cache_mla);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  // cache_ops.impl("convert_fp8", torch::kPrivateUse1, &convert_fp8);
}

TORCH_LIBRARY_FRAGMENT(CONCAT(_moe, TORCH_EXTENSION_NAME), moe_ops) {
  // Apply topk softmax to the gating outputs.
  std::optional<c10::OperatorHandle> handle;

  handle =
      c10::Dispatcher::singleton().findSchema({"_moe_C::topk_softmax", ""});
  if (!handle.has_value()) {
    moe_ops.def(
        "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
        "token_expert_indices, Tensor gating_output) -> ()");
  }
  moe_ops.impl("topk_softmax", torch::kPrivateUse1, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  // moe_ops.def("moe_sum(Tensor! input, Tensor output) -> ()");
  // moe_ops.impl("moe_sum", torch::kPrivateUse1, &moe_sum);

  moe_ops.def(
      "moe_sum_pad(Tensor(a!) out, Tensor input, Tensor size, int dim, bool "
      "keepdim) -> ()");
  moe_ops.impl("moe_sum_pad", torch::kPrivateUse1, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  handle = c10::Dispatcher::singleton().findSchema(
      {"_moe_C::moe_align_block_size", ""});
  if (!handle.has_value()) {
    moe_ops.def(
        "moe_align_block_size(Tensor topk_ids, int num_experts,"
        "                     int block_size, Tensor! sorted_token_ids,"
        "                     Tensor! experts_ids,"
        "                     Tensor! num_tokens_post_pad) -> ()");
  }
  moe_ops.impl("moe_align_block_size", torch::kPrivateUse1,
               &moe_align_block_size);

  // temporarily adapted from
  // https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
  handle = c10::Dispatcher::singleton().findSchema(
      {"_moe_C::sgl_moe_align_block_size", ""});
  if (!handle.has_value()) {
    moe_ops.def(
        "sgl_moe_align_block_size(Tensor topk_ids, int num_experts,"
        "                         int block_size, Tensor! sorted_token_ids,"
        "                         Tensor! experts_ids,"
        "                         Tensor! num_tokens_post_pad) -> ()");
  }
  moe_ops.impl("sgl_moe_align_block_size", torch::kPrivateUse1,
               &sgl_moe_align_block_size);

  // moe_ops.def(
  //     "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
  //     "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
  //     "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
  //     "int b_q_type, SymInt size_m, "
  //     "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
  //     "topk, "
  //     "int moe_block_size, bool replicate_input, bool apply_weights)"
  //     " -> Tensor");
  // conditionally compiled so impl registration is in source file
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

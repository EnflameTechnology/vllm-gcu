/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <c10/util/env.h>

#include <string>
#include <tuple>
#include <vector>

#define EXPECT_TRUE(condition, op)                \
  do {                                            \
    if (!(condition)) {                           \
      fprintf(stderr, "op %s has mismatch", #op); \
    }                                             \
  } while (0)

// Fallback CPU logging macro for NDEBUG builds
#define VLLM_FALLBACK_CPU_LOG(op_name, message)                              \
  do {                                                                        \
    auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");        \
    if (fallback_ops.has_value() &&                                          \
        (fallback_ops->find(op_name) != std::string::npos ||                 \
         (*fallback_ops) == "all")) {                                        \
      fprintf(stderr, "[VLLM_FALLBACK_CPU] %s: %s\n", op_name, message);    \
    }                                                                         \
  } while (0)

// FP8 format constants
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E5M2_MAX = 57344.0f;

void vllmGetEPIndices(at::Tensor& ep_count, at::Tensor& ep_token_indices,
                      at::Tensor& ep_valid_token_indices,
                      const at::Tensor& topk_ids, int expert_per_rank,
                      int ep_size);

void vllmRmsNorm(at::Tensor& output, const at::Tensor& input,
                 const at::Tensor& gamma, const double epsilon);

void vllmConcatAndCacheMla(at::Tensor& kv_cache, const at::Tensor& kv_c,
                           const at::Tensor& k_pe,
                           const at::Tensor& slot_mapping,
                           const char* kv_cache_dtype, const at::Tensor& scale);

// Native implementation using torch operations
// Parameters: q, kv, x, weight, x_scale, weight_scale, group_size
void extsFusedQKVProj(at::Tensor& q, at::Tensor& kv, const at::Tensor& x,
                      const at::Tensor& weight, const at::Tensor& x_scale,
                      const at::Tensor& weight_scale, int64_t group_size);

/**
 * @brief Sum reduction computation
 * @param output: The output tensor
 * @param input: The input tensor
 * @param real_bs: The real batch size tensor
 * @param dimensions: The dimensions to reduce sum
 * @param keepdims: Whether the output tensor has dim retained or not
 * @param data_type: Data type
 * @param stream: Optional stream parameter (not used in torch implementation)
 */
void extsSum(at::Tensor& output, const at::Tensor& input,
             const at::Tensor& real_bs, const at::IntArrayRef& dimensions,
             const bool keepdims, const at::ScalarType& data_type,
             void* stream = nullptr);

void extsMoeAlignBlockSize(at::Tensor& sorted_topk_ids, at::Tensor& expert_ids,
                           at::Tensor& num_tokens_post_pad,
                           const at::Tensor& topk_ids,
                           const at::Tensor& real_token_num,
                           const at::Tensor& expert_map, int64_t num_experts,
                           int64_t block_size);

/**
 * @brief Moe align block size without real_token_num parameter
 * @param sorted_token_ids: Sorted token indices according to their allocated
 * expert
 * @param experts_ids: The assigned expert index for each block
 * @param num_tokens_post_pad: Number of tokens after padding
 * @param topk_ids: The top-k expert indices for each token
 * @param num_experts: Number of experts
 * @param block_size: Block size
 * @return int: 0 for success, non-zero for error
 */
int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids, at::Tensor& experts_ids,
                          at::Tensor& num_tokens_post_pad,
                          const at::Tensor& topk_ids, int num_experts,
                          int block_size);

/**
 * @brief Moe align block size with real_token_num parameter
 * @param sorted_token_ids: Sorted token indices according to their allocated
 * expert
 * @param experts_ids: The assigned expert index for each block
 * @param num_tokens_post_pad: Number of tokens after padding
 * @param topk_ids: The top-k expert indices for each token
 * @param real_token_num: The actual number of valid tokens
 * @param num_experts: Number of experts
 * @param block_size: Block size
 * @return int: 0 for success, non-zero for error
 */
int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids, at::Tensor& experts_ids,
                          at::Tensor& num_tokens_post_pad,
                          const at::Tensor& topk_ids,
                          const at::Tensor& real_token_num, int num_experts,
                          int block_size);

void vllmSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
                                   const at::Tensor& in,
                                   const int32_t group_size);

/**
 * @brief RMS Norm Per Token Group Quantization to FP8 (implementation)
 */
void vllmRmsNormPerTokenGroupQuantFp8Impl(
    at::Tensor& output, at::Tensor& scaling, const at::Tensor& input,
    const at::Tensor& weight, const float epsilon, const int64_t group_size,
    at::ScalarType output_dtype);

/**
 * @brief RMS Norm Per Token Group Quantization to FP8
 * @param output: Output quantized tensor
 * @param scaling: Scaling factors tensor
 * @param input: Input tensor to be normalized and quantized
 * @param weight: Weight tensor for post-normalization scaling
 * @param epsilon: Numerical stability constant for RMS normalization
 * @param group_size: Number of consecutive elements to group together for
 * quantization
 */
void vllmRmsNormPerTokenGroupQuantFp8(at::Tensor& output, at::Tensor& scaling,
                                      const at::Tensor& input,
                                      const at::Tensor& weight,
                                      const float epsilon,
                                      const int64_t group_size);

/**
 * @brief In-place fused Add and RMS Normalization
 * @param input: Input tensor, modified in-place
 * @param residual: Residual tensor
 * @param weight: Weight tensor
 * @param epsilon: Value added to denominator for numerical stability
 */
void vllmFusedAddRmsNorm(at::Tensor& input, at::Tensor& residual,
                         const at::Tensor& weight, float epsilon);

void extSiluAndMul(at::Tensor& out, const at::Tensor& in,
                   const at::Tensor& size = {});

/**
 * @brief Rotary embedding with KV cache
 * @param q_out: Output query tensor
 * @param kv_cache: KV cache tensor
 * @param q: Input query tensor
 * @param kv: Input key-value tensor
 * @param positions: Position tensor
 * @param cos_sin_cache: Cosine and sine cache
 * @param weight: Weight tensor
 * @param slot_mapping: Slot mapping tensor
 * @param scale: Scale tensor
 * @param eps: Epsilon value
 * @param split_size: Split size vector
 * @param kv_cache_dtype: KV cache data type
 */
void extsRotaryEmbeddingWithKVCache(at::Tensor& q_out, at::Tensor& kv_cache,
                                    const at::Tensor& q, const at::Tensor& kv,
                                    const at::Tensor& positions,
                                    const at::Tensor& cos_sin_cache,
                                    const at::Tensor& weight,
                                    const at::Tensor& slot_mapping,
                                    const at::Tensor& scale, double eps,
                                    const std::vector<int64_t>& split_size,
                                    const std::string& kv_cache_dtype = "auto");

/**
 * @brief Fused dispatch decode
 */
void extsFusedDispatchDecode(std::vector<at::Tensor>& outputs,
                             const at::Tensor& input,
                             const at::Tensor& split_sizes_tensor,
                             const std::vector<int64_t>& split_sizes);

/**
 * @brief Paged Attention V1
 */
void vllmPagedAttentionV1(at::Tensor& out, const at::Tensor& q,
                          const at::Tensor& k, const at::Tensor& v,
                          const at::Tensor& head_mapping, const float scale,
                          const at::Tensor& block_tables,
                          const at::Tensor& context_lens, const int block_size,
                          const int max_context_len,
                          const at::Tensor& alibi_slopes = at::Tensor(),
                          const std::string kv_cache_dtype = "int8",
                          const float k_scale = 1.0f, const float k_zp = 0.0f,
                          const float v_scale = 1.0f, const float v_zp = 0.0f,
                          const at::Tensor& out_scales = at::Tensor());

/**
 * @brief SiLU and multiplication
 */
void vllmSiluAndMul(at::Tensor& output, const at::Tensor& input);

/**
 * @brief Fused Add RMS Norm with quantization
 */
void vllmFusedAddRmsNormQuant(at::Tensor& output, const at::Tensor& input,
                              const at::Tensor& residual,
                              const at::Tensor& weight, double epsilon,
                              const at::Tensor& scaling);

/**
 * @brief Linear quantization operator
 */
void atenLinearQuant(at::Tensor& out, const at::Tensor& lhs,
                     const at::Tensor& rhs, const at::Tensor& bias,
                     const at::Tensor& lhs_scale, const at::Tensor& rhs_scale);

/**
 * @brief Fused MoE non-gather quantization kernel (general version)
 */
//void vllmInvokeFusedMoeNonGatherQuantKernel(
//    at::Tensor& c, const at::Tensor& a, const at::Tensor& b,
//    const at::Tensor& scale, int64_t gs, const at::Tensor& deq_zeros,
//    const at::Tensor& bias, const at::Tensor& topk_weights,
//   const at::Tensor& topk_ids, const at::Tensor& sorted_token_ids,
//    const at::Tensor& experts_ids, const at::Tensor& num_tokens_post_pad,
//    const at::Tensor& real_token_num, bool mul_routed_weight, int64_t topk,
//    int64_t block_size);

/**
 * @brief Fused MoE non-gather quantization kernel (W8A8 version)
 */
void vllmInvokeFusedMoeNonGatherQuantKernel(
    at::Tensor &C,       // Output tensor C [M, topk, N] - inplace modification
    const at::Tensor &A, // Input tensor A [*, K] (fp16/bf16)
    const at::Tensor &B, // Weight tensor B [E, N, K] (int8)
    const at::Tensor &AScale, // Scale for A [M, K/gs] (fp32)
    const at::Tensor &Scale,  // Scale for w8a8 quant [E, N/gs, K/gs] (fp32)
    int64_t gs,               // Group size for per-group quantization (128)
    const at::Tensor
        &Deq_Zeros, // Zero point for dequant (unused, kept for compatibility)
    const at::Tensor &bias, // Bias tensor (unused, kept for compatibility)
    const at::Tensor &topk_weights, // Top-k expert weights [M, topk] (fp32)
    const at::Tensor &topk_ids,     // Top-k expert indices [M, topk] (int32)
    const at::Tensor
        &sorted_token_ids, // Sorted token indices [num_tokens_post_pad] (int32)
    const at::Tensor &experts_ids, // Expert index for each block (int32)
    const at::Tensor &num_tokens_post_pad, // Number of tokens after padding [1]
    const at::Tensor &real_token_num,      // Actual number of valid tokens [1]
    bool mul_routed_weight, // Flag for topk_weights participation
    int64_t topk,           // Number of experts for each token
    int64_t b);

/**
 * @brief SiLU multiplication per token group quantization
 */
void extsSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
                                   const at::Tensor& in, const at::Tensor& size,
                                   const int32_t group_size);

/**
 * @brief Fused Add RMS Norm per token group quantization FP8
 */
void vllmFusedAddRmsNormPerTokenGroupQuantFp8(
    at::Tensor& output, at::Tensor& residual_update, at::Tensor& scale,
    const at::Tensor& input, const at::Tensor& residual,
    const at::Tensor& weight, float epsilon, int32_t group_size);

/**
 * @brief Dynamic per token group FP8 quantization
 */
void vllmDynamicPerTokenGroupFP8Quant(at::Tensor& output, at::Tensor& scale,
                                      const at::Tensor& input,
                                      const int32_t group_size);

/**
 * @brief Dynamic split operation
 */
void extsDynamicSplit(std::vector<at::Tensor>& outputs, const at::Tensor& input,
                      const at::Tensor& input_size,
                      const std::vector<int64_t>& split_size,
                      const int64_t dim);

/**
 * @brief Helper function for stable topk
 */
std::tuple<at::Tensor, at::Tensor> stableTopkMultiDim(const at::Tensor& input,
                                                      int64_t k,
                                                      int64_t dim = -1,
                                                      bool largest = true);

/**
 * @brief Grouped TopK expert selection for mixture-of-experts models (with
 * bias)
 * @param gating: Input gating output tensor
 * @param topk: Number of experts to select per token
 * @param renormalize: Whether to perform L1 normalization on TopK weights
 * @param num_expert_group: Number of expert groups
 * @param topk_group: Number of TopK groups to select per token
 * @param e_score_correction_bias: Expert score correction bias tensor
 * @param scoring_func: Scoring function ("softmax" or "sigmoid")
 * @return std::tuple<at::Tensor, at::Tensor>: TopK weights and indices
 */
std::tuple<at::Tensor, at::Tensor> vllmGroupedTopk(
    const at::Tensor& gating, const int32_t topk, const bool renormalize,
    const int32_t num_expert_group, const int32_t topk_group,
    const at::Tensor& e_score_correction_bias,
    const char* scoring_func = "softmax");

/**
 * @brief Grouped TopK expert selection for mixture-of-experts models (without
 * bias)
 * @param gating: Input gating output tensor
 * @param topk: Number of experts to select per token
 * @param renormalize: Whether to perform L1 normalization on TopK weights
 * @param num_expert_group: Number of expert groups
 * @param topk_group: Number of TopK groups to select per token
 * @param scoring_func: Scoring function ("softmax" or "sigmoid")
 * @return std::tuple<at::Tensor, at::Tensor>: TopK weights and indices
 */
std::tuple<at::Tensor, at::Tensor> vllmGroupedTopk(
    const at::Tensor& gating, const int32_t topk, const bool renormalize,
    const int32_t num_expert_group, const int32_t topk_group,
    const char* scoring_func = "softmax");

// From aten_linear_quant/aten_linear_quant_check.h
bool atenLinearQuantCheck(const std::tuple<at::Tensor>& cpu_tensor_tuple,
                          const std::tuple<at::Tensor>& target_tensor_tuple);

// From ext_silu_and_mul/ext_silu_and_mul_check.h
bool extSiluAndMulCheck(const std::tuple<at::Tensor>& cpu_tensor_tuple,
                        const std::tuple<at::Tensor>& target_tensor_tuple);

// From exts_dynamic_split/exts_dynamic_split_check.h
bool extsDynamicSplitCheck(const std::tuple<at::Tensor>& cpu_tensor_tuple,
                           const std::tuple<at::Tensor>& target_tensor_tuple);

// From exts_fused_dispatch_decode/exts_fused_dispatch_decode_check.h
bool extsFusedDispatchDecodeCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

// From exts_fused_qkv_proj/exts_fused_qkv_proj_check.h
bool extsFusedQKVProjCheck(
    std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From exts_moe_align_block_size/exts_moe_align_block_size_check.h
bool extsMoeAlignBlockSizeCheck(
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& target_tensor_tuple);

// From
// exts_rotary_embedding_with_kvcache/exts_rotary_embedding_with_kvcache_check.h
bool extsRotaryEmbeddingWithKVCacheCheck(
    const std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From
// exts_silu_mul_per_token_group_quant/exts_silu_mul_per_token_group_quant_check.h
bool extsSiluMulPerTokenGroupQuantCheck(
    const std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From exts_sum/exts_sum_check.h
bool extsSumCheck(const std::tuple<at::Tensor>& cpu_tensor_tuple,
                  const std::tuple<at::Tensor>& target_tensor_tuple);

// From vllm_concat_and_cache_mla/vllm_concat_and_cache_mla_check.h
bool vllmConcatAndCacheMlaCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

// From
// vllm_dynamic_per_token_group_fp8_quant/vllm_dynamic_per_token_group_fp8_quant_check.h
bool vllmDynamicPerTokenGroupFP8QuantCheck(
    const std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From vllm_fused_add_rms_norm/vllm_fused_add_rms_norm_check.h
bool vllmFusedAddRmsNormCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

// From
// vllm_fused_add_rms_norm_per_token_group_quant_fp8/vllm_fused_add_rms_norm_per_token_group_quant_fp8_check.h
bool vllmFusedAddRmsNormPerTokenGroupQuantFp8Check(
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& target_tensor_tuple);

// From vllm_fused_add_rms_norm_quant/vllm_fused_add_rms_norm_quant_check.h
bool vllmFusedAddRmsNormQuantCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

// From vllm_get_ep_indices/vllm_get_ep_indices_check.h
bool vllmGetEPIndicesCheck(
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor, at::Tensor, at::Tensor>& target_tensor_tuple);

// From vllm_grouped_topk/vllm_grouped_topk_check.h
bool vllmGroupedTopkCheck(
    std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From
// vllm_invoke_fused_moe_non_gather_quant/vllm_invoke_fused_moe_non_gather_quant_check.h
bool vllmInvokeFusedMoeNonGatherQuantCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

// From vllm_moe_align_block_size/vllm_moe_align_block_size_check.h
bool vllmMoeAlignBlockSizeCheck(
    std::tuple<at::Tensor, at::Tensor, at::Tensor>& cpu_tensor_tuple,
    std::tuple<at::Tensor, at::Tensor, at::Tensor>& target_tensor_tuple);

// From vllm_paged_attention_v1/vllm_paged_attention_v1_check.h
bool vllmPagedAttentionV1Check(std::tuple<at::Tensor>& cpu_tensor_tuple,
                               std::tuple<at::Tensor>& target_tensor_tuple);

// From vllm_rms_norm/vllm_rms_norm_check.h
bool vllmRmsNormCheck(std::tuple<at::Tensor>& cpu_tensor_tuple,
                      std::tuple<at::Tensor>& target_tensor_tuple);

// From
// vllm_rms_norm_per_token_group_quant_fp8/vllm_rms_norm_per_token_group_quant_fp8_check.h
bool vllmRmsNormPerTokenGroupQuantFp8Check(
    std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

// From vllm_silu_and_mul/vllm_silu_and_mul_check.h
bool vllmSiluAndMulCheck(const std::tuple<at::Tensor>& cpu_tensor_tuple,
                         const std::tuple<at::Tensor>& target_tensor_tuple);

// From
// vllm_silu_mul_per_token_group_quant/vllm_silu_mul_per_token_group_quant_check.h
bool vllmSiluMulPerTokenGroupQuantCheck(
    std::tuple<at::Tensor, at::Tensor>& cpu_tensor_tuple,
    std::tuple<at::Tensor, at::Tensor>& target_tensor_tuple);

bool vllmInvokeFusedMoeNonGatherQuantKernelCheck(
    const std::tuple<at::Tensor>& cpu_tensor_tuple,
    const std::tuple<at::Tensor>& target_tensor_tuple);

/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#pragma once

#include <torch/torch.h>

#include <vector>

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
// Parameters follow the interface definition: q, kv, x, weight, x_scale,
// weight_scale, group_size
void extsFusedQKVProj(at::Tensor& q, at::Tensor& kv, const at::Tensor& x,
                      const at::Tensor& weight, const at::Tensor& x_scale,
                      const at::Tensor& weight_scale, int64_t group_size);

/**
 * @brief This function performs a sum reduction computation.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param real_bs: The real batch size tensor.
 * @param dimensions: The dimensions to reduce sum.
 * @param keepdims: Whether the output tensor has dim retained or not.
 * @param data_type: Data type.
 * @param stream: Optional stream parameter (not used in torch implementation).
 */
void extsSum(at::Tensor& output, const at::Tensor& input,
             const at::Tensor& real_bs, const at::IntArrayRef& dimensions,
             const bool keepdims, const at::ScalarType& data_type,
             void* stream = nullptr);

// PyTorch实现的ets_moe_align_block_size
void extsMoeAlignBlockSize(at::Tensor& sorted_topk_ids, at::Tensor& expert_ids,
                           at::Tensor& num_tokens_post_pad,
                           const at::Tensor& topk_ids,
                           const at::Tensor& real_token_num,
                           const at::Tensor& expert_map, int64_t num_experts,
                           int64_t block_size);

/**
 * @brief Moe align block size native implementation without real_token_num
 * parameter
 *
 * @param sorted_token_ids: Sorted token indices according to their allocated
 *                            expert. Max shape:
 *                            [total_token*topk + topk*(block_size - 1)].
 * @param experts_ids: The assigned expert index for each block. Max
 *                            shape: [total_token*topk + expert].
 * @param num_tokens_post_pad: Shape: [1]. Number of tokens after padding.
 * @param topk_ids: Shape: [total_tokens, topk]. The top-k expert
 *                            indices for each token.
 * @param num_experts: Number of experts.
 * @param block_size: Block size.
 * @return int: 0 for success, non-zero for error.
 */
int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids, at::Tensor& experts_ids,
                          at::Tensor& num_tokens_post_pad,
                          const at::Tensor& topk_ids, int num_experts,
                          int block_size);

/**
 * @brief Moe align block size native implementation with real_token_num
 * parameter
 *
 * @param sorted_token_ids: Sorted token indices according to their allocated
 *                            expert. Max shape:
 *                            [total_token*topk + topk*(block_size - 1)].
 * @param experts_ids: The assigned expert index for each block. Max
 *                            shape: [total_token*topk + expert].
 * @param num_tokens_post_pad: Shape: [1]. Number of tokens after padding.
 * @param topk_ids: Shape: [total_tokens, topk]. The top-k expert
 *                            indices for each token.
 * @param real_token_num: Shape: [1]. The actual number of valid tokens.
 * @param num_experts: Number of experts.
 * @param block_size: Block size.
 * @return int: 0 for success, non-zero for error.
 */
int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids, at::Tensor& experts_ids,
                          at::Tensor& num_tokens_post_pad,
                          const at::Tensor& topk_ids,
                          const at::Tensor& real_token_num, int num_experts,
                          int block_size);

void vllmSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
                                   const at::Tensor& in,
                                   const int32_t group_size);

// /**
//  * @brief RMS Norm Per Token Group Quantization implementation using PyTorch
//  * native operations
//  *
//  * This function performs RMS normalization followed by group-wise
//  quantization
//  * to FP8 format. It uses PyTorch's built-in tensor operations for efficient
//  * computation.
//  *
//  * @param output       Output quantized tensor of shape [N, D]. This tensor
//  * will store the quantized results in the specified FP8 format (e4m3fn or
//  * e5m2).
//  * @param scaling      Scaling factors tensor of shape [N, D/group_size].
//  Each
//  * element represents the scaling factor for one group of consecutive
//  elements.
//  * @param input        Input tensor of shape [N, D]. Can be any
//  floating-point
//  * format. N represents the batch/token dimension, D represents the feature
//  * dimension.
//  * @param weight       Weight tensor of shape [D]. Used for scaling after RMS
//  * normalization.
//  * @param epsilon      Small value added to variance for numerical stability
//  * (typically 1e-6). Prevents division by zero in RMS normalization.
//  * @param group_size   Number of consecutive elements to group together for
//  * quantization. Must divide D evenly. Common values: 32, 64, 128, 256.
//  * @param output_dtype Target quantization format (torch::kFloat8_e4m3fn or
//  * torch::kFloat8_e5m2).
//  */
// inline void vllmRmsNormPerTokenGroupQuantFp8Impl(
//     at::Tensor& output, at::Tensor& scaling, const at::Tensor& input,
//     const at::Tensor& weight, const float epsilon, const int64_t group_size,
//     at::ScalarType output_dtype) {
//   // Convert inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);

//   // Get tensor dimensions
//   auto sizes = input_f32.sizes();
//   int64_t n = 1;
//   for (int i = 0; i < sizes.size() - 1; ++i) {
//     n *= sizes[i];
//   }
//   int64_t d = sizes[sizes.size() - 1];
//   int64_t group_num = d / group_size;

//   // Determine FP8 max value based on output type
//   float fp8_max;
//   if (output_dtype == torch::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;  // FP8_E4M3_MAX
//   } else if (output_dtype == torch::kFloat8_e5m2) {
//     fp8_max = 57344.0f;  // FP8_E5M2_MAX
//   } else {
//     fp8_max = 127.0f;  // For int8 fallback
//   }
//   float recp_fp8 = 1.0f / fp8_max;

//   // Step 1: Compute RMS normalization
//   // variance = mean(input^2, dim=-1, keepdim=True)
//   auto variance = at::mean(input_f32 * input_f32, /*dim=*/-1,
//   /*keepdim=*/true);

//   // Apply RMS normalization: input * rsqrt(variance + epsilon)
//   auto normalized = input_f32 * at::rsqrt(variance + epsilon);

//   // Step 2: Apply weight scaling
//   auto weighted = normalized * weight_f32;

//   // Step 3: Group quantization
//   // Reshape to [N, group_num, group_size] for group-wise operations
//   auto reshaped = weighted.view({n, group_num, group_size});

//   // Find maximum absolute value per group: [N, group_num]
//   auto max_result = at::max(at::abs(reshaped), /*dim=*/-1);
//   auto group_max = std::get<0>(max_result);

//   // Calculate scaling factors: [N, group_num]
//   auto scaling_factors = group_max * recp_fp8;

//   // Expand scaling factors to match reshaped tensor: [N, group_num, 1]
//   auto scaling_expanded = scaling_factors.unsqueeze(-1);

//   // Quantize: divide by scaling and clamp to fp8 range
//   auto quantized = (reshaped / scaling_expanded).clamp(-fp8_max, fp8_max);

//   // Reshape back to original shape [N, D]
//   auto quantized_reshaped = quantized.view({n, d});

//   // Convert to target output type
//   at::Tensor final_output;
//   if (output_dtype == torch::kFloat8_e4m3fn ||
//       output_dtype == torch::kFloat8_e5m2) {
//     final_output = quantized_reshaped.to(output_dtype);
//   } else {
//     // For integer types, round first then convert
//     final_output = at::round(quantized_reshaped).to(output_dtype);
//   }

//   // Copy results to output tensors
//   output.copy_(final_output);
//   scaling.copy_(scaling_factors);
// }

// /**
//  * @brief Unified entry point for RMS Norm Per Token Group Quantization
//  *
//  * Automatically determines the output data type from the output tensor and
//  * calls the appropriate implementation.
//  *
//  * @param output    Pre-allocated output tensor with the desired quantization
//  * format. Shape: [N, D]. The scalar type determines the quantization target.
//  * @param scaling   Pre-allocated scaling tensor for storing group scaling
//  * factors. Shape: [N, D/group_size]. Must be float32 format.
//  * @param input     Input tensor to be normalized and quantized. Shape: [N,
//  D].
//  * @param weight    Weight tensor for post-normalization scaling. Shape: [D].
//  * @param epsilon   Numerical stability constant for RMS normalization.
//  * @param group_size Size of each quantization group. Must divide D evenly.
//  */
// inline void vllmRmsNormPerTokenGroupQuantFp8(
//     at::Tensor& output, at::Tensor& scaling, const at::Tensor& input,
//     const at::Tensor& weight, const float epsilon, const int64_t group_size)
//     {
//   auto output_dtype = output.scalar_type();
//   vllmRmsNormPerTokenGroupQuantFp8Impl(output, scaling, input, weight,
//   epsilon,
//                                        group_size, output_dtype);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// /**
//  * @brief In-place fused Add and RMS Normalization (Native implementation
//  using
//  * torch interfaces).
//  *
//  * @param input: Input tensor of shape [..., hidden_size]. Modified in-place.
//  * @param residual: Residual tensor of shape [..., hidden_size].
//  * @param weight: Weight tensor of shape [hidden_size].
//  * @param epsilon: Value added to denominator for numerical stability.
//  */
// inline void vllmFusedAddRmsNorm(at::Tensor& input, at::Tensor& residual,
//                                 const at::Tensor& weight, float epsilon) {
//   auto data_type = input.dtype();

//   // Follow the exact logic from rule.MD reference implementation
//   at::Tensor input_tensor, weight_tensor, residual_tensor;

//   if (data_type != at::kBFloat16) {
//     input_tensor =
//         input;  // torch.tensor(input) - but input is already a tensor
//     weight_tensor = weight;      // torch.tensor(weight)
//     residual_tensor = residual;  // torch.tensor(residual)
//   } else {
//     input_tensor = input;
//     weight_tensor = weight;
//     residual_tensor = residual;
//   }

//   // Always convert to float32 for computation (following rule.MD exactly)
//   auto input_tensor_f32 = input_tensor.to(at::kFloat);
//   auto weight_f32 = weight_tensor.to(at::kFloat);
//   auto residual_tensor_f32 = residual_tensor.to(at::kFloat);

//   // Add input and residual: input = input + residual
//   input_tensor_f32 = input_tensor_f32 + residual_tensor_f32;

//   // Compute RMS norm square: x^2
//   auto rms_norm_square = input_tensor_f32.pow(2);

//   // Compute rsqrt(mean(x^2) + epsilon)
//   auto rms_norm_rsqrt =
//       at::rsqrt(rms_norm_square.mean(-1, /*keepdim=*/true) + epsilon);

//   // Apply RMS normalization and weight scaling: x * rsqrt * weight
//   auto rms_output_tensor = input_tensor_f32 * rms_norm_rsqrt * weight_f32;

//   // Convert back to original dtype and copy in-place to input tensor
//   input.copy_(rms_output_tensor.to(data_type));
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// inline void extSiluAndMul(at::Tensor& out, const at::Tensor& in,
//                           const at::Tensor& size = {}) {
//   // Get tensor properties
//   int64_t rank = in.dim();
//   int64_t hidden_size = in.size(-1);
//   int64_t d = hidden_size / 2;
//   int64_t batch_size = in.size(0);

//   // Extract real_size from tensor or use default
//   int64_t real_size;
//   if (size.defined() && size.numel() > 0) {
//     // Extract value from scalar tensor (convert to int64 for consistency)
//     real_size = size.cpu().item<int32_t>();
//   } else {
//     real_size = batch_size;  // Default to full batch size
//   }

//   // Check constraints
//   TORCH_CHECK(real_size <= batch_size, "real_size (", real_size,
//               ") must be <= batch_size (", batch_size, ")");

//   // Check tensor rank (support 2D and 3D)
//   TORCH_CHECK(rank == 2 || rank == 3, "Only 2D and 3D tensors are
//   supported");

//   // Check that the last dimension of input tensor is even
//   TORCH_CHECK(hidden_size % 2 == 0,
//               "Last dimension of input_tensor must be even");

//   // Check that the last dimension of output tensor is half of input
//   TORCH_CHECK(out.size(-1) == d,
//               "Last dimension of out_tensor must be half of input_tensor's "
//               "last dimension");

//   // Check that all dimensions except the last one match (considering
//   real_size) TORCH_CHECK(
//       out.dim() == rank,
//       "Input and output tensors must have the same number of dimensions");
//   TORCH_CHECK(out.size(0) >= real_size,
//               "Output tensor's first dimension must be >= real_size");
//   for (int64_t i = 1; i < rank - 1; ++i) {
//     TORCH_CHECK(out.size(i) == in.size(i),
//                 "Dimension " + std::to_string(i) +
//                     " of output tensor must match input tensor");
//   }

//   // Get the effective region of input tensor (first real_size batches)
//   auto effective_input = in.narrow(0, 0, real_size);

//   // Split the effective input tensor along the last dimension
//   // This works for both 2D and 3D tensors due to Ellipsis
//   auto x1 = effective_input.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(0, d)});
//   auto x2 = effective_input.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(d, hidden_size)});

//   // Apply SiLU activation function: x1 / (1 + exp(-x1))
//   auto silu = x1 / (1 + torch::exp(-x1));

//   // Multiply by the second half
//   auto result = silu * x2;

//   // Convert result to the same dtype as output and copy only to the
//   effective
//   // region
//   out.narrow(0, 0, real_size).copy_(result.to(out.scalar_type()));
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// inline void extsRotaryEmbeddingWithKVCache(
//     at::Tensor& q_out, at::Tensor& kv_cache, const at::Tensor& q,
//     const at::Tensor& kv, const at::Tensor& positions,
//     const at::Tensor& cos_sin_cache, const at::Tensor& weight,
//     const at::Tensor& slot_mapping, const at::Tensor& scale, double eps,
//     const std::vector<int64_t>& split_size,
//     const std::string& kv_cache_dtype = "auto") {
//   // Convert inputs to float32 for computation
//   auto q_f32 = q.to(at::kFloat);
//   auto kv_f32 = kv.to(at::kFloat);
//   auto cos_sin_f32 = cos_sin_cache.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);

//   auto q_sizes = q_f32.sizes();
//   auto kv_sizes = kv_f32.sizes();

//   int64_t tokens = q_sizes[0];
//   int64_t q_dim = q_sizes[1];
//   int64_t kv_dim = kv_sizes[1];
//   int64_t rms_dim = split_size[0];
//   int64_t rot_dim = 32;  // ROT_DIM from the test code

//   // Apply rotary embedding to q using tensor operations
//   auto q_result = q_f32.clone();

//   // Extract rotary dimensions from q (first rot_dim*2 elements)
//   auto q_rot = q_f32.narrow(1, 0, rot_dim * 2).view({tokens, rot_dim, 2});
//   auto q_x = q_rot.select(2, 0);  // shape: [tokens, rot_dim]
//   auto q_y = q_rot.select(2, 1);  // shape: [tokens, rot_dim]

//   // Extract cos and sin from cos_sin_cache tensor
//   auto cos_vals =
//       cos_sin_f32.narrow(1, 0, rot_dim);  // shape: [tokens, rot_dim]
//   auto sin_vals =
//       cos_sin_f32.narrow(1, rot_dim, rot_dim);  // shape: [tokens, rot_dim]

//   // Apply rotary transformation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
//   auto q_x_new = q_x * cos_vals - q_y * sin_vals;
//   auto q_y_new = q_x * sin_vals + q_y * cos_vals;

//   // Stack and reshape back to original format
//   auto q_rot_new = at::stack({q_x_new, q_y_new}, 2).view({tokens, rot_dim *
//   2});

//   // Update q_result with rotated values
//   q_result.narrow(1, 0, rot_dim * 2).copy_(q_rot_new);

//   // Process kv tensor
//   auto kv_result = kv_f32.clone();

//   // Extract the first rms_dim elements for RMS normalization
//   auto kv_rms_part = kv_f32.narrow(1, 0, rms_dim);  // shape: [tokens,
//   rms_dim]

//   // Compute RMS normalization using tensor operations
//   auto variance = at::mean(kv_rms_part * kv_rms_part, /*dim=*/1,
//                            /*keepdim=*/true);        // shape: [tokens, 1]
//   auto rms_norm_factor = at::rsqrt(variance + eps);  // shape: [tokens, 1]

//   // Apply RMS normalization and weight scaling
//   auto kv_normalized = kv_rms_part * rms_norm_factor;  // Broadcasting:
//   [tokens,
//                                                        // rms_dim] * [tokens,
//                                                        1]
//   auto kv_weighted = kv_normalized * weight_f32.unsqueeze(0);  //
//   Broadcasting:
//                                                                // [tokens,
//                                                                // rms_dim] *
//                                                                [1,
//                                                                // rms_dim]

//   // Update kv_result with normalized and weighted values
//   kv_result.narrow(1, 0, rms_dim).copy_(kv_weighted);

//   // Apply rotary embedding to k part (starting from rms_dim offset)
//   if (rms_dim + rot_dim * 2 <= kv_dim) {
//     // Extract rotary dimensions from k part
//     auto k_rot =
//         kv_f32.narrow(1, rms_dim, rot_dim * 2).view({tokens, rot_dim, 2});
//     auto k_x = k_rot.select(2, 0);  // shape: [tokens, rot_dim]
//     auto k_y = k_rot.select(2, 1);  // shape: [tokens, rot_dim]

//     // Apply rotary transformation to k using the same cos/sin values
//     auto k_x_new = k_x * cos_vals - k_y * sin_vals;
//     auto k_y_new = k_x * sin_vals + k_y * cos_vals;

//     // Stack and reshape back to original format
//     auto k_rot_new =
//         at::stack({k_x_new, k_y_new}, 2).view({tokens, rot_dim * 2});

//     // Update kv_result with rotated k values
//     kv_result.narrow(1, rms_dim, rot_dim * 2).copy_(k_rot_new);
//   }

//   // Copy results to output tensors with proper dtype conversion
//   q_out.copy_(q_result.to(q_out.dtype()));
//   kv_cache.copy_(kv_result.to(kv_cache.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// #include <vector>

// // 使用PyTorch API实现的版本
// inline void extsFusedDispatchDecode(std::vector<at::Tensor>& outputs,
//                                     const at::Tensor& input,
//                                     const at::Tensor& split_sizes_tensor,
//                                     const std::vector<int64_t>& split_sizes)
//                                     {
//   // 参数检查
//   TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
//   TORCH_CHECK(split_sizes_tensor.dim() == 1, "Split sizes tensor must be
//   1D");

//   // 计算有效行数
//   at::Tensor reduce_sum = split_sizes_tensor.sum(0, true);
//   int64_t valid_rows = reduce_sum.item<int64_t>();
//   valid_rows = std::min(valid_rows, input.size(0));

//   // 计算分割偏移量
//   std::vector<int64_t> split_offsets(split_sizes.size());
//   split_offsets[0] = 0;
//   for (size_t i = 1; i < split_sizes.size(); ++i) {
//     split_offsets[i] = split_offsets[i - 1] + split_sizes[i - 1];
//   }

//   // 生成输出张量
//   outputs.clear();
//   int64_t dim = input.dim() - 1;  // 在最后一维分割

//   for (size_t i = 0; i < split_sizes.size(); ++i) {
//     // 切片并保持连续性
//     at::Tensor slice =
//         input.slice(dim, split_offsets[i], split_offsets[i] + split_sizes[i])
//             .contiguous();

//     // 只保留有效行
//     if (valid_rows < input.size(0)) {
//       slice = slice.slice(0, 0, valid_rows);
//     }

//     outputs.push_back(slice);
//   }
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// #include <c10/util/Exception.h>

// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <vector>

// // PyTorch风格实现的PagedAttentionV1，使用张量操作而非直接内存访问
// inline void vllmPagedAttentionV1(
//     at::Tensor& out, const at::Tensor& q, const at::Tensor& k,
//     const at::Tensor& v, const at::Tensor& head_mapping, const float scale,
//     const at::Tensor& block_tables, const at::Tensor& context_lens,
//     const int block_size, const int max_context_len,
//     const at::Tensor& alibi_slopes = at::Tensor(),
//     const std::string kv_cache_dtype = "int8", const float k_scale = 1.0f,
//     const float k_zp = 0.0f, const float v_scale = 1.0f,
//     const float v_zp = 0.0f, const at::Tensor& out_scales = at::Tensor()) {
//   // 参数检查
//   TORCH_CHECK(q.dim() == 3,
//               "q should be 3D tensor [batch_size, num_heads, head_size]");
//   TORCH_CHECK(
//       block_tables.dim() == 2,
//       "block_tables should be 2D tensor [batch_size, max_blocks_per_seq]");
//   TORCH_CHECK(context_lens.dim() == 1,
//               "context_lens should be 1D tensor [batch_size]");

//   // 获取尺寸
//   int batch_size = q.size(0);
//   int num_heads = q.size(1);
//   int head_size = q.size(2);
//   int max_blocks_per_seq = block_tables.size(1);

//   // 检查KV缓存形状
//   TORCH_CHECK(k.dim() == 4,
//               "k should be 4D tensor [num_blocks, num_kv_heads, block_size, "
//               "head_size]");
//   int num_kv_heads = k.size(1);
//   TORCH_CHECK(head_size == k.size(3), "Head size of q and k should match");
//   TORCH_CHECK(block_size == k.size(2), "Block size does not match
//   k.size(2)"); TORCH_CHECK(v.dim() == 4,
//               "v should be 4D tensor [num_blocks, num_kv_heads, block_size, "
//               "head_size]");
//   TORCH_CHECK(num_kv_heads == v.size(1),
//               "Number of KV heads should match between k and v");

//   // 初始化输出
//   out.zero_();

//   // 获取设备信息，确保所有操作都在同一设备上
//   auto device = q.device();
//   auto dtype = q.dtype();

//   // 检查是否使用alibi
//   bool use_alibi = alibi_slopes.defined() && alibi_slopes.numel() > 0;

//   // 检查输出量化
//   bool use_out_quant = out_scales.defined() && out_scales.numel() > 0;

//   // 计算k_zp和v_zp的实际值
//   float k_zero_point = k_scale * k_zp;
//   float v_zero_point = v_scale * v_zp;

//   // 获取head_mapping (如果提供的话)
//   at::Tensor effective_head_mapping;
//   if (head_mapping.defined() && head_mapping.numel() > 0) {
//     effective_head_mapping = head_mapping;
//   } else {
//     // 如果没有提供head_mapping，创建一个恒等映射
//     effective_head_mapping = torch::arange(
//         num_heads, torch::TensorOptions().dtype(torch::kInt).device(device));
//   }

//   // 创建int8量化的解码方法（如果需要）
//   auto dequantize_int8 = [&](const at::Tensor& int8_tensor, float scale,
//                              float zero_point) {
//     return int8_tensor.to(torch::kFloat) * scale - zero_point;
//   };

//   // 对每个批次单独处理
//   for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
//     // 获取当前序列的上下文长度
//     int ctx_len = std::min(context_lens[seq_idx].item<int>(),
//     max_context_len);

//     // 计算需要的块数
//     int num_blocks = (ctx_len + block_size - 1) / block_size;
//     num_blocks = std::min(num_blocks, max_blocks_per_seq);

//     // 对每个头单独处理
//     for (int head_idx = 0; head_idx < num_heads; head_idx++) {
//       // 获取对应的KV头索引
//       int kv_head_idx = effective_head_mapping[head_idx].item<int>();

//       // 安全检查
//       TORCH_CHECK(kv_head_idx < num_kv_heads,
//                   "KV head index out of range: ", kv_head_idx,
//                   " >= ", num_kv_heads);

//       // 获取当前查询向量
//       at::Tensor query_vec = q[seq_idx][head_idx];

//       // 准备存储当前序列所有token的KV值
//       std::vector<at::Tensor> key_vecs;
//       std::vector<at::Tensor> value_vecs;

//       // 遍历块以收集所有KV向量
//       for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
//         // 获取KV块索引
//         int kv_block_idx = block_tables[seq_idx][block_idx].item<int>();

//         // 获取当前块中有效的token数量
//         int valid_tokens =
//             std::min(block_size, ctx_len - block_idx * block_size);

//         // 提取当前块的K和V
//         at::Tensor k_block =
//             k[kv_block_idx][kv_head_idx].slice(0, 0, valid_tokens);
//         at::Tensor v_block =
//             v[kv_block_idx][kv_head_idx].slice(0, 0, valid_tokens);

//         // 如果是int8量化，则进行解量化
//         if (kv_cache_dtype == "int8") {
//           k_block = dequantize_int8(k_block, k_scale, k_zero_point);
//           v_block = dequantize_int8(v_block, v_scale, v_zero_point);
//         }

//         // 添加到列表
//         key_vecs.push_back(k_block);
//         value_vecs.push_back(v_block);
//       }

//       // 合并所有KV向量
//       at::Tensor keys = torch::cat(key_vecs, 0);
//       at::Tensor values = torch::cat(value_vecs, 0);

//       // 计算注意力分数 (Q·K^T)
//       at::Tensor attn_scores =
//           torch::matmul(query_vec.unsqueeze(0), keys.transpose(0, 1))
//               .squeeze(0);

//       // 应用缩放因子
//       attn_scores = attn_scores * scale;

//       // 如果使用alibi，添加位置偏置
//       if (use_alibi) {
//         // 创建位置索引
//         at::Tensor positions = torch::arange(
//             ctx_len,
//             torch::TensorOptions().dtype(torch::kFloat).device(device));
//         positions = positions - ctx_len;  // 从-ctx_len到-1

//         // 获取当前头的alibi斜率
//         float alibi_slope = alibi_slopes[head_idx].item<float>();

//         // 应用alibi偏置
//         attn_scores = attn_scores + alibi_slope * positions;
//       }

//       // 仅保留有效上下文长度的分数
//       attn_scores = attn_scores.slice(0, 0, ctx_len);

//       // 应用softmax
//       at::Tensor attn_probs = torch::softmax(attn_scores, 0);

//       // 计算加权和 (attn_probs·V)
//       at::Tensor output =
//           torch::matmul(attn_probs.unsqueeze(0), values).squeeze(0);

//       // 写入结果
//       if (use_out_quant) {
//         // 输出量化处理
//         bool per_token_quant = out_scales.numel() > 1;

//         if (per_token_quant) {
//           // 每个token单独量化
//           at::Tensor scale_factors = out_scales.slice(
//               0, head_idx * head_size, (head_idx + 1) * head_size);
//           at::Tensor quant_vals = output * scale_factors;
//           quant_vals = torch::clamp(torch::round(quant_vals), -127.0, 127.0);
//           out[seq_idx][head_idx] = quant_vals.to(torch::kInt8);
//         } else {
//           // 单一量化因子
//           float scale_factor = out_scales[0].item<float>();
//           at::Tensor quant_vals = output * scale_factor;
//           quant_vals = torch::clamp(torch::round(quant_vals), -127.0, 127.0);
//           out[seq_idx][head_idx] = quant_vals.to(torch::kInt8);
//         }
//       } else {
//         // 直接复制浮点结果
//         out[seq_idx][head_idx] = output;
//       }
//     }
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmSiluAndMul(at::Tensor& output, const at::Tensor& input) {
//   // Get tensor dimensions
//   auto input_sizes = input.sizes();
//   int64_t batch_size = 1;
//   for (int i = 0; i < input_sizes.size() - 1; ++i) {
//     batch_size *= input_sizes[i];
//   }
//   int64_t hidden_size = input_sizes[input_sizes.size() - 1];
//   int64_t half_hidden_size = hidden_size / 2;

//   // Check that the last dimension of input tensor is even
//   TORCH_CHECK(hidden_size % 2 == 0,
//               "Last dimension of input tensor must be even");

//   // Check that output tensor has the correct shape
//   TORCH_CHECK(
//       output.size(-1) == half_hidden_size,
//       "Last dimension of output tensor must be half of input tensor's last "
//       "dimension");

//   // Convert input to float32 for computation
//   auto input_f32 = input.to(at::kFloat);

//   // Split the input tensor along the last dimension
//   auto x1 = input_f32.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(0,
//       half_hidden_size)});
//   auto x2 =
//       input_f32.index({torch::indexing::Ellipsis,
//                        torch::indexing::Slice(half_hidden_size,
//                        hidden_size)});

//   // Apply SiLU activation function: x1 / (1 + exp(-x1))
//   auto silu_result = x1 / (1 + torch::exp(-x1));

//   // Multiply by the second half: silu_result * x2
//   auto result = silu_result * x2;

//   // Convert back to original dtype and copy to output tensor
//   output.copy_(result.to(input.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmFusedAddRmsNormQuant(at::Tensor& output,
//                                      const at::Tensor& input,
//                                      const at::Tensor& residual,
//                                      const at::Tensor& weight, double
//                                      epsilon, const at::Tensor& scaling) {
//   // Convert inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto residual_f32 = residual.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);
//   auto scales_f32 = scaling.to(at::kFloat);

//   // Add input and residual: x = input + residual
//   auto x = input_f32 + residual_f32;

//   // Compute variance: variance = mean(x * x, dim=-1, keepdim=True)
//   auto variance = at::mean(x * x, /*dim=*/-1, /*keepdim=*/true);

//   // Apply RMS normalization: x = x * rsqrt(variance + epsilon)
//   x = x * at::rsqrt(variance + epsilon);

//   // Apply weight scaling: x = x * weight
//   x = x * weight_f32;

//   // Quantize: round(x / scales) and clamp to int8 range [-128, 127]
//   auto scales_expanded = scales_f32.unsqueeze(-1);
//   auto quant_result =
//       at::round(x / scales_expanded).clamp(-128, 127).to(at::kChar);

//   // Copy result to output tensor
//   output.copy_(quant_result);
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// #include <iostream>

// /**
//  * @brief Linear quant operator (Native implementation using torch
//  interfaces)
//  *
//  * @param out The output tensor
//  * @param lhs The input tensor of lhs
//  * @param rhs The input tensor of rhs
//  * @param bias The input tensor of bias
//  * @param lhs_scale The lhs tensor of scale
//  * @param rhs_scale The rhs tensor of scale
//  */
// inline void atenLinearQuant(at::Tensor& out, const at::Tensor& lhs,
//                             const at::Tensor& rhs, const at::Tensor& bias,
//                             const at::Tensor& lhs_scale,
//                             const at::Tensor& rhs_scale) {
//   // Convert tensors to appropriate types if needed
//   at::Tensor input_tensor = lhs.to(at::kFloat);
//   at::Tensor weight_tensor = rhs;
//   at::Tensor bias_tensor = bias.to(at::kFloat);
//   at::Tensor scale_tensor = rhs_scale;

//   // Transpose weight tensor
//   at::Tensor weight_transpose_tensor = at::transpose(weight_tensor, 0, 1);

//   // Reshape the weight tensor
//   auto weight_shape = weight_transpose_tensor.sizes().vec();
//   int64_t n = weight_shape[1];
//   int64_t k = weight_shape[0];
//   int64_t group_num = scale_tensor.size(0);
//   int64_t group_size = k / group_num;

//   std::vector<int64_t> weight_transpose_shape_tmp = {group_num, group_size,
//   n}; at::Tensor weight_split_tensor =
//       at::reshape(weight_transpose_tensor, weight_transpose_shape_tmp);

//   // Apply scale to weight
//   at::Tensor quant_result = at::mul(weight_split_tensor, scale_tensor);

//   // Reshape back
//   at::Tensor quant_result2 = at::reshape(quant_result, {k, n});

//   // Transpose again
//   at::Tensor quant_result3 = at::transpose(quant_result2, 0, 1);

//   // Convert to float32 for linear operation
//   at::Tensor aten_scale_mul_tensor = quant_result3.to(at::kFloat);

//   // Perform linear operation
//   at::Tensor output_tensor;
//   if (bias.numel() > 0) {
//     output_tensor =
//         at::linear(input_tensor, aten_scale_mul_tensor, bias_tensor);
//   } else {
//     output_tensor = at::linear(input_tensor, aten_scale_mul_tensor);
//   }

//   // Handle activation if needed (assuming no activation is required based on
//   // the interface)

//   // Handle data type clamping if needed for int8 (char) type
//   bool is_int8_type = (out.scalar_type() == at::ScalarType::Char);
//   if (is_int8_type) {
//     output_tensor = at::clamp(output_tensor, -127.0f, 127.0f);
//   }

//   // Copy result to output tensor with proper dtype conversion
//   out.copy_(output_tensor.to(out.scalar_type()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// // First algorithm: General quantization version with Deq_Zeros support
// inline void vllmInvokeFusedMoeNonGatherQuantKernel(
//     at::Tensor& c,        // Output tensor C [M, topk, N] - inplace
//     modification const at::Tensor& a,  // Input tensor A [*, K] const
//     at::Tensor& b,  // Weight tensor B [E, N, K] const at::Tensor& scale,  //
//     Scale for dequant [E, N] int64_t gs,               // Group size for
//     pre-group type dequant const at::Tensor&
//         deq_zeros,           // Zero point for dequant [E, N] or [E, K/gs, N]
//     const at::Tensor& bias,  // Bias tensor [N]
//     const at::Tensor& topk_weights,  // Top-k expert weights for each token
//     const at::Tensor& topk_ids,      // Top-k expert indices for each token
//     const at::Tensor&
//         sorted_token_ids,  // Sorted token indices according to allocated
//         expert
//     const at::Tensor& experts_ids,  // Assigned expert index for each block
//     const at::Tensor&
//         num_tokens_post_pad,           // Number of tokens after padding [1]
//     const at::Tensor& real_token_num,  // Actual number of valid tokens [1]
//     bool mul_routed_weight,            // Flag for topk_weights participation
//     int64_t topk,                      // Number of experts for each token
//     int64_t block_size                 // Block size
// ) {
//   // Get input shapes - match CPU implementation exactly
//   auto a_sizes = a.sizes();
//   auto b_sizes = b.sizes();

//   int64_t a_numel = a.numel();
//   int64_t k = a_sizes[a_sizes.size() - 1];  // Last dimension is K
//   int64_t m = a_numel / k;                  // Total number of tokens
//   int64_t e = b_sizes[0];                   // Number of experts
//   int64_t n = b_sizes[1];                   // Output feature dimension

//   // Calculate token_num from topk_weights shape
//   auto topk_weights_sizes = topk_weights.sizes();
//   int64_t token_num = topk_weights_sizes[0];
//   int64_t actual_topk = topk_weights_sizes[1];

//   // Validate inputs
//   if (k == 0 || e == 0 || n == 0 || token_num == 0 || actual_topk == 0) {
//     return;
//   }

//   // Get tensors on same device and contiguous - avoid unnecessary device
//   // transfers
//   auto a_cont = a.contiguous().to(torch::kFloat32).view({m, k});
//   auto b_cont = b.contiguous().to(torch::kFloat32);
//   auto scale_cont = scale.contiguous().to(torch::kFloat32);
//   auto topk_weights_cont = topk_weights.contiguous().to(torch::kFloat32);
//   auto sorted_token_ids_cont =
//   sorted_token_ids.contiguous().to(torch::kInt32); auto experts_ids_cont =
//   experts_ids.contiguous().to(torch::kInt32);

//   // Handle optional tensors
//   at::Tensor deq_zeros_cont;
//   if (deq_zeros.defined() && deq_zeros.numel() > 0) {
//     deq_zeros_cont = deq_zeros.contiguous().to(torch::kFloat32);
//   }

//   at::Tensor bias_cont;
//   if (bias.defined() && bias.numel() > 0) {
//     bias_cont = bias.contiguous().to(torch::kFloat32);
//   }

//   int32_t num_tokens_post_pad_val = num_tokens_post_pad.item<int32_t>();

//   // Ensure output tensor is contiguous and float32
//   auto c_cont = c.contiguous().to(torch::kFloat32);
//   c_cont.zero_();  // Zero out the output tensor

//   int64_t block_num = num_tokens_post_pad_val / block_size;

//   if (block_num <= 0) {
//     return;
//   }

//   // Pre-scan valid tokens in each block - exactly like CPU implementation
//   std::vector<int64_t> block_valid_offsets(block_num, 0);
//   int64_t offset = 0;

//   for (int64_t b = 0; b < block_num; b++) {
//     if (b >= experts_ids_cont.size(0) ||
//         experts_ids_cont[b].item<int32_t>() < 0) {
//       block_valid_offsets[b] = offset;
//       if (b < experts_ids_cont.size(0) &&
//           experts_ids_cont[b].item<int32_t>() < 0) {
//         offset += 1;
//       }
//       continue;
//     }

//     int64_t valid_count = 0;
//     for (int64_t i = 0; i < block_size; i++) {
//       int64_t index = b * block_size + i;
//       if (index >= sorted_token_ids_cont.size(0)) break;

//       int32_t sorted_id = sorted_token_ids_cont[index].item<int32_t>();
//       if (sorted_id >= token_num * actual_topk) break;
//       valid_count++;
//     }

//     block_valid_offsets[b] = offset;
//     offset += valid_count;
//   }

//   // Main computation loop - match CPU implementation exactly
//   for (int64_t block_id = 0;
//        block_id < block_num && block_id < experts_ids_cont.size(0);
//        block_id++) {
//     int32_t expert_id = experts_ids_cont[block_id].item<int32_t>();
//     if (expert_id < 0 || expert_id >= e) continue;

//     // Extract expert weights: B[expert_id, :, :] -> [N, K]
//     auto expert_weights = b_cont.select(0, expert_id);    // [N, K]
//     auto expert_scale = scale_cont.select(0, expert_id);  // [N]

//     // Extract expert zeros if available
//     at::Tensor expert_zeros;
//     if (deq_zeros_cont.defined()) {
//       expert_zeros = deq_zeros_cont.select(0, expert_id);  // [N] or [K/gs,
//       N]
//     }

//     for (int64_t token_id = 0; token_id < block_size; token_id++) {
//       int64_t sorted_idx = token_id + block_id * block_size;
//       if (sorted_idx >= sorted_token_ids_cont.size(0)) continue;

//       int32_t sorted_id = sorted_token_ids_cont[sorted_idx].item<int32_t>();
//       if (sorted_id < 0 || sorted_id >= token_num * actual_topk) continue;

//       // Calculate cal_id and store_id - exactly like CPU implementation
//       int64_t cal_id = mul_routed_weight
//                            ? (block_valid_offsets[block_id] + token_id)
//                            : (sorted_id / actual_topk);
//       int64_t store_token_id = mul_routed_weight
//                                    ? (sorted_id / actual_topk)
//                                    : (block_valid_offsets[block_id] +
//                                    token_id);
//       int64_t store_expert_id = sorted_id % actual_topk;

//       if (cal_id >= m || store_token_id >= token_num) continue;

//       // Get input token using cal_id
//       auto input_token = a_cont.select(0, cal_id);  // [K]

//       // Perform matrix multiplication using torch.matmul: expert_weights @
//       // input_token -> [N]
//       auto result =
//           torch::matmul(expert_weights, input_token);  // [N, K] x [K] -> [N]

//       // Apply dequantization: result * scale
//       result = result * expert_scale;

//       // Apply zeros if available
//       if (expert_zeros.defined()) {
//         if (expert_zeros.dim() == 1) {  // [N] shape
//           result = result - expert_zeros;
//         } else if (expert_zeros.dim() == 2) {  // [K/gs, N] shape - grouped
//                                                // dequant
//           // For grouped dequantization, apply zeros per group
//           int64_t groups = expert_zeros.size(0);
//           for (int64_t g = 0; g < groups; g++) {
//             auto group_zero = expert_zeros.select(0, g);  // [N]
//             result = result - group_zero;
//           }
//         }
//       }

//       // Add bias if available (bias is shared across experts)
//       if (bias_cont.defined()) {
//         result = result + bias_cont;
//       }

//       // Store result in output tensor using store_token_id and
//       store_expert_id if (store_token_id < c_cont.size(0) && store_expert_id
//       < c_cont.size(1)) {
//         c_cont.select(0, store_token_id)
//             .select(0, store_expert_id)
//             .copy_(result);
//       }

//       // Apply routed weight if needed
//       if (mul_routed_weight) {
//         int64_t token_idx = sorted_id / actual_topk;
//         int64_t expert_idx = sorted_id % actual_topk;
//         int64_t weight_idx = token_idx * actual_topk + expert_idx;
//         if (weight_idx < topk_weights_cont.numel()) {
//           auto weight = topk_weights_cont.view({-1})[weight_idx];
//           // Apply weight to the stored result
//           auto stored_result =
//               c_cont.select(0, store_token_id).select(0, store_expert_id);
//           stored_result.mul_(weight);
//         }
//       }
//     }
//   }

//   // Copy result back to original C tensor if needed
//   if (!c.is_same(c_cont)) {
//     c.copy_(c_cont.to(c.dtype()));
//   }
// }

// // Second algorithm: W8A8 quantization version
// inline void vllmInvokeFusedMoeNonGatherQuantKernel(
//     at::Tensor& c,        // Output tensor C [M, topk, N] - inplace
//     modification const at::Tensor& a,  // Input tensor A [*, K] (fp16/bf16)
//     const at::Tensor& b,  // Weight tensor B [E, N, K] (int8)
//     const at::Tensor& a_scale,       // Scale for A [E, K] or [M, 1] or [1,
//     1] const at::Tensor& scale,         // Scale for w8a8 quant [E, N] (fp32)
//     const at::Tensor& bias,          // Bias tensor [E, N] (fp32)
//     const at::Tensor& topk_weights,  // Top-k expert weights (fp32)
//     const at::Tensor& topk_ids,      // Top-k expert indices (int32)
//     const at::Tensor& sorted_token_ids,  // Sorted token indices (int32)
//     const at::Tensor& experts_ids,       // Expert index for each block
//     (int32) const at::Tensor&
//         num_tokens_post_pad,           // Number of tokens after padding [1]
//     const at::Tensor& real_token_num,  // Actual number of valid tokens [1]
//     bool mul_routed_weight,            // Flag for topk_weights participation
//     int64_t topk,                      // Number of experts for each token
//     int64_t block_size                 // Block size
// ) {
//   // Get input shapes - match CPU implementation exactly
//   auto a_sizes = a.sizes();
//   auto b_sizes = b.sizes();

//   int64_t a_numel = a.numel();
//   int64_t k = a_sizes[a_sizes.size() - 1];  // Last dimension is K
//   int64_t m = a_numel / k;                  // Total number of tokens
//   int64_t e = b_sizes[0];                   // Number of experts
//   int64_t n = b_sizes[1];                   // Output feature dimension

//   // Calculate token_num from topk_weights shape
//   auto topk_weights_sizes = topk_weights.sizes();
//   int64_t token_num = topk_weights_sizes[0];
//   int64_t actual_topk = topk_weights_sizes[1];

//   // Validate inputs
//   if (k == 0 || e == 0 || n == 0 || token_num == 0 || actual_topk == 0) {
//     return;
//   }

//   // Get tensors on same device and contiguous - avoid unnecessary device
//   // transfers
//   auto a_cont = a.contiguous().to(torch::kFloat32).view({m, k});
//   auto b_cont = b.contiguous().to(torch::kFloat32);
//   auto a_scale_cont = a_scale.contiguous().to(torch::kFloat32);
//   auto scale_cont = scale.contiguous().to(torch::kFloat32);
//   auto topk_weights_cont = topk_weights.contiguous().to(torch::kFloat32);
//   auto sorted_token_ids_cont =
//   sorted_token_ids.contiguous().to(torch::kInt32); auto experts_ids_cont =
//   experts_ids.contiguous().to(torch::kInt32);

//   // Handle optional tensors
//   at::Tensor bias_cont;
//   if (bias.defined() && bias.numel() > 0) {
//     bias_cont = bias.contiguous().to(torch::kFloat32);
//   }

//   int32_t num_tokens_post_pad_val = num_tokens_post_pad.item<int32_t>();

//   // Ensure output tensor is contiguous and float32
//   auto c_cont = c.contiguous().to(torch::kFloat32);
//   c_cont.zero_();  // Zero out the output tensor

//   int64_t block_num = num_tokens_post_pad_val / block_size;

//   if (block_num <= 0) {
//     return;
//   }

//   // Pre-scan valid tokens in each block - exactly like CPU implementation
//   std::vector<int64_t> block_valid_offsets(block_num, 0);
//   int64_t offset = 0;

//   for (int64_t b = 0; b < block_num; b++) {
//     if (b >= experts_ids_cont.size(0) ||
//         experts_ids_cont[b].item<int32_t>() < 0) {
//       block_valid_offsets[b] = offset;
//       if (b < experts_ids_cont.size(0) &&
//           experts_ids_cont[b].item<int32_t>() < 0) {
//         offset += 1;
//       }
//       continue;
//     }

//     int64_t valid_count = 0;
//     for (int64_t i = 0; i < block_size; i++) {
//       int64_t index = b * block_size + i;
//       if (index >= sorted_token_ids_cont.size(0)) break;

//       int32_t sorted_id = sorted_token_ids_cont[index].item<int32_t>();
//       if (sorted_id >= token_num * actual_topk) break;
//       valid_count++;
//     }

//     block_valid_offsets[b] = offset;
//     offset += valid_count;
//   }

//   // Determine AScale mode
//   bool is_per_expert_channel =
//       (a_scale_cont.dim() == 2 && a_scale_cont.size(0) == e);
//   bool is_per_token = (a_scale_cont.dim() == 2 && a_scale_cont.size(0) == m);

//   // Main computation loop - match CPU implementation exactly
//   for (int64_t block_id = 0;
//        block_id < block_num && block_id < experts_ids_cont.size(0);
//        block_id++) {
//     int32_t expert_id = experts_ids_cont[block_id].item<int32_t>();
//     if (expert_id < 0 || expert_id >= e) continue;

//     // Extract expert weights and scales
//     auto expert_weights = b_cont.select(0, expert_id);      // [N, K]
//     auto expert_w_scale = scale_cont.select(0, expert_id);  // [N]

//     // Extract expert bias if available
//     at::Tensor expert_bias;
//     if (bias_cont.defined()) {
//       expert_bias = bias_cont.select(0, expert_id);  // [N]
//     }

//     // Get A scale for this expert
//     at::Tensor expert_a_scale;
//     if (is_per_expert_channel) {
//       expert_a_scale = a_scale_cont.select(0, expert_id);  // [K]
//     } else if (is_per_token) {
//       // Will be handled per token
//     } else {
//       expert_a_scale =
//           a_scale_cont.view({-1}).slice(0, 0, 1);  // scalar as tensor
//     }

//     for (int64_t token_id = 0; token_id < block_size; token_id++) {
//       int64_t sorted_idx = token_id + block_id * block_size;
//       if (sorted_idx >= sorted_token_ids_cont.size(0)) continue;

//       int32_t sorted_id = sorted_token_ids_cont[sorted_idx].item<int32_t>();
//       if (sorted_id < 0 || sorted_id >= token_num * actual_topk) continue;

//       // Calculate cal_id and store_id - exactly like CPU implementation
//       int64_t cal_id = mul_routed_weight
//                            ? (block_valid_offsets[block_id] + token_id)
//                            : (sorted_id / actual_topk);
//       int64_t store_token_id = mul_routed_weight
//                                    ? (sorted_id / actual_topk)
//                                    : (block_valid_offsets[block_id] +
//                                    token_id);
//       int64_t store_expert_id = sorted_id % actual_topk;

//       if (cal_id >= m || store_token_id >= token_num) continue;

//       // Get input token using cal_id
//       auto input_token = a_cont.select(0, cal_id);  // [K]

//       // Get A scale for this token
//       at::Tensor token_a_scale;
//       if (is_per_token) {
//         token_a_scale = a_scale_cont.select(0, cal_id);
//       } else if (is_per_expert_channel) {
//         token_a_scale = expert_a_scale;
//       } else {
//         token_a_scale = expert_a_scale;
//       }

//       // For W8A8 quantization, we need to simulate the CPU's approach more
//       // closely
//       at::Tensor result;
//       if (is_per_expert_channel) {
//         // Per-channel A scale: CPU版本有特殊处理，直接计算而不做int8量化
//         result = torch::zeros({n}, input_token.options());
//         // Get original B in int8 format for this calculation
//         auto B_int8_cont = B.contiguous().to(torch::kInt8);
//         auto expert_weights_int8 =
//             B_int8_cont.select(0, expert_id);  // [N, K] int8

//         for (int64_t n = 0; n < N; n++) {
//           float scaled_sum = 0.0f;
//           for (int64_t k = 0; k < K; k++) {
//             float a_val = input_token[k].item<float>();
//             float w_val = static_cast<float>(
//                 expert_weights_int8.select(0, n)[k].item<int8_t>());
//             float a_scale_val = token_a_scale[k].item<float>();
//             scaled_sum += a_val * w_val * a_scale_val;
//           }
//           result[n] = scaled_sum;
//         }
//       } else {
//         // Scalar or per-token A scale: 模拟真正的int8量化过程
//         auto B_int8_cont = B.contiguous().to(torch::kInt8);
//         auto expert_weights_int8 =
//             B_int8_cont.select(0, expert_id);  // [N, K] int8

//         // 量化输入A到int8
//         at::Tensor input_quantized = torch::zeros({K}, torch::kInt8);
//         float a_scale_val = token_a_scale.numel() == 1
//                                 ? token_a_scale.item<float>()
//                                 : token_a_scale[0].item<float>();

//         for (int64_t k = 0; k < K; k++) {
//           float quantized_val = input_token[k].item<float>() / a_scale_val;
//           // 模拟fp32_to_int8_cpu函数
//           int32_t i32 = static_cast<int32_t>(std::round(quantized_val));
//           if (i32 > 127)
//             i32 = 127;
//           else if (i32 < -127)
//             i32 = -127;
//           input_quantized[k] = static_cast<int8_t>(i32);
//         }

//         // 执行int8矩阵乘法
//         result = torch::zeros({N}, torch::kFloat32);
//         for (int64_t n = 0; n < N; n++) {
//           int32_t sum = 0;
//           for (int64_t k = 0; k < K; k++) {
//             sum += static_cast<int32_t>(input_quantized[k].item<int8_t>()) *
//                    static_cast<int32_t>(
//                        expert_weights_int8.select(0, n)[k].item<int8_t>());
//           }

//           // 反量化：result * a_scale
//           float dequant_result = static_cast<float>(sum) * a_scale_val;
//           result[n] = dequant_result;
//         }
//       }

//       // Apply weight scale
//       result = result * expert_w_scale;

//       // Add bias if available
//       if (expert_bias.defined()) {
//         result = result + expert_bias;
//       }

//       // Store result in output tensor using store_token_id and
//       store_expert_id if (store_token_id < c_cont.size(0) && store_expert_id
//       < c_cont.size(1)) {
//         c_cont.select(0, store_token_id)
//             .select(0, store_expert_id)
//             .copy_(result);
//       }

//       // Apply routed weight if needed
//       if (mul_routed_weight) {
//         int64_t token_idx = sorted_id / actual_topk;
//         int64_t expert_idx = sorted_id % actual_topk;
//         int64_t weight_idx = token_idx * actual_topk + expert_idx;
//         if (weight_idx < topk_weights_cont.numel()) {
//           auto weight = topk_weights_cont.view({-1})[weight_idx];
//           // Apply weight to the stored result
//           auto stored_result =
//               c_cont.select(0, store_token_id).select(0, store_expert_id);
//           stored_result.mul_(weight);
//         }
//       }
//     }
//   }

//   // Copy result back to original C tensor if needed
//   if (!c.is_same(c_cont)) {
//     c.copy_(c_cont.to(c.dtype()));
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void extsSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
//                                           const at::Tensor& in,
//                                           const at::Tensor& size,
//                                           const int32_t group_size) {
//   // Get real size
//   int32_t real_size = size.item<int32_t>();

//   // Get tensor dimensions
//   auto sizes = in.sizes();

//   // Calculate dimensions following the same logic as CPU implementation
//   int dim0 = real_size;
//   int dim1;
//   int lowest_dim_index = sizes.size() - 1;

//   for (int i = 1; i < lowest_dim_index; i++) {
//     dim0 *= sizes[i];
//   }
//   dim1 = sizes[lowest_dim_index];

//   int out_dim1 = dim1 / 2;
//   int out_elem_num = dim0 * out_dim1;
//   int group_num = out_elem_num / group_size;

//   // Convert input to float32 for computation (like CPU implementation)
//   auto in_f32 = in.to(at::kFloat);

//   // Split input tensor into two halves along the last dimension
//   auto in1 = in_f32.narrow(-1, 0, out_dim1);         // First half
//   auto in2 = in_f32.narrow(-1, out_dim1, out_dim1);  // Second half

//   // Apply SiLU to first half: x * sigmoid(x)
//   auto silu_result = in1 * at::sigmoid(in1);

//   // Multiply with second half
//   auto silu_mul = silu_result * in2;

//   // Only process the real_size portion along the first dimension
//   auto silu_mul_real = silu_mul.narrow(0, 0, real_size);

//   // Flatten for group processing - this should match the CPU implementation
//   // exactly
//   auto silu_mul_flat = silu_mul_real.contiguous().view(-1);

//   // Verify we have the expected number of elements
//   if (silu_mul_flat.numel() != out_elem_num) {
//     throw std::runtime_error("Element count mismatch in native
//     implementation");
//   }

//   // Reshape to groups for scale computation
//   auto silu_mul_groups = silu_mul_flat.view({group_num, group_size});

//   // Compute absolute values and find max per group
//   auto abs_values = at::abs(silu_mul_groups);
//   auto group_max = std::get<0>(at::max(abs_values, /*dim=*/1));

//   // Determine FP8 max value based on output tensor type
//   float fp8_max;
//   if (out.scalar_type() == at::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;
//   } else {
//     fp8_max = 57344.0f;  // e5m2
//   }

//   // Compute scales
//   auto scales = group_max / fp8_max;
//   scale.copy_(scales);

//   // Apply quantization: divide by scale
//   auto scales_expanded = scales.unsqueeze(1).expand({group_num, group_size});
//   auto quantized = silu_mul_groups / scales_expanded;

//   // Flatten back to 1D
//   auto quantized_flat = quantized.view(-1);

//   // Convert to fp8 - this is the critical step
//   // We need to ensure the conversion happens correctly
//   auto quantized_fp8 = quantized_flat.to(out.scalar_type());

//   // Copy to output tensor - ensure we're copying to the right location
//   auto out_real = out.narrow(0, 0, real_size);
//   auto out_flat = out_real.contiguous().view(-1);
//   out_flat.copy_(quantized_fp8);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// namespace detail {

// // Helper function to perform per-token group quantization using torch
// // operations
// inline std::tuple<at::Tensor, at::Tensor> perTokenGroupQuantizeFP8Native(
//     const at::Tensor& input, int32_t group_size) {
//   auto input_sizes = input.sizes();
//   int64_t n = input_sizes[0];
//   int64_t d = input_sizes[1];

//   TORCH_CHECK(d % group_size == 0,
//               "Hidden dimension must be divisible by group_size");

//   int64_t group_num = d / group_size;

//   // Reshape to [n, group_num, group_size] for per-group processing
//   auto reshaped = input.view({n, group_num, group_size});

//   // Find max absolute value per group: [n, group_num, 1]
//   auto abs_values = at::abs(reshaped);
//   auto group_max =
//       std::get<0>(at::max(abs_values, /*dim=*/-1, /*keepdim=*/true));

//   // Determine FP8 max based on output type - for now assume E4M3
//   constexpr float E4M3_MAX = 448.0f;
//   constexpr float E5M2_MAX = 57344.0f;
//   float fp8_max = E4M3_MAX;  // Default to E4M3

//   // Compute scales: scale = group_max / fp8_max
//   auto scales = group_max / fp8_max;

//   // Avoid division by zero
//   auto safe_scales = at::where(scales > 0, scales, at::ones_like(scales));

//   // Quantize: quantized = input / scale
//   auto quantized = reshaped / safe_scales;

//   // Reshape back to original shape
//   auto output = quantized.view(input_sizes);

//   // Return both quantized values and scales
//   // Note: scales shape is [n, group_num]
//   return std::make_tuple(output, scales.squeeze(-1));
// }

// }  // namespace detail

// /**
//  * @brief Fused Add and RMS Normalization Dyn Per Token Group Quantize FP8
//  * Native implementation.
//  *
//  * @param output          Output tensor of shape [N, hidden_size].
//  * @param residual_update Update residual tensor of shape [N, hidden_size].
//  * @param scale           Scale tensor of shape [N, hidden_size /
//  group_size].
//  * @param input           Input tensor of shape [N, hidden_size].
//  * @param residual        Residual tensor of shape [N, hidden_size].
//  * @param weight          Weight tensor of shape [hidden_size].
//  * @param epsilon         Value added to denominator for numerical stability.
//  * @param group_size      The group size used for quantization.
//  */
// inline void vllmFusedAddRmsNormPerTokenGroupQuantFp8(
//     at::Tensor& output, at::Tensor& residual_update, at::Tensor& scale,
//     const at::Tensor& input, const at::Tensor& residual,
//     const at::Tensor& weight, float epsilon, int32_t group_size) {
//   // Convert all inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto residual_f32 =
//       residual.numel() > 0 ? residual.to(at::kFloat) : at::Tensor();
//   auto weight_f32 = weight.numel() > 0 ? weight.to(at::kFloat) :
//   at::Tensor();

//   // Step 1: Fused Add - Add input and residual
//   at::Tensor x;
//   at::Tensor residual_update_computed;
//   if (residual_f32.numel() > 0) {
//     x = input_f32 + residual_f32;
//     residual_update_computed = x.clone();  // Store the updated residual
//   } else {
//     x = input_f32;
//     residual_update_computed = at::Tensor();  // Empty tensor
//   }

//   // Step 2: RMS Normalization
//   // variance = mean(x^2, dim=-1, keepdim=True)
//   auto variance = at::mean(x * x, /*dim=*/-1, /*keepdim=*/true);

//   // Apply RMS normalization: x = x * rsqrt(variance + epsilon)
//   x = x * at::rsqrt(variance + epsilon);

//   // Step 3: Apply weight scaling if provided
//   if (weight_f32.numel() > 0) {
//     x = x * weight_f32;
//   }

//   // Step 4: Dynamic Per-Token Group Quantization FP8
//   auto input_sizes = x.sizes();
//   int64_t n = 1;
//   for (int i = 0; i < input_sizes.size() - 1; ++i) {
//     n *= input_sizes[i];
//   }
//   int64_t d = input_sizes[input_sizes.size() - 1];

//   TORCH_CHECK(d % group_size == 0,
//               "Hidden dimension must be divisible by group_size");

//   int64_t group_num = d / group_size;

//   // Reshape to [n, group_num, group_size] for per-group processing
//   auto x_reshaped = x.view({n, group_num, group_size});

//   // Find max absolute value per group: [n, group_num, 1]
//   auto abs_values = at::abs(x_reshaped);
//   auto group_max =
//       std::get<0>(at::max(abs_values, /*dim=*/-1, /*keepdim=*/true));

//   // Determine FP8 max values
//   constexpr float E4M3_MAX = 448.0f;
//   constexpr float E5M2_MAX = 57344.0f;
//   float fp8_max = E4M3_MAX;  // Default to E4M3 for now

//   // Compute scales: scale = group_max / fp8_max
//   auto scales = group_max / fp8_max;

//   // Avoid division by zero
//   auto safe_scales = at::where(scales > 0, scales, at::ones_like(scales));

//   // Quantize: quantized = x / scale
//   auto quantized = x_reshaped / safe_scales;

//   // Reshape back to original shape
//   auto quantized_output = quantized.view(input_sizes);

//   // Reshape scales to [n, group_num]
//   auto output_scales = scales.squeeze(-1);

//   // Copy results to output tensors
//   output.copy_(quantized_output.to(output.scalar_type()));

//   // Copy residual update if provided
//   if (residual_update_computed.numel() > 0 && residual_update.numel() > 0) {
//     residual_update.copy_(
//         residual_update_computed.to(residual_update.scalar_type()));
//   }

//   // Copy scales
//   scale.copy_(output_scales.to(scale.scalar_type()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmDynamicPerTokenGroupFP8Quant(at::Tensor& output,
//                                              at::Tensor& scale,
//                                              const at::Tensor& input,
//                                              const int32_t group_size) {
//   // Get tensor dimensions
//   int ele_num = input.numel();
//   int group_num = ele_num / group_size;

//   // Validate that the input can be evenly divided into groups
//   if (ele_num % group_size != 0) {
//     throw std::runtime_error(
//         "Input tensor size must be divisible by group_size");
//   }

//   // Validate scale tensor size
//   if (scale.numel() != group_num) {
//     throw std::runtime_error("Scale tensor size mismatch");
//   }

//   // Convert input to float32 for computation
//   auto input_f32 = input.to(at::kFloat);

//   // Flatten for group processing
//   auto input_flat = input_f32.contiguous().view(-1);

//   // Reshape to groups for scale computation - ensure correct dimensions
//   auto input_groups = input_flat.view({group_num, group_size});

//   // Compute absolute values and find max per group
//   auto abs_values = at::abs(input_groups);
//   auto group_max = std::get<0>(at::max(abs_values, /*dim=*/1));

//   // Apply epsilon to avoid division by zero
//   const float eps = 1e-10f;
//   group_max = at::clamp_min(group_max, eps);

//   // Determine FP8 max value based on output tensor type
//   float fp8_max;
//   if (output.scalar_type() == at::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;  // E4M3_MAX
//   } else {
//     fp8_max = 57344.0f;  // E5M2_MAX for e5m2
//   }

//   // Compute scales
//   auto scales = group_max / fp8_max;

//   // Flatten scale tensor for assignment, then reshape back
//   auto scale_flat = scale.contiguous().view(-1);
//   scale_flat.copy_(scales);

//   // Apply quantization: divide by scale
//   // Use repeat_interleave instead of expand for more reliable broadcasting
//   auto scales_repeated = scales.repeat_interleave(group_size);
//   auto quantized_flat = input_flat / scales_repeated;

//   // Convert to fp8 output type
//   auto quantized_fp8 = quantized_flat.to(output.scalar_type());
//   if (output.numel() != quantized_fp8.numel()) {
//     throw std::runtime_error("Output tensor size mismatch");
//   }
//   output.copy_(quantized_fp8.view_as(output));
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// #include <vector>

// inline void extsDynamicSplit(std::vector<at::Tensor>& outputs,
//                              const at::Tensor& input,
//                              const at::Tensor& input_size,
//                              const std::vector<int64_t>& split_size,
//                              const int64_t dim) {
//   // Parameter validation
//   TORCH_CHECK(input.numel() > 0, "Input tensor cannot be empty");
//   TORCH_CHECK(!split_size.empty(), "Split sizes cannot be empty");

//   // Normalize dimension
//   int64_t normalized_dim = dim < 0 ? input.dim() + dim : dim;
//   TORCH_CHECK(normalized_dim >= 0 && normalized_dim < input.dim(),
//               "Invalid dimension");

//   // Get input size value
//   TORCH_CHECK(input_size.numel() == 1, "Input size must be a scalar tensor");
//   int64_t update_size = input_size.item<int64_t>();

//   // Validate split sizes sum
//   int64_t total_split_size = 0;
//   for (auto size : split_size) {
//     total_split_size += size;
//   }
//   TORCH_CHECK(total_split_size <= input.size(normalized_dim),
//               "Sum of split sizes exceeds input dimension size");

//   // Clear outputs and reserve space
//   outputs.clear();
//   outputs.reserve(split_size.size());

//   // Calculate split offsets
//   std::vector<int64_t> split_offsets(split_size.size());
//   split_offsets[0] = 0;
//   for (size_t i = 1; i < split_size.size(); ++i) {
//     split_offsets[i] = split_offsets[i - 1] + split_size[i - 1];
//   }

//   // Perform splits using torch.split
//   at::IntArrayRef split_size_ref(split_size);
//   auto split_tensors = at::split(input, split_size_ref, normalized_dim);

//   // Apply input_size constraint to each split
//   for (size_t i = 0; i < split_tensors.size(); ++i) {
//     auto output = split_tensors[i].clone();

//     // Apply input_size constraint: only keep first update_size elements
//     along
//     // dim 0
//     if (update_size < input.size(0)) {
//       std::vector<at::indexing::TensorIndex> indices(output.dim(),
//                                                      at::indexing::Slice());
//       indices[0] = at::indexing::Slice(0, update_size);
//       output = output.index(indices).contiguous();
//     }

//     outputs.push_back(output);
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// #include <vector>

// // PyTorch实现的get_ep_indices
// inline void vllmGetEPIndices(at::Tensor& ep_count, at::Tensor&
// ep_token_indices,
//                              at::Tensor& ep_valid_token_indices,
//                              const at::Tensor& topk_ids, int expert_per_rank,
//                              int ep_size) {
//   // 参数检查
//   TORCH_CHECK(topk_ids.dim() == 2, "topk_ids should be 2D tensor");
//   TORCH_CHECK(ep_count.dim() == 1 && ep_count.size(0) >= ep_size,
//               "ep_count should be 1D tensor with size >= ep_size");

//   // 生成一个索引数组 [0, 1, 2, ..., topk_ids.size(0)-1]
//   auto idx =
//       torch::arange(0, topk_ids.size(0),
//       topk_ids.options().dtype(torch::kInt));

//   // 创建存储每个ep的count和indices的容器
//   std::vector<at::Tensor> idx_list;
//   std::vector<at::Tensor> count_list;

//   // 对每个ep进行处理
//   for (int i = 0; i < ep_size; ++i) {
//     // 创建掩码：找出topk_ids中属于当前ep范围的专家ID
//     // mask表示每个token是否选择了当前ep范围内的专家
//     auto mask = torch::logical_and(topk_ids >= (expert_per_rank * i),
//                                    topk_ids < (expert_per_rank * (i + 1)));
//     // 沿着dim=1汇总，表示如果token的任何一个topk选择了当前ep的专家，则为true
//     mask = mask.sum(1) > 0;

//     // 计算当前ep中的token数量
//     count_list.push_back(mask.sum(0, /*keepdim=*/true));

//     // 选择符合条件的token索引
//     idx_list.push_back(idx.masked_select(mask));
//   }

//   // 合并所有ep的count和token索引
//   auto ep_count_result = torch::cat(count_list);
//   auto ep_token_indices_result = torch::cat(idx_list);

//   // 复制到输出tensor
//   ep_count.copy_(ep_count_result);

//   // 部分复制（确保不越界）
//   auto size =
//       std::min(ep_token_indices_result.numel(), ep_token_indices.numel());
//   if (size > 0) {
//     ep_token_indices.slice(0, 0, size)
//         .copy_(ep_token_indices_result.slice(0, 0, size));
//   }

//   // 设置有效token的数量 - 使用安全的方式
//   if (ep_valid_token_indices.numel() > 0) {
//     // 创建一个值为ep_token_indices_result.numel()的标量张量
//     auto valid_count =
//         torch::tensor({ep_token_indices_result.numel()},
//                       ep_valid_token_indices.options().dtype(torch::kInt));
//     // 使用copy_代替直接访问内存
//     ep_valid_token_indices.copy_(valid_count);
//   }
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// inline void vllmRmsNorm(at::Tensor& output, const at::Tensor& input,
//                         const at::Tensor& gamma, const double epsilon) {
//   // Convert inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto gamma_f32 = gamma.to(at::kFloat);

//   // Compute variance: variance = mean(x * x, dim=-1, keepdim=True)
//   auto variance = at::mean(input_f32 * input_f32, /*dim=*/-1,
//   /*keepdim=*/true);

//   // Apply RMS normalization: x = x * rsqrt(variance + epsilon)
//   auto normalized = input_f32 * at::rsqrt(variance + epsilon);

//   // Apply gamma scaling: x = x * gamma
//   auto result = normalized * gamma_f32;

//   // Convert back to original dtype and copy to output tensor
//   output.copy_(result.to(input.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmConcatAndCacheMla(at::Tensor& kv_cache, const at::Tensor&
// kv_c,
//                                   const at::Tensor& k_pe,
//                                   const at::Tensor& slot_mapping,
//                                   const char* kv_cache_dtype,
//                                   const at::Tensor& scale) {
//   // Get tensor properties
//   auto num_tokens = kv_c.size(0);
//   auto kv_lora_rank = kv_c.size(1);
//   auto pe_dim = k_pe.size(1);
//   auto num_blocks = kv_cache.size(0);
//   auto block_size = kv_cache.size(1);
//   auto entry_stride = kv_cache.size(2);

//   // Validate dimensions
//   TORCH_CHECK(k_pe.size(0) == num_tokens,
//               "k_pe and kv_c must have same number of tokens");
//   TORCH_CHECK(entry_stride == kv_lora_rank + pe_dim,
//               "kv_cache entry_stride must equal kv_lora_rank + pe_dim");
//   TORCH_CHECK(slot_mapping.size(0) == num_tokens,
//               "slot_mapping must have same length as number of tokens");

//   // Ensure tensors are contiguous for efficient processing
//   auto kv_c_contiguous = kv_c.contiguous();
//   auto k_pe_contiguous = k_pe.contiguous();
//   auto slot_mapping_contiguous = slot_mapping.contiguous();

//   // Use torch operations to concatenate kv_c and k_pe
//   // Shape: [num_tokens, kv_lora_rank + pe_dim]
//   auto concatenated = at::cat({kv_c_contiguous, k_pe_contiguous}, /*dim=*/1);

//   // Process each token using torch tensor operations
//   for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++) {
//     int slot_idx = slot_mapping_contiguous[token_idx].item<int>();

//     if (slot_idx < 0) {
//       continue;
//     }

//     int64_t block_idx = slot_idx / block_size;
//     int64_t block_offset = slot_idx % block_size;

//     // Validate block index
//     TORCH_CHECK(block_idx < num_blocks,
//                 "Block index out of bounds: ", block_idx, " >= ",
//                 num_blocks);

//     // Use torch tensor copy instead of memcpy to support CUDA tensors
//     // Copy concatenated[token_idx] to kv_cache[block_idx, block_offset]
//     kv_cache[block_idx][block_offset].copy_(concatenated[token_idx]);
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// // Native implementation using torch operations
// // Parameters follow the interface definition: q, kv, x, weight, x_scale,
// // weight_scale, group_size
// inline void extsFusedQKVProj(at::Tensor& q, at::Tensor& kv, const at::Tensor&
// x,
//                              const at::Tensor& weight,
//                              const at::Tensor& x_scale,
//                              const at::Tensor& weight_scale,
//                              int64_t group_size) {
//   // Convert inputs to float32 for computation to match CPU implementation
//   // precision
//   auto x_f32 = x.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);
//   auto x_scale_f32 = x_scale.to(at::kFloat);
//   auto weight_scale_f32 = weight_scale.to(at::kFloat);

//   // Get dimensions
//   // x: [M, K], weight: [N, K], q: [M, N_0], kv: [M, N_1]
//   auto x_sizes = x_f32.sizes();
//   int64_t m = 1;
//   for (int i = 0; i < x_sizes.size() - 1; ++i) {
//     m *= x_sizes[i];
//   }
//   int64_t k = x_sizes[x_sizes.size() - 1];

//   // Get N_0 and N_1 from output tensor dimensions
//   int64_t n_0 = q.size(-1);
//   int64_t n_1 = kv.size(-1);
//   int64_t n = n_0 + n_1;

//   // Reshape input to [M, K], weight should be [N, K]
//   auto x_2d = x_f32.view({m, k});

//   // Apply group quantization scaling
//   int64_t k_group_num = (k + group_size - 1) / group_size;
//   int64_t n_group_num = (n + group_size - 1) / group_size;

//   // Create output tensors with the correct shape
//   std::vector<int64_t> q_shape(x_sizes.begin(), x_sizes.end() - 1);
//   q_shape.push_back(n_0);
//   std::vector<int64_t> kv_shape(x_sizes.begin(), x_sizes.end() - 1);
//   kv_shape.push_back(n_1);

//   auto result_q = torch::zeros(q_shape, at::kFloat);
//   auto result_kv = torch::zeros(kv_shape, at::kFloat);

//   // Flatten result tensors to [M, N_0] and [M, N_1] for easier computation
//   auto result_q_2d = result_q.view({m, n_0});
//   auto result_kv_2d = result_kv.view({m, n_1});

//   // Flatten scale tensors for direct indexing
//   auto x_scale_flat = x_scale_f32.view(-1);
//   auto weight_scale_flat = weight_scale_f32.view(-1);

//   // Main computation loop following the exact same logic as CPU
//   implementation
//   // This ensures identical numerical results
//   for (int64_t m_idx = 0; m_idx < m; m_idx++) {
//     for (int64_t n_idx = 0; n_idx < n; n_idx++) {
//       float tmp = 0.0f;

//       // Inner loop for K dimension - exactly as in CPU implementation
//       for (int64_t k_idx = 0; k_idx < k; k_idx++) {
//         // Calculate indices exactly as in CPU implementation
//         int64_t x_idx = m_idx * k + k_idx;       // x index
//         int64_t weight_idx = n_idx * k + k_idx;  // weight index
//         int64_t ks_idx = k_idx / group_size;
//         int64_t ns_idx = n_idx / group_size;
//         int64_t x_scale_idx = m_idx * k_group_num + ks_idx;  // x_scale index
//         int64_t weight_scale_idx =
//             ns_idx * k_group_num + ks_idx;  // weight_scale index

//         // Compute exactly as in CPU implementation:
//         // tmp = tmp + (x[x_idx] * x_scale[x_scale_idx]) *
//         (weight[weight_idx]
//         // * weight_scale[weight_scale_idx])
//         float x_val = x_2d.view(-1)[x_idx].item<float>();
//         float weight_val = weight_f32.view(-1)[weight_idx].item<float>();
//         float x_scale_val = x_scale_flat[x_scale_idx].item<float>();
//         float weight_scale_val =
//             weight_scale_flat[weight_scale_idx].item<float>();

//         tmp = tmp + (x_val * x_scale_val) * (weight_val * weight_scale_val);
//       }

//       // Store result exactly as in CPU implementation
//       if (n_idx < n_0) {
//         result_q_2d[m_idx][n_idx] = tmp;
//       } else {
//         result_kv_2d[m_idx][n_idx - n_0] = tmp;
//       }
//     }
//   }

//   // Convert back to original dtype and copy to output tensors
//   q.copy_(result_q.to(x.dtype()));
//   kv.copy_(result_kv.to(x.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// /**
//  * @brief This function performs a sum reduction computation.
//  *
//  * @param output: The output tensor.
//  * @param input: The input tensor.
//  * @param real_bs: The real batch size tensor.
//  * @param dimensions: The dimensions to reduce sum.
//  * @param keepdims: Whether the output tensor has dim retained or not.
//  * @param data_type: Data type.
//  * @param stream: Optional stream parameter (not used in torch
//  implementation).
//  */
// inline void extsSum(at::Tensor& output, const at::Tensor& input,
//                     const at::Tensor& real_bs,
//                     const at::IntArrayRef& dimensions, const bool keepdims,
//                     const at::ScalarType& data_type, void* stream = nullptr)
//                     {
//   // Check input dimensions
//   if (input.dim() != 3) {
//     throw std::runtime_error("Input tensor must be 3-dimensional (N, R, C)");
//   }

//   // Extract real batch size value
//   int64_t real_bs_value = real_bs.item<int64_t>();

//   // Get input dimensions
//   auto sizes = input.sizes();
//   int64_t n = std::min(real_bs_value, sizes[0]);  // Use real batch size
//   int64_t r = sizes[1];                           // Reduce dimension
//   int64_t c = sizes[2];                           // Feature dimension

//   // Ensure input is in the specified data type
//   auto input_typed = input.to(data_type);

//   // Perform sum along the specified dimensions (default dim=1 for R
//   dimension) int64_t reduce_dim = dimensions.size() > 0 ? dimensions[0] : 1;

//   // Take only the real batch size portion of input
//   auto input_real_bs = input_typed.slice(0, 0, n);

//   // Perform sum reduction
//   auto result = at::sum(input_real_bs, reduce_dim, keepdims);

//   // Copy result to output tensor
//   output.copy_(result.to(data_type));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// #include <vector>

// // PyTorch实现的ets_moe_align_block_size
// inline void extsMoeAlignBlockSize(at::Tensor& sorted_topk_ids,
//                                   at::Tensor& expert_ids,
//                                   at::Tensor& num_tokens_post_pad,
//                                   const at::Tensor& topk_ids,
//                                   const at::Tensor& real_token_num,
//                                   const at::Tensor& expert_map,
//                                   int64_t num_experts, int64_t block_size) {
//   // 参数检查
//   TORCH_CHECK(topk_ids.dim() == 2, "topk_ids should be 2D tensor");
//   if (expert_map.defined() && expert_map.numel() > 0) {
//     TORCH_CHECK(expert_map.dim() == 1 && expert_map.size(0) >= num_experts,
//                 "expert_map should be 1D tensor with size >= num_experts");
//   }

//   // 获取token数量和topk值
//   int token_num = topk_ids.size(0);
//   int topk = topk_ids.size(1);

//   // 如果提供了real_token_num，则使用它
//   if (real_token_num.defined() && real_token_num.numel() > 0) {
//     token_num = real_token_num.item<int>();
//   }

//   // 计算每个expert处理的token数量
//   auto expert_token_cnt = torch::zeros({num_experts}, topk_ids.options());

//   // 使用PyTorch操作计算每个expert的token数
//   for (int i = 0; i < token_num; i++) {
//     for (int j = 0; j < topk; j++) {
//       int expert = topk_ids[i][j].item<int>();
//       expert_token_cnt[expert] += 1;
//     }
//   }

//   // 计算每个expert对齐block_size后的累积和
//   auto padded_cumsum = torch::zeros({num_experts + 1}, topk_ids.options());
//   for (int i = 0; i < num_experts; i++) {
//     int expert_tokens = expert_token_cnt[i].item<int>();
//     int padded_tokens =
//         ((expert_tokens + block_size - 1) / block_size) * block_size;
//     padded_cumsum[i + 1] = padded_cumsum[i].item<int>() + padded_tokens;
//   }

//   // 总token数（对齐后）
//   int total_tokens_padded = padded_cumsum[num_experts].item<int>();

//   // 设置输出tensor
//   if (num_tokens_post_pad.numel() > 0) {
//     num_tokens_post_pad.fill_(total_tokens_padded);
//   }

//   // 为每个block分配expert ID
//   auto expert_ids_tensor =
//       torch::zeros({total_tokens_padded / block_size}, topk_ids.options());
//   int expert = 0;
//   for (int i = 0; i < total_tokens_padded; i += block_size) {
//     while (i >= padded_cumsum[expert + 1].item<int>()) {
//       expert += 1;
//     }
//     // 应用expert映射
//     int mapped_expert = expert;
//     if (expert_map.defined() && expert_map.numel() > 0) {
//       mapped_expert = expert_map[expert].item<int>();
//     }
//     expert_ids_tensor[i / block_size] = mapped_expert;
//   }
//   expert_ids.copy_(expert_ids_tensor);

//   // 排序token IDs
//   auto sorted_ids =
//       torch::full({total_tokens_padded}, token_num * topk,
//       topk_ids.options());
//   auto expert_token_idx = torch::zeros({num_experts}, topk_ids.options());

//   int idx = 0;
//   for (int i = 0; i < token_num; i++) {
//     for (int j = 0; j < topk; j++) {
//       int expert = topk_ids[i][j].item<int>();
//       int pos = padded_cumsum[expert].item<int>() +
//                 expert_token_idx[expert].item<int>();
//       sorted_ids[pos] = idx;
//       expert_token_idx[expert] += 1;
//       idx += 1;
//     }
//   }

//   sorted_topk_ids.copy_(sorted_ids);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// /**
//  * @brief Moe align block size native implementation without real_token_num
//  * parameter
//  *
//  * @param sorted_token_ids: Sorted token indices according to their allocated
//  *                            expert. Max shape:
//  *                            [total_token*topk + topk*(block_size - 1)].
//  * @param experts_ids: The assigned expert index for each block. Max
//  *                            shape: [total_token*topk + expert].
//  * @param num_tokens_post_pad: Shape: [1]. Number of tokens after padding.
//  * @param topk_ids: Shape: [total_tokens, topk]. The top-k expert
//  *                            indices for each token.
//  * @param num_experts: Number of experts.
//  * @param block_size: Block size.
//  * @return int: 0 for success, non-zero for error.
//  */
// inline int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids,
//                                  at::Tensor& experts_ids,
//                                  at::Tensor& num_tokens_post_pad,
//                                  const at::Tensor& topk_ids, int num_experts,
//                                  int block_size) {
//   try {
//     TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D tensor");
//     TORCH_CHECK(topk_ids.dtype() == at::kInt, "topk_ids must be int32
//     tensor");

//     auto device = topk_ids.device();
//     auto options =
//     torch::TensorOptions().dtype(torch::kInt32).device(device);

//     int64_t token_num = topk_ids.size(0);
//     int64_t topk = topk_ids.size(1);
//     int64_t real_token_num =
//         token_num * topk;  // Use all tokens when real_token_num not provided

//     // Validate expert IDs are in range
//     auto topk_ids_flat = topk_ids.flatten();
//     auto topk_ids_cpu = topk_ids_flat.cpu();
//     auto topk_ids_ptr = topk_ids_cpu.data_ptr<int32_t>();
//     for (int64_t i = 0; i < real_token_num; ++i) {
//       int32_t expert_id = topk_ids_ptr[i];
//       TORCH_CHECK(expert_id >= 0 && expert_id < num_experts,
//                   "Expert ID out of range: ", expert_id);
//     }

//     // Step 1: Count tokens per expert using bincount
//     auto expert_token_cnt =
//         torch::bincount(topk_ids_flat, torch::Tensor(), num_experts);
//     expert_token_cnt = expert_token_cnt.to(torch::kInt32);

//     // Step 2: Calculate padded cumulative sum
//     auto expert_token_cnt_cpu = expert_token_cnt.cpu();
//     auto expert_token_cnt_accessor =
//         expert_token_cnt_cpu.accessor<int32_t, 1>();

//     std::vector<int32_t> padded_cumsum_vec(num_experts + 1, 0);
//     for (int64_t i = 0; i < num_experts; ++i) {
//       int32_t expert_cnt = expert_token_cnt_accessor[i];
//       int32_t padded_count =
//           (expert_cnt + block_size - 1) / block_size * block_size;
//       padded_cumsum_vec[i + 1] = padded_cumsum_vec[i] + padded_count;
//     }
//     int32_t num_tokens_post_pad_val = padded_cumsum_vec[num_experts];

//     // Create output tensors with correct sizes
//     auto padded_cumsum =
//         torch::from_blob(padded_cumsum_vec.data(), {num_experts + 1},
//                          torch::dtype(torch::kInt32))
//             .clone()
//             .to(device);

//     // Step 3: Generate expert IDs for each block
//     auto expert_id_num = num_tokens_post_pad_val / block_size;
//     auto expert_ids_result = torch::zeros({expert_id_num}, options);

//     auto padded_cumsum_cpu = padded_cumsum.cpu().to(torch::kInt32);
//     auto padded_cumsum_accessor = padded_cumsum_cpu.accessor<int32_t, 1>();
//     auto expert_ids_accessor = expert_ids_result.accessor<int32_t, 1>();

//     int32_t expert = 0;
//     for (int32_t i = 0; i < num_tokens_post_pad_val; i += block_size) {
//       while (i >= padded_cumsum_accessor[expert + 1]) {
//         expert += 1;
//       }
//       expert_ids_accessor[i / block_size] = expert;
//     }

//     // Step 4: Sort token indices by expert assignment
//     auto sorted_topk_ids_result =
//         torch::full({num_tokens_post_pad_val}, real_token_num, options);

//     auto topk_ids_flat_cpu = topk_ids_flat.cpu();
//     auto sorted_topk_ids_cpu = sorted_topk_ids_result.cpu();
//     auto topk_ids_flat_ptr = topk_ids_flat_cpu.data_ptr<int32_t>();
//     auto sorted_topk_ids_ptr = sorted_topk_ids_cpu.data_ptr<int32_t>();

//     std::vector<int32_t> expert_token_idx(num_experts, 0);

//     int32_t idx = 0;
//     for (int64_t i = 0; i < real_token_num; ++i) {
//       int32_t expert_id = topk_ids_flat_ptr[i];
//       int32_t pos =
//           padded_cumsum_accessor[expert_id] + expert_token_idx[expert_id];
//       sorted_topk_ids_ptr[pos] = idx;
//       expert_token_idx[expert_id] += 1;
//       ++idx;
//     }

//     // Copy results to output tensors using copy_()
//     if (num_tokens_post_pad_val > 0) {
//       sorted_token_ids = sorted_topk_ids_cpu.to(device);
//     } else {
//       sorted_token_ids = torch::empty({0}, options);
//     }
//     if (expert_id_num > 0) {
//       experts_ids = expert_ids_result;
//     } else {
//       experts_ids = torch::empty({0}, options);
//     }
//     num_tokens_post_pad = torch::tensor({num_tokens_post_pad_val}, options);

//     return 0;  // Success
//   } catch (const std::exception& e) {
//     return -1;  // Error
//   }
// }

// /**
//  * @brief Moe align block size native implementation with real_token_num
//  * parameter
//  *
//  * @param sorted_token_ids: Sorted token indices according to their allocated
//  *                            expert. Max shape:
//  *                            [total_token*topk + topk*(block_size - 1)].
//  * @param experts_ids: The assigned expert index for each block. Max
//  *                            shape: [total_token*topk + expert].
//  * @param num_tokens_post_pad: Shape: [1]. Number of tokens after padding.
//  * @param topk_ids: Shape: [total_tokens, topk]. The top-k expert
//  *                            indices for each token.
//  * @param real_token_num: Shape: [1]. The actual number of valid tokens.
//  * @param num_experts: Number of experts.
//  * @param block_size: Block size.
//  * @return int: 0 for success, non-zero for error.
//  */
// inline int vllmMoeAlignBlockSize(at::Tensor& sorted_token_ids,
//                                  at::Tensor& experts_ids,
//                                  at::Tensor& num_tokens_post_pad,
//                                  const at::Tensor& topk_ids,
//                                  const at::Tensor& real_token_num,
//                                  int num_experts, int block_size) {
//   try {
//     TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D tensor");
//     TORCH_CHECK(topk_ids.dtype() == at::kInt, "topk_ids must be int32
//     tensor"); TORCH_CHECK(real_token_num.dim() == 1 && real_token_num.size(0)
//     == 1,
//                 "real_token_num must be [1] tensor");
//     TORCH_CHECK(real_token_num.dtype() == at::kInt,
//                 "real_token_num must be int32 tensor");

//     auto device = topk_ids.device();
//     auto options =
//     torch::TensorOptions().dtype(torch::kInt32).device(device);

//     int64_t token_num = topk_ids.size(0);
//     int64_t topk = topk_ids.size(1);
//     int32_t real_token_count = real_token_num.item<int32_t>();

//     TORCH_CHECK(real_token_count <= token_num * topk,
//                 "real_token_num cannot exceed total tokens: ",
//                 real_token_count, " > ", token_num * topk);

//     // Validate expert IDs are in range
//     auto topk_ids_flat = topk_ids.flatten();
//     auto topk_ids_cpu = topk_ids_flat.cpu();
//     auto topk_ids_ptr = topk_ids_cpu.data_ptr<int32_t>();
//     for (int64_t i = 0; i < real_token_count; ++i) {
//       int32_t expert_id = topk_ids_ptr[i];
//       TORCH_CHECK(expert_id >= 0 && expert_id < num_experts,
//                   "Expert ID out of range: ", expert_id);
//     }

//     // Step 1: Count tokens per expert using bincount (only count up to
//     // real_token_num)
//     auto valid_topk_ids = topk_ids_flat.slice(0, 0, real_token_count);
//     auto expert_token_cnt =
//         torch::bincount(valid_topk_ids, torch::Tensor(), num_experts);
//     expert_token_cnt = expert_token_cnt.to(torch::kInt32);

//     // Step 2: Calculate padded cumulative sum
//     auto expert_token_cnt_cpu = expert_token_cnt.cpu();
//     auto expert_token_cnt_accessor =
//         expert_token_cnt_cpu.accessor<int32_t, 1>();

//     std::vector<int32_t> padded_cumsum_vec(num_experts + 1, 0);
//     for (int64_t i = 0; i < num_experts; ++i) {
//       int32_t expert_cnt = expert_token_cnt_accessor[i];
//       int32_t padded_count =
//           (expert_cnt + block_size - 1) / block_size * block_size;
//       padded_cumsum_vec[i + 1] = padded_cumsum_vec[i] + padded_count;
//     }
//     int32_t num_tokens_post_pad_val = padded_cumsum_vec[num_experts];

//     // Create output tensors with correct sizes
//     auto padded_cumsum =
//         torch::from_blob(padded_cumsum_vec.data(), {num_experts + 1},
//                          torch::dtype(torch::kInt32))
//             .clone()
//             .to(device);

//     // Step 3: Generate expert IDs for each block
//     auto expert_id_num = num_tokens_post_pad_val / block_size;
//     auto expert_ids_result = torch::zeros({expert_id_num}, options);

//     auto padded_cumsum_cpu = padded_cumsum.cpu().to(torch::kInt32);
//     auto padded_cumsum_accessor = padded_cumsum_cpu.accessor<int32_t, 1>();
//     auto expert_ids_accessor = expert_ids_result.accessor<int32_t, 1>();

//     int32_t expert = 0;
//     for (int32_t i = 0; i < num_tokens_post_pad_val; i += block_size) {
//       while (i >= padded_cumsum_accessor[expert + 1]) {
//         expert += 1;
//       }
//       expert_ids_accessor[i / block_size] = expert;
//     }

//     // Step 4: Sort token indices by expert assignment (only use
//     // real_token_count)
//     auto sorted_topk_ids_result =
//         torch::full({num_tokens_post_pad_val}, real_token_count, options);

//     auto topk_ids_flat_cpu = topk_ids_flat.cpu();
//     auto sorted_topk_ids_cpu = sorted_topk_ids_result.cpu();
//     auto topk_ids_flat_ptr = topk_ids_flat_cpu.data_ptr<int32_t>();
//     auto sorted_topk_ids_ptr = sorted_topk_ids_cpu.data_ptr<int32_t>();

//     std::vector<int32_t> expert_token_idx(num_experts, 0);

//     int32_t idx = 0;
//     for (int32_t i = 0; i < real_token_count; ++i) {
//       int32_t expert_id = topk_ids_flat_ptr[i];
//       int32_t pos =
//           padded_cumsum_accessor[expert_id] + expert_token_idx[expert_id];
//       sorted_topk_ids_ptr[pos] = idx;
//       expert_token_idx[expert_id] += 1;
//       ++idx;
//     }

//     // Copy results to output tensors using copy_()
//     if (num_tokens_post_pad_val > 0) {
//       sorted_token_ids = sorted_topk_ids_cpu.to(device);
//     } else {
//       sorted_token_ids = torch::empty({0}, options);
//     }
//     if (expert_id_num > 0) {
//       experts_ids = expert_ids_result;
//     } else {
//       experts_ids = torch::empty({0}, options);
//     }
//     num_tokens_post_pad = torch::tensor({num_tokens_post_pad_val}, options);

//     return 0;  // Success
//   } catch (const std::exception& e) {
//     return -1;  // Error
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
//                                           const at::Tensor& in,
//                                           const int32_t group_size) {
//   // Get tensor dimensions
//   auto sizes = in.sizes();

//   // Calculate dimensions following the same logic as CPU implementation
//   int dim0 = 1;
//   int dim1;
//   int lowest_dim_index = sizes.size() - 1;

//   for (int i = 0; i < lowest_dim_index; i++) {
//     dim0 *= sizes[i];
//   }
//   dim1 = sizes[lowest_dim_index];

//   int out_dim1 = dim1 / 2;
//   int out_elem_num = dim0 * out_dim1;
//   int group_num = out_elem_num / group_size;

//   // Convert input to float32 for computation (consistent with CPU
//   // implementation)
//   auto in_f32 = in.to(at::kFloat);

//   // Split input tensor into two halves along the last dimension
//   auto in1 = in_f32.narrow(-1, 0, out_dim1);         // First half: [..., N]
//   auto in2 = in_f32.narrow(-1, out_dim1, out_dim1);  // Second half: [..., N]

//   // Step 1: Apply SiLU activation to first half
//   // SiLU formula: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   auto silu_result = in1 * at::sigmoid(in1);

//   // Step 2: Element-wise multiplication with second half
//   auto silu_mul = silu_result * in2;

//   // Flatten for group processing - maintain same memory layout as CPU
//   // implementation
//   auto silu_mul_flat = silu_mul.contiguous().view(-1);

//   // Verify we have the expected number of elements
//   if (silu_mul_flat.numel() != out_elem_num) {
//     throw std::runtime_error("Element count mismatch in native
//     implementation");
//   }

//   // Step 3: Per-group quantization
//   // Reshape to groups for scale computation: [group_num, group_size]
//   auto silu_mul_groups = silu_mul_flat.view({group_num, group_size});

//   // Compute absolute values and find maximum per group
//   auto abs_values = at::abs(silu_mul_groups);
//   auto group_max = std::get<0>(at::max(abs_values, /*dim=*/1));

//   // Determine FP8 maximum value based on output tensor type
//   float fp8_max;
//   if (out.scalar_type() == at::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;  // E4M3_MAX
//   } else {
//     fp8_max = 57344.0f;  // E5M2_MAX
//   }

//   // Step 4: Compute scale factors
//   // scale = max(abs(group_values)) / fp8_max
//   auto scales = group_max / fp8_max;
//   scale.copy_(scales);

//   // Step 5: Apply quantization by dividing by scale
//   // quantized = group_values / scale
//   auto scales_expanded = scales.unsqueeze(1).expand({group_num, group_size});
//   auto quantized = silu_mul_groups / scales_expanded;

//   // Step 6: Convert to FP8 format
//   // Flatten back to 1D and convert to target FP8 type
//   auto quantized_flat = quantized.view(-1);
//   auto quantized_fp8 = quantized_flat.to(out.scalar_type());

//   // Copy to output tensor
//   auto out_flat = out.contiguous().view(-1);
//   out_flat.copy_(quantized_fp8);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// // FP8 format constants
// constexpr float FP8_E4M3_MAX = 448.0f;
// constexpr float FP8_E5M2_MAX = 57344.0f;

// /**
//  * @brief RMS Norm Per Token Group Quantization implementation using PyTorch
//  * native operations
//  *
//  * This function performs RMS normalization followed by group-wise
//  quantization
//  * to FP8 format. It uses PyTorch's built-in tensor operations for efficient
//  * computation.
//  *
//  * @param output       Output quantized tensor of shape [N, D]. This tensor
//  * will store the quantized results in the specified FP8 format (e4m3fn or
//  * e5m2).
//  * @param scaling      Scaling factors tensor of shape [N, D/group_size].
//  Each
//  * element represents the scaling factor for one group of consecutive
//  elements.
//  * @param input        Input tensor of shape [N, D]. Can be any
//  floating-point
//  * format. N represents the batch/token dimension, D represents the feature
//  * dimension.
//  * @param weight       Weight tensor of shape [D]. Used for scaling after RMS
//  * normalization.
//  * @param epsilon      Small value added to variance for numerical stability
//  * (typically 1e-6). Prevents division by zero in RMS normalization.
//  * @param group_size   Number of consecutive elements to group together for
//  * quantization. Must divide D evenly. Common values: 32, 64, 128, 256.
//  * @param output_dtype Target quantization format (torch::kFloat8_e4m3fn or
//  * torch::kFloat8_e5m2).
//  */
// inline void vllmRmsNormPerTokenGroupQuantFp8Impl(
//     at::Tensor& output, at::Tensor& scaling, const at::Tensor& input,
//     const at::Tensor& weight, const float epsilon, const int64_t group_size,
//     at::ScalarType output_dtype) {
//   // Convert inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);

//   // Get tensor dimensions
//   auto sizes = input_f32.sizes();
//   int64_t n = 1;
//   for (int i = 0; i < sizes.size() - 1; ++i) {
//     n *= sizes[i];
//   }
//   int64_t d = sizes[sizes.size() - 1];
//   int64_t group_num = d / group_size;

//   // Determine FP8 max value based on output type
//   float fp8_max;
//   if (output_dtype == torch::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;  // FP8_E4M3_MAX
//   } else if (output_dtype == torch::kFloat8_e5m2) {
//     fp8_max = 57344.0f;  // FP8_E5M2_MAX
//   } else {
//     fp8_max = 127.0f;  // For int8 fallback
//   }
//   float recp_fp8 = 1.0f / fp8_max;

//   // Step 1: Compute RMS normalization
//   // variance = mean(input^2, dim=-1, keepdim=True)
//   auto variance = at::mean(input_f32 * input_f32, /*dim=*/-1,
//   /*keepdim=*/true);

//   // Apply RMS normalization: input * rsqrt(variance + epsilon)
//   auto normalized = input_f32 * at::rsqrt(variance + epsilon);

//   // Step 2: Apply weight scaling
//   auto weighted = normalized * weight_f32;

//   // Step 3: Group quantization
//   // Reshape to [N, group_num, group_size] for group-wise operations
//   auto reshaped = weighted.view({n, group_num, group_size});

//   // Find maximum absolute value per group: [N, group_num]
//   auto max_result = at::max(at::abs(reshaped), /*dim=*/-1);
//   auto group_max = std::get<0>(max_result);

//   // Calculate scaling factors: [N, group_num]
//   auto scaling_factors = group_max * recp_fp8;

//   // Expand scaling factors to match reshaped tensor: [N, group_num, 1]
//   auto scaling_expanded = scaling_factors.unsqueeze(-1);

//   // Quantize: divide by scaling and clamp to fp8 range
//   auto quantized = (reshaped / scaling_expanded).clamp(-fp8_max, fp8_max);

//   // Reshape back to original shape [N, D]
//   auto quantized_reshaped = quantized.view({n, d});

//   // Convert to target output type
//   at::Tensor final_output;
//   if (output_dtype == torch::kFloat8_e4m3fn ||
//       output_dtype == torch::kFloat8_e5m2) {
//     final_output = quantized_reshaped.to(output_dtype);
//   } else {
//     // For integer types, round first then convert
//     final_output = at::round(quantized_reshaped).to(output_dtype);
//   }

//   // Copy results to output tensors
//   output.copy_(final_output);
//   scaling.copy_(scaling_factors);
// }

// /**
//  * @brief Unified entry point for RMS Norm Per Token Group Quantization
//  *
//  * Automatically determines the output data type from the output tensor and
//  * calls the appropriate implementation.
//  *
//  * @param output    Pre-allocated output tensor with the desired quantization
//  * format. Shape: [N, D]. The scalar type determines the quantization target.
//  * @param scaling   Pre-allocated scaling tensor for storing group scaling
//  * factors. Shape: [N, D/group_size]. Must be float32 format.
//  * @param input     Input tensor to be normalized and quantized. Shape: [N,
//  D].
//  * @param weight    Weight tensor for post-normalization scaling. Shape: [D].
//  * @param epsilon   Numerical stability constant for RMS normalization.
//  * @param group_size Size of each quantization group. Must divide D evenly.
//  */
// inline void vllmRmsNormPerTokenGroupQuantFp8(
//     at::Tensor& output, at::Tensor& scaling, const at::Tensor& input,
//     const at::Tensor& weight, const float epsilon, const int64_t group_size)
//     {
//   auto output_dtype = output.scalar_type();
//   vllmRmsNormPerTokenGroupQuantFp8Impl(output, scaling, input, weight,
//   epsilon,
//                                        group_size, output_dtype);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// /**
//  * @brief In-place fused Add and RMS Normalization (Native implementation
//  using
//  * torch interfaces).
//  *
//  * @param input: Input tensor of shape [..., hidden_size]. Modified in-place.
//  * @param residual: Residual tensor of shape [..., hidden_size].
//  * @param weight: Weight tensor of shape [hidden_size].
//  * @param epsilon: Value added to denominator for numerical stability.
//  */
// inline void vllmFusedAddRmsNorm(at::Tensor& input, at::Tensor& residual,
//                                 const at::Tensor& weight, float epsilon) {
//   auto data_type = input.dtype();

//   // Follow the exact logic from rule.MD reference implementation
//   at::Tensor input_tensor, weight_tensor, residual_tensor;

//   if (data_type != at::kBFloat16) {
//     input_tensor =
//         input;  // torch.tensor(input) - but input is already a tensor
//     weight_tensor = weight;      // torch.tensor(weight)
//     residual_tensor = residual;  // torch.tensor(residual)
//   } else {
//     input_tensor = input;
//     weight_tensor = weight;
//     residual_tensor = residual;
//   }

//   // Always convert to float32 for computation (following rule.MD exactly)
//   auto input_tensor_f32 = input_tensor.to(at::kFloat);
//   auto weight_f32 = weight_tensor.to(at::kFloat);
//   auto residual_tensor_f32 = residual_tensor.to(at::kFloat);

//   // Add input and residual: input = input + residual
//   input_tensor_f32 = input_tensor_f32 + residual_tensor_f32;

//   // Compute RMS norm square: x^2
//   auto rms_norm_square = input_tensor_f32.pow(2);

//   // Compute rsqrt(mean(x^2) + epsilon)
//   auto rms_norm_rsqrt =
//       at::rsqrt(rms_norm_square.mean(-1, /*keepdim=*/true) + epsilon);

//   // Apply RMS normalization and weight scaling: x * rsqrt * weight
//   auto rms_output_tensor = input_tensor_f32 * rms_norm_rsqrt * weight_f32;

//   // Convert back to original dtype and copy in-place to input tensor
//   input.copy_(rms_output_tensor.to(data_type));
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// inline void extSiluAndMul(at::Tensor& out, const at::Tensor& in,
//                           const at::Tensor& size = {}) {
//   // Get tensor properties
//   int64_t rank = in.dim();
//   int64_t hidden_size = in.size(-1);
//   int64_t d = hidden_size / 2;
//   int64_t batch_size = in.size(0);

//   // Extract real_size from tensor or use default
//   int64_t real_size;
//   if (size.defined() && size.numel() > 0) {
//     // Extract value from scalar tensor (convert to int64 for consistency)
//     real_size = size.cpu().item<int32_t>();
//   } else {
//     real_size = batch_size;  // Default to full batch size
//   }

//   // Check constraints
//   TORCH_CHECK(real_size <= batch_size, "real_size (", real_size,
//               ") must be <= batch_size (", batch_size, ")");

//   // Check tensor rank (support 2D and 3D)
//   TORCH_CHECK(rank == 2 || rank == 3, "Only 2D and 3D tensors are
//   supported");

//   // Check that the last dimension of input tensor is even
//   TORCH_CHECK(hidden_size % 2 == 0,
//               "Last dimension of input_tensor must be even");

//   // Check that the last dimension of output tensor is half of input
//   TORCH_CHECK(out.size(-1) == d,
//               "Last dimension of out_tensor must be half of input_tensor's "
//               "last dimension");

//   // Check that all dimensions except the last one match (considering
//   real_size) TORCH_CHECK(
//       out.dim() == rank,
//       "Input and output tensors must have the same number of dimensions");
//   TORCH_CHECK(out.size(0) >= real_size,
//               "Output tensor's first dimension must be >= real_size");
//   for (int64_t i = 1; i < rank - 1; ++i) {
//     TORCH_CHECK(out.size(i) == in.size(i),
//                 "Dimension " + std::to_string(i) +
//                     " of output tensor must match input tensor");
//   }

//   // Get the effective region of input tensor (first real_size batches)
//   auto effective_input = in.narrow(0, 0, real_size);

//   // Split the effective input tensor along the last dimension
//   // This works for both 2D and 3D tensors due to Ellipsis
//   auto x1 = effective_input.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(0, d)});
//   auto x2 = effective_input.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(d, hidden_size)});

//   // Apply SiLU activation function: x1 / (1 + exp(-x1))
//   auto silu = x1 / (1 + torch::exp(-x1));

//   // Multiply by the second half
//   auto result = silu * x2;

//   // Convert result to the same dtype as output and copy only to the
//   effective
//   // region
//   out.narrow(0, 0, real_size).copy_(result.to(out.scalar_type()));
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// inline void extsRotaryEmbeddingWithKVCache(
//     at::Tensor& q_out, at::Tensor& kv_cache, const at::Tensor& q,
//     const at::Tensor& kv, const at::Tensor& positions,
//     const at::Tensor& cos_sin_cache, const at::Tensor& weight,
//     const at::Tensor& slot_mapping, const at::Tensor& scale, double eps,
//     const std::vector<int64_t>& split_size,
//     const std::string& kv_cache_dtype = "auto") {
//   // Convert inputs to float32 for computation
//   auto q_f32 = q.to(at::kFloat);
//   auto kv_f32 = kv.to(at::kFloat);
//   auto cos_sin_f32 = cos_sin_cache.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);

//   auto q_sizes = q_f32.sizes();
//   auto kv_sizes = kv_f32.sizes();

//   int64_t tokens = q_sizes[0];
//   int64_t q_dim = q_sizes[1];
//   int64_t kv_dim = kv_sizes[1];
//   int64_t rms_dim = split_size[0];
//   int64_t rot_dim = 32;  // ROT_DIM from the test code

//   // Apply rotary embedding to q using tensor operations
//   auto q_result = q_f32.clone();

//   // Extract rotary dimensions from q (first rot_dim*2 elements)
//   auto q_rot = q_f32.narrow(1, 0, rot_dim * 2).view({tokens, rot_dim, 2});
//   auto q_x = q_rot.select(2, 0);  // shape: [tokens, rot_dim]
//   auto q_y = q_rot.select(2, 1);  // shape: [tokens, rot_dim]

//   // Extract cos and sin from cos_sin_cache tensor
//   auto cos_vals =
//       cos_sin_f32.narrow(1, 0, rot_dim);  // shape: [tokens, rot_dim]
//   auto sin_vals =
//       cos_sin_f32.narrow(1, rot_dim, rot_dim);  // shape: [tokens, rot_dim]

//   // Apply rotary transformation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
//   auto q_x_new = q_x * cos_vals - q_y * sin_vals;
//   auto q_y_new = q_x * sin_vals + q_y * cos_vals;

//   // Stack and reshape back to original format
//   auto q_rot_new = at::stack({q_x_new, q_y_new}, 2).view({tokens, rot_dim *
//   2});

//   // Update q_result with rotated values
//   q_result.narrow(1, 0, rot_dim * 2).copy_(q_rot_new);

//   // Process kv tensor
//   auto kv_result = kv_f32.clone();

//   // Extract the first rms_dim elements for RMS normalization
//   auto kv_rms_part = kv_f32.narrow(1, 0, rms_dim);  // shape: [tokens,
//   rms_dim]

//   // Compute RMS normalization using tensor operations
//   auto variance = at::mean(kv_rms_part * kv_rms_part, /*dim=*/1,
//                            /*keepdim=*/true);        // shape: [tokens, 1]
//   auto rms_norm_factor = at::rsqrt(variance + eps);  // shape: [tokens, 1]

//   // Apply RMS normalization and weight scaling
//   auto kv_normalized = kv_rms_part * rms_norm_factor;  // Broadcasting:
//   [tokens,
//                                                        // rms_dim] * [tokens,
//                                                        1]
//   auto kv_weighted = kv_normalized * weight_f32.unsqueeze(0);  //
//   Broadcasting:
//                                                                // [tokens,
//                                                                // rms_dim] *
//                                                                [1,
//                                                                // rms_dim]

//   // Update kv_result with normalized and weighted values
//   kv_result.narrow(1, 0, rms_dim).copy_(kv_weighted);

//   // Apply rotary embedding to k part (starting from rms_dim offset)
//   if (rms_dim + rot_dim * 2 <= kv_dim) {
//     // Extract rotary dimensions from k part
//     auto k_rot =
//         kv_f32.narrow(1, rms_dim, rot_dim * 2).view({tokens, rot_dim, 2});
//     auto k_x = k_rot.select(2, 0);  // shape: [tokens, rot_dim]
//     auto k_y = k_rot.select(2, 1);  // shape: [tokens, rot_dim]

//     // Apply rotary transformation to k using the same cos/sin values
//     auto k_x_new = k_x * cos_vals - k_y * sin_vals;
//     auto k_y_new = k_x * sin_vals + k_y * cos_vals;

//     // Stack and reshape back to original format
//     auto k_rot_new =
//         at::stack({k_x_new, k_y_new}, 2).view({tokens, rot_dim * 2});

//     // Update kv_result with rotated k values
//     kv_result.narrow(1, rms_dim, rot_dim * 2).copy_(k_rot_new);
//   }

//   // Copy results to output tensors with proper dtype conversion
//   q_out.copy_(q_result.to(q_out.dtype()));
//   kv_cache.copy_(kv_result.to(kv_cache.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// #include <vector>

// // 使用PyTorch API实现的版本
// inline void extsFusedDispatchDecode(std::vector<at::Tensor>& outputs,
//                                     const at::Tensor& input,
//                                     const at::Tensor& split_sizes_tensor,
//                                     const std::vector<int64_t>& split_sizes)
//                                     {
//   // 参数检查
//   TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
//   TORCH_CHECK(split_sizes_tensor.dim() == 1, "Split sizes tensor must be
//   1D");

//   // 计算有效行数
//   at::Tensor reduce_sum = split_sizes_tensor.sum(0, true);
//   int64_t valid_rows = reduce_sum.item<int64_t>();
//   valid_rows = std::min(valid_rows, input.size(0));

//   // 计算分割偏移量
//   std::vector<int64_t> split_offsets(split_sizes.size());
//   split_offsets[0] = 0;
//   for (size_t i = 1; i < split_sizes.size(); ++i) {
//     split_offsets[i] = split_offsets[i - 1] + split_sizes[i - 1];
//   }

//   // 生成输出张量
//   outputs.clear();
//   int64_t dim = input.dim() - 1;  // 在最后一维分割

//   for (size_t i = 0; i < split_sizes.size(); ++i) {
//     // 切片并保持连续性
//     at::Tensor slice =
//         input.slice(dim, split_offsets[i], split_offsets[i] + split_sizes[i])
//             .contiguous();

//     // 只保留有效行
//     if (valid_rows < input.size(0)) {
//       slice = slice.slice(0, 0, valid_rows);
//     }

//     outputs.push_back(slice);
//   }
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// #include <c10/util/Exception.h>

// #include <algorithm>
// #include <cmath>
// #include <iomanip>
// #include <vector>

// // PyTorch风格实现的PagedAttentionV1，使用张量操作而非直接内存访问
// inline void vllmPagedAttentionV1(
//     at::Tensor& out, const at::Tensor& q, const at::Tensor& k,
//     const at::Tensor& v, const at::Tensor& head_mapping, const float scale,
//     const at::Tensor& block_tables, const at::Tensor& context_lens,
//     const int block_size, const int max_context_len,
//     const at::Tensor& alibi_slopes = at::Tensor(),
//     const std::string kv_cache_dtype = "int8", const float k_scale = 1.0f,
//     const float k_zp = 0.0f, const float v_scale = 1.0f,
//     const float v_zp = 0.0f, const at::Tensor& out_scales = at::Tensor()) {
//   // 参数检查
//   TORCH_CHECK(q.dim() == 3,
//               "q should be 3D tensor [batch_size, num_heads, head_size]");
//   TORCH_CHECK(
//       block_tables.dim() == 2,
//       "block_tables should be 2D tensor [batch_size, max_blocks_per_seq]");
//   TORCH_CHECK(context_lens.dim() == 1,
//               "context_lens should be 1D tensor [batch_size]");

//   // 获取尺寸
//   int batch_size = q.size(0);
//   int num_heads = q.size(1);
//   int head_size = q.size(2);
//   int max_blocks_per_seq = block_tables.size(1);

//   // 检查KV缓存形状
//   TORCH_CHECK(k.dim() == 4,
//               "k should be 4D tensor [num_blocks, num_kv_heads, block_size, "
//               "head_size]");
//   int num_kv_heads = k.size(1);
//   TORCH_CHECK(head_size == k.size(3), "Head size of q and k should match");
//   TORCH_CHECK(block_size == k.size(2), "Block size does not match
//   k.size(2)"); TORCH_CHECK(v.dim() == 4,
//               "v should be 4D tensor [num_blocks, num_kv_heads, block_size, "
//               "head_size]");
//   TORCH_CHECK(num_kv_heads == v.size(1),
//               "Number of KV heads should match between k and v");

//   // 初始化输出
//   out.zero_();

//   // 获取设备信息，确保所有操作都在同一设备上
//   auto device = q.device();
//   auto dtype = q.dtype();

//   // 检查是否使用alibi
//   bool use_alibi = alibi_slopes.defined() && alibi_slopes.numel() > 0;

//   // 检查输出量化
//   bool use_out_quant = out_scales.defined() && out_scales.numel() > 0;

//   // 计算k_zp和v_zp的实际值
//   float k_zero_point = k_scale * k_zp;
//   float v_zero_point = v_scale * v_zp;

//   // 获取head_mapping (如果提供的话)
//   at::Tensor effective_head_mapping;
//   if (head_mapping.defined() && head_mapping.numel() > 0) {
//     effective_head_mapping = head_mapping;
//   } else {
//     // 如果没有提供head_mapping，创建一个恒等映射
//     effective_head_mapping = torch::arange(
//         num_heads, torch::TensorOptions().dtype(torch::kInt).device(device));
//   }

//   // 创建int8量化的解码方法（如果需要）
//   auto dequantize_int8 = [&](const at::Tensor& int8_tensor, float scale,
//                              float zero_point) {
//     return int8_tensor.to(torch::kFloat) * scale - zero_point;
//   };

//   // 对每个批次单独处理
//   for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
//     // 获取当前序列的上下文长度
//     int ctx_len = std::min(context_lens[seq_idx].item<int>(),
//     max_context_len);

//     // 计算需要的块数
//     int num_blocks = (ctx_len + block_size - 1) / block_size;
//     num_blocks = std::min(num_blocks, max_blocks_per_seq);

//     // 对每个头单独处理
//     for (int head_idx = 0; head_idx < num_heads; head_idx++) {
//       // 获取对应的KV头索引
//       int kv_head_idx = effective_head_mapping[head_idx].item<int>();

//       // 安全检查
//       TORCH_CHECK(kv_head_idx < num_kv_heads,
//                   "KV head index out of range: ", kv_head_idx,
//                   " >= ", num_kv_heads);

//       // 获取当前查询向量
//       at::Tensor query_vec = q[seq_idx][head_idx];

//       // 准备存储当前序列所有token的KV值
//       std::vector<at::Tensor> key_vecs;
//       std::vector<at::Tensor> value_vecs;

//       // 遍历块以收集所有KV向量
//       for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
//         // 获取KV块索引
//         int kv_block_idx = block_tables[seq_idx][block_idx].item<int>();

//         // 获取当前块中有效的token数量
//         int valid_tokens =
//             std::min(block_size, ctx_len - block_idx * block_size);

//         // 提取当前块的K和V
//         at::Tensor k_block =
//             k[kv_block_idx][kv_head_idx].slice(0, 0, valid_tokens);
//         at::Tensor v_block =
//             v[kv_block_idx][kv_head_idx].slice(0, 0, valid_tokens);

//         // 如果是int8量化，则进行解量化
//         if (kv_cache_dtype == "int8") {
//           k_block = dequantize_int8(k_block, k_scale, k_zero_point);
//           v_block = dequantize_int8(v_block, v_scale, v_zero_point);
//         }

//         // 添加到列表
//         key_vecs.push_back(k_block);
//         value_vecs.push_back(v_block);
//       }

//       // 合并所有KV向量
//       at::Tensor keys = torch::cat(key_vecs, 0);
//       at::Tensor values = torch::cat(value_vecs, 0);

//       // 计算注意力分数 (Q·K^T)
//       at::Tensor attn_scores =
//           torch::matmul(query_vec.unsqueeze(0), keys.transpose(0, 1))
//               .squeeze(0);

//       // 应用缩放因子
//       attn_scores = attn_scores * scale;

//       // 如果使用alibi，添加位置偏置
//       if (use_alibi) {
//         // 创建位置索引
//         at::Tensor positions = torch::arange(
//             ctx_len,
//             torch::TensorOptions().dtype(torch::kFloat).device(device));
//         positions = positions - ctx_len;  // 从-ctx_len到-1

//         // 获取当前头的alibi斜率
//         float alibi_slope = alibi_slopes[head_idx].item<float>();

//         // 应用alibi偏置
//         attn_scores = attn_scores + alibi_slope * positions;
//       }

//       // 仅保留有效上下文长度的分数
//       attn_scores = attn_scores.slice(0, 0, ctx_len);

//       // 应用softmax
//       at::Tensor attn_probs = torch::softmax(attn_scores, 0);

//       // 计算加权和 (attn_probs·V)
//       at::Tensor output =
//           torch::matmul(attn_probs.unsqueeze(0), values).squeeze(0);

//       // 写入结果
//       if (use_out_quant) {
//         // 输出量化处理
//         bool per_token_quant = out_scales.numel() > 1;

//         if (per_token_quant) {
//           // 每个token单独量化
//           at::Tensor scale_factors = out_scales.slice(
//               0, head_idx * head_size, (head_idx + 1) * head_size);
//           at::Tensor quant_vals = output * scale_factors;
//           quant_vals = torch::clamp(torch::round(quant_vals), -127.0, 127.0);
//           out[seq_idx][head_idx] = quant_vals.to(torch::kInt8);
//         } else {
//           // 单一量化因子
//           float scale_factor = out_scales[0].item<float>();
//           at::Tensor quant_vals = output * scale_factor;
//           quant_vals = torch::clamp(torch::round(quant_vals), -127.0, 127.0);
//           out[seq_idx][head_idx] = quant_vals.to(torch::kInt8);
//         }
//       } else {
//         // 直接复制浮点结果
//         out[seq_idx][head_idx] = output;
//       }
//     }
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmSiluAndMul(at::Tensor& output, const at::Tensor& input) {
//   // Get tensor dimensions
//   auto input_sizes = input.sizes();
//   int64_t batch_size = 1;
//   for (int i = 0; i < input_sizes.size() - 1; ++i) {
//     batch_size *= input_sizes[i];
//   }
//   int64_t hidden_size = input_sizes[input_sizes.size() - 1];
//   int64_t half_hidden_size = hidden_size / 2;

//   // Check that the last dimension of input tensor is even
//   TORCH_CHECK(hidden_size % 2 == 0,
//               "Last dimension of input tensor must be even");

//   // Check that output tensor has the correct shape
//   TORCH_CHECK(
//       output.size(-1) == half_hidden_size,
//       "Last dimension of output tensor must be half of input tensor's last "
//       "dimension");

//   // Convert input to float32 for computation
//   auto input_f32 = input.to(at::kFloat);

//   // Split the input tensor along the last dimension
//   auto x1 = input_f32.index(
//       {torch::indexing::Ellipsis, torch::indexing::Slice(0,
//       half_hidden_size)});
//   auto x2 =
//       input_f32.index({torch::indexing::Ellipsis,
//                        torch::indexing::Slice(half_hidden_size,
//                        hidden_size)});

//   // Apply SiLU activation function: x1 / (1 + exp(-x1))
//   auto silu_result = x1 / (1 + torch::exp(-x1));

//   // Multiply by the second half: silu_result * x2
//   auto result = silu_result * x2;

//   // Convert back to original dtype and copy to output tensor
//   output.copy_(result.to(input.dtype()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void vllmFusedAddRmsNormQuant(at::Tensor& output,
//                                      const at::Tensor& input,
//                                      const at::Tensor& residual,
//                                      const at::Tensor& weight, double
//                                      epsilon, const at::Tensor& scaling) {
//   // Convert inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto residual_f32 = residual.to(at::kFloat);
//   auto weight_f32 = weight.to(at::kFloat);
//   auto scales_f32 = scaling.to(at::kFloat);

//   // Add input and residual: x = input + residual
//   auto x = input_f32 + residual_f32;

//   // Compute variance: variance = mean(x * x, dim=-1, keepdim=True)
//   auto variance = at::mean(x * x, /*dim=*/-1, /*keepdim=*/true);

//   // Apply RMS normalization: x = x * rsqrt(variance + epsilon)
//   x = x * at::rsqrt(variance + epsilon);

//   // Apply weight scaling: x = x * weight
//   x = x * weight_f32;

//   // Quantize: round(x / scales) and clamp to int8 range [-128, 127]
//   auto scales_expanded = scales_f32.unsqueeze(-1);
//   auto quant_result =
//       at::round(x / scales_expanded).clamp(-128, 127).to(at::kChar);

//   // Copy result to output tensor
//   output.copy_(quant_result);
// } /**
//    * Copyright 2025 Enflame. All Rights Reserved.
//    */

// #include <iostream>

// /**
//  * @brief Linear quant operator (Native implementation using torch
//  interfaces)
//  *
//  * @param out The output tensor
//  * @param lhs The input tensor of lhs
//  * @param rhs The input tensor of rhs
//  * @param bias The input tensor of bias
//  * @param lhs_scale The lhs tensor of scale
//  * @param rhs_scale The rhs tensor of scale
//  */
// inline void atenLinearQuant(at::Tensor& out, const at::Tensor& lhs,
//                             const at::Tensor& rhs, const at::Tensor& bias,
//                             const at::Tensor& lhs_scale,
//                             const at::Tensor& rhs_scale) {
//   // Convert tensors to appropriate types if needed
//   at::Tensor input_tensor = lhs.to(at::kFloat);
//   at::Tensor weight_tensor = rhs;
//   at::Tensor bias_tensor = bias.to(at::kFloat);
//   at::Tensor scale_tensor = rhs_scale;

//   // Transpose weight tensor
//   at::Tensor weight_transpose_tensor = at::transpose(weight_tensor, 0, 1);

//   // Reshape the weight tensor
//   auto weight_shape = weight_transpose_tensor.sizes().vec();
//   int64_t n = weight_shape[1];
//   int64_t k = weight_shape[0];
//   int64_t group_num = scale_tensor.size(0);
//   int64_t group_size = k / group_num;

//   std::vector<int64_t> weight_transpose_shape_tmp = {group_num, group_size,
//   n}; at::Tensor weight_split_tensor =
//       at::reshape(weight_transpose_tensor, weight_transpose_shape_tmp);

//   // Apply scale to weight
//   at::Tensor quant_result = at::mul(weight_split_tensor, scale_tensor);

//   // Reshape back
//   at::Tensor quant_result2 = at::reshape(quant_result, {k, n});

//   // Transpose again
//   at::Tensor quant_result3 = at::transpose(quant_result2, 0, 1);

//   // Convert to float32 for linear operation
//   at::Tensor aten_scale_mul_tensor = quant_result3.to(at::kFloat);

//   // Perform linear operation
//   at::Tensor output_tensor;
//   if (bias.numel() > 0) {
//     output_tensor =
//         at::linear(input_tensor, aten_scale_mul_tensor, bias_tensor);
//   } else {
//     output_tensor = at::linear(input_tensor, aten_scale_mul_tensor);
//   }

//   // Handle activation if needed (assuming no activation is required based on
//   // the interface)

//   // Handle data type clamping if needed for int8 (char) type
//   bool is_int8_type = (out.scalar_type() == at::ScalarType::Char);
//   if (is_int8_type) {
//     output_tensor = at::clamp(output_tensor, -127.0f, 127.0f);
//   }

//   // Copy result to output tensor with proper dtype conversion
//   out.copy_(output_tensor.to(out.scalar_type()));
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// // First algorithm: General quantization version with Deq_Zeros support
// inline void vllmInvokeFusedMoeNonGatherQuantKernel(
//     at::Tensor& c,        // Output tensor C [M, topk, N] - inplace
//     modification const at::Tensor& a,  // Input tensor A [*, K] const
//     at::Tensor& b,  // Weight tensor B [E, N, K] const at::Tensor& scale,  //
//     Scale for dequant [E, N] int64_t gs,               // Group size for
//     pre-group type dequant const at::Tensor&
//         deq_zeros,           // Zero point for dequant [E, N] or [E, K/gs, N]
//     const at::Tensor& bias,  // Bias tensor [N]
//     const at::Tensor& topk_weights,  // Top-k expert weights for each token
//     const at::Tensor& topk_ids,      // Top-k expert indices for each token
//     const at::Tensor&
//         sorted_token_ids,  // Sorted token indices according to allocated
//         expert
//     const at::Tensor& experts_ids,  // Assigned expert index for each block
//     const at::Tensor&
//         num_tokens_post_pad,           // Number of tokens after padding [1]
//     const at::Tensor& real_token_num,  // Actual number of valid tokens [1]
//     bool mul_routed_weight,            // Flag for topk_weights participation
//     int64_t topk,                      // Number of experts for each token
//     int64_t block_size                 // Block size
// ) {
//   // Get input shapes - match CPU implementation exactly
//   auto a_sizes = a.sizes();
//   auto b_sizes = b.sizes();

//   int64_t a_numel = a.numel();
//   int64_t k = a_sizes[a_sizes.size() - 1];  // Last dimension is K
//   int64_t m = a_numel / k;                  // Total number of tokens
//   int64_t e = b_sizes[0];                   // Number of experts
//   int64_t n = b_sizes[1];                   // Output feature dimension

//   // Calculate token_num from topk_weights shape
//   auto topk_weights_sizes = topk_weights.sizes();
//   int64_t token_num = topk_weights_sizes[0];
//   int64_t actual_topk = topk_weights_sizes[1];

//   // Validate inputs
//   if (k == 0 || e == 0 || n == 0 || token_num == 0 || actual_topk == 0) {
//     return;
//   }

//   // Get tensors on same device and contiguous - avoid unnecessary device
//   // transfers
//   auto a_cont = a.contiguous().to(torch::kFloat32).view({m, k});
//   auto b_cont = b.contiguous().to(torch::kFloat32);
//   auto scale_cont = scale.contiguous().to(torch::kFloat32);
//   auto topk_weights_cont = topk_weights.contiguous().to(torch::kFloat32);
//   auto sorted_token_ids_cont =
//   sorted_token_ids.contiguous().to(torch::kInt32); auto experts_ids_cont =
//   experts_ids.contiguous().to(torch::kInt32);

//   // Handle optional tensors
//   at::Tensor deq_zeros_cont;
//   if (deq_zeros.defined() && deq_zeros.numel() > 0) {
//     deq_zeros_cont = deq_zeros.contiguous().to(torch::kFloat32);
//   }

//   at::Tensor bias_cont;
//   if (bias.defined() && bias.numel() > 0) {
//     bias_cont = bias.contiguous().to(torch::kFloat32);
//   }

//   int32_t num_tokens_post_pad_val = num_tokens_post_pad.item<int32_t>();

//   // Ensure output tensor is contiguous and float32
//   auto c_cont = c.contiguous().to(torch::kFloat32);
//   c_cont.zero_();  // Zero out the output tensor

//   int64_t block_num = num_tokens_post_pad_val / block_size;

//   if (block_num <= 0) {
//     return;
//   }

//   // Pre-scan valid tokens in each block - exactly like CPU implementation
//   std::vector<int64_t> block_valid_offsets(block_num, 0);
//   int64_t offset = 0;

//   for (int64_t b = 0; b < block_num; b++) {
//     if (b >= experts_ids_cont.size(0) ||
//         experts_ids_cont[b].item<int32_t>() < 0) {
//       block_valid_offsets[b] = offset;
//       if (b < experts_ids_cont.size(0) &&
//           experts_ids_cont[b].item<int32_t>() < 0) {
//         offset += 1;
//       }
//       continue;
//     }

//     int64_t valid_count = 0;
//     for (int64_t i = 0; i < block_size; i++) {
//       int64_t index = b * block_size + i;
//       if (index >= sorted_token_ids_cont.size(0)) break;

//       int32_t sorted_id = sorted_token_ids_cont[index].item<int32_t>();
//       if (sorted_id >= token_num * actual_topk) break;
//       valid_count++;
//     }

//     block_valid_offsets[b] = offset;
//     offset += valid_count;
//   }

//   // Main computation loop - match CPU implementation exactly
//   for (int64_t block_id = 0;
//        block_id < block_num && block_id < experts_ids_cont.size(0);
//        block_id++) {
//     int32_t expert_id = experts_ids_cont[block_id].item<int32_t>();
//     if (expert_id < 0 || expert_id >= e) continue;

//     // Extract expert weights: B[expert_id, :, :] -> [N, K]
//     auto expert_weights = b_cont.select(0, expert_id);    // [N, K]
//     auto expert_scale = scale_cont.select(0, expert_id);  // [N]

//     // Extract expert zeros if available
//     at::Tensor expert_zeros;
//     if (deq_zeros_cont.defined()) {
//       expert_zeros = deq_zeros_cont.select(0, expert_id);  // [N] or [K/gs,
//       N]
//     }

//     for (int64_t token_id = 0; token_id < block_size; token_id++) {
//       int64_t sorted_idx = token_id + block_id * block_size;
//       if (sorted_idx >= sorted_token_ids_cont.size(0)) continue;

//       int32_t sorted_id = sorted_token_ids_cont[sorted_idx].item<int32_t>();
//       if (sorted_id < 0 || sorted_id >= token_num * actual_topk) continue;

//       // Calculate cal_id and store_id - exactly like CPU implementation
//       int64_t cal_id = mul_routed_weight
//                            ? (block_valid_offsets[block_id] + token_id)
//                            : (sorted_id / actual_topk);
//       int64_t store_token_id = mul_routed_weight
//                                    ? (sorted_id / actual_topk)
//                                    : (block_valid_offsets[block_id] +
//                                    token_id);
//       int64_t store_expert_id = sorted_id % actual_topk;

//       if (cal_id >= m || store_token_id >= token_num) continue;

//       // Get input token using cal_id
//       auto input_token = a_cont.select(0, cal_id);  // [K]

//       // Perform matrix multiplication using torch.matmul: expert_weights @
//       // input_token -> [N]
//       auto result =
//           torch::matmul(expert_weights, input_token);  // [N, K] x [K] -> [N]

//       // Apply dequantization: result * scale
//       result = result * expert_scale;

//       // Apply zeros if available
//       if (expert_zeros.defined()) {
//         if (expert_zeros.dim() == 1) {  // [N] shape
//           result = result - expert_zeros;
//         } else if (expert_zeros.dim() == 2) {  // [K/gs, N] shape - grouped
//                                                // dequant
//           // For grouped dequantization, apply zeros per group
//           int64_t groups = expert_zeros.size(0);
//           for (int64_t g = 0; g < groups; g++) {
//             auto group_zero = expert_zeros.select(0, g);  // [N]
//             result = result - group_zero;
//           }
//         }
//       }

//       // Add bias if available (bias is shared across experts)
//       if (bias_cont.defined()) {
//         result = result + bias_cont;
//       }

//       // Store result in output tensor using store_token_id and
//       store_expert_id if (store_token_id < c_cont.size(0) && store_expert_id
//       < c_cont.size(1)) {
//         c_cont.select(0, store_token_id)
//             .select(0, store_expert_id)
//             .copy_(result);
//       }

//       // Apply routed weight if needed
//       if (mul_routed_weight) {
//         int64_t token_idx = sorted_id / actual_topk;
//         int64_t expert_idx = sorted_id % actual_topk;
//         int64_t weight_idx = token_idx * actual_topk + expert_idx;
//         if (weight_idx < topk_weights_cont.numel()) {
//           auto weight = topk_weights_cont.view({-1})[weight_idx];
//           // Apply weight to the stored result
//           auto stored_result =
//               c_cont.select(0, store_token_id).select(0, store_expert_id);
//           stored_result.mul_(weight);
//         }
//       }
//     }
//   }

//   // Copy result back to original C tensor if needed
//   if (!c.is_same(c_cont)) {
//     c.copy_(c_cont.to(c.dtype()));
//   }
// }

// // Second algorithm: W8A8 quantization version
// inline void vllmInvokeFusedMoeNonGatherQuantKernel(
//     at::Tensor& c,        // Output tensor C [M, topk, N] - inplace
//     modification const at::Tensor& a,  // Input tensor A [*, K] (fp16/bf16)
//     const at::Tensor& b,  // Weight tensor B [E, N, K] (int8)
//     const at::Tensor& a_scale,       // Scale for A [E, K] or [M, 1] or [1,
//     1] const at::Tensor& scale,         // Scale for w8a8 quant [E, N] (fp32)
//     const at::Tensor& bias,          // Bias tensor [E, N] (fp32)
//     const at::Tensor& topk_weights,  // Top-k expert weights (fp32)
//     const at::Tensor& topk_ids,      // Top-k expert indices (int32)
//     const at::Tensor& sorted_token_ids,  // Sorted token indices (int32)
//     const at::Tensor& experts_ids,       // Expert index for each block
//     (int32) const at::Tensor&
//         num_tokens_post_pad,           // Number of tokens after padding [1]
//     const at::Tensor& real_token_num,  // Actual number of valid tokens [1]
//     bool mul_routed_weight,            // Flag for topk_weights participation
//     int64_t topk,                      // Number of experts for each token
//     int64_t block_size                 // Block size
// ) {
//   // Get input shapes - match CPU implementation exactly
//   auto a_sizes = a.sizes();
//   auto b_sizes = b.sizes();

//   int64_t a_numel = a.numel();
//   int64_t k = a_sizes[a_sizes.size() - 1];  // Last dimension is K
//   int64_t m = a_numel / k;                  // Total number of tokens
//   int64_t e = b_sizes[0];                   // Number of experts
//   int64_t n = b_sizes[1];                   // Output feature dimension

//   // Calculate token_num from topk_weights shape
//   auto topk_weights_sizes = topk_weights.sizes();
//   int64_t token_num = topk_weights_sizes[0];
//   int64_t actual_topk = topk_weights_sizes[1];

//   // Validate inputs
//   if (k == 0 || e == 0 || n == 0 || token_num == 0 || actual_topk == 0) {
//     return;
//   }

//   // Get tensors on same device and contiguous - avoid unnecessary device
//   // transfers
//   auto a_cont = a.contiguous().to(torch::kFloat32).view({m, k});
//   auto b_cont = b.contiguous().to(torch::kFloat32);
//   auto a_scale_cont = a_scale.contiguous().to(torch::kFloat32);
//   auto scale_cont = scale.contiguous().to(torch::kFloat32);
//   auto topk_weights_cont = topk_weights.contiguous().to(torch::kFloat32);
//   auto sorted_token_ids_cont =
//   sorted_token_ids.contiguous().to(torch::kInt32); auto experts_ids_cont =
//   experts_ids.contiguous().to(torch::kInt32);

//   // Handle optional tensors
//   at::Tensor bias_cont;
//   if (bias.defined() && bias.numel() > 0) {
//     bias_cont = bias.contiguous().to(torch::kFloat32);
//   }

//   int32_t num_tokens_post_pad_val = num_tokens_post_pad.item<int32_t>();

//   // Ensure output tensor is contiguous and float32
//   auto c_cont = c.contiguous().to(torch::kFloat32);
//   c_cont.zero_();  // Zero out the output tensor

//   int64_t block_num = num_tokens_post_pad_val / block_size;

//   if (block_num <= 0) {
//     return;
//   }

//   // Pre-scan valid tokens in each block - exactly like CPU implementation
//   std::vector<int64_t> block_valid_offsets(block_num, 0);
//   int64_t offset = 0;

//   for (int64_t b = 0; b < block_num; b++) {
//     if (b >= experts_ids_cont.size(0) ||
//         experts_ids_cont[b].item<int32_t>() < 0) {
//       block_valid_offsets[b] = offset;
//       if (b < experts_ids_cont.size(0) &&
//           experts_ids_cont[b].item<int32_t>() < 0) {
//         offset += 1;
//       }
//       continue;
//     }

//     int64_t valid_count = 0;
//     for (int64_t i = 0; i < block_size; i++) {
//       int64_t index = b * block_size + i;
//       if (index >= sorted_token_ids_cont.size(0)) break;

//       int32_t sorted_id = sorted_token_ids_cont[index].item<int32_t>();
//       if (sorted_id >= token_num * actual_topk) break;
//       valid_count++;
//     }

//     block_valid_offsets[b] = offset;
//     offset += valid_count;
//   }

//   // Determine AScale mode
//   bool is_per_expert_channel =
//       (a_scale_cont.dim() == 2 && a_scale_cont.size(0) == e);
//   bool is_per_token = (a_scale_cont.dim() == 2 && a_scale_cont.size(0) == m);

//   // Main computation loop - match CPU implementation exactly
//   for (int64_t block_id = 0;
//        block_id < block_num && block_id < experts_ids_cont.size(0);
//        block_id++) {
//     int32_t expert_id = experts_ids_cont[block_id].item<int32_t>();
//     if (expert_id < 0 || expert_id >= e) continue;

//     // Extract expert weights and scales
//     auto expert_weights = b_cont.select(0, expert_id);      // [N, K]
//     auto expert_w_scale = scale_cont.select(0, expert_id);  // [N]

//     // Extract expert bias if available
//     at::Tensor expert_bias;
//     if (bias_cont.defined()) {
//       expert_bias = bias_cont.select(0, expert_id);  // [N]
//     }

//     // Get A scale for this expert
//     at::Tensor expert_a_scale;
//     if (is_per_expert_channel) {
//       expert_a_scale = a_scale_cont.select(0, expert_id);  // [K]
//     } else if (is_per_token) {
//       // Will be handled per token
//     } else {
//       expert_a_scale =
//           a_scale_cont.view({-1}).slice(0, 0, 1);  // scalar as tensor
//     }

//     for (int64_t token_id = 0; token_id < block_size; token_id++) {
//       int64_t sorted_idx = token_id + block_id * block_size;
//       if (sorted_idx >= sorted_token_ids_cont.size(0)) continue;

//       int32_t sorted_id = sorted_token_ids_cont[sorted_idx].item<int32_t>();
//       if (sorted_id < 0 || sorted_id >= token_num * actual_topk) continue;

//       // Calculate cal_id and store_id - exactly like CPU implementation
//       int64_t cal_id = mul_routed_weight
//                            ? (block_valid_offsets[block_id] + token_id)
//                            : (sorted_id / actual_topk);
//       int64_t store_token_id = mul_routed_weight
//                                    ? (sorted_id / actual_topk)
//                                    : (block_valid_offsets[block_id] +
//                                    token_id);
//       int64_t store_expert_id = sorted_id % actual_topk;

//       if (cal_id >= m || store_token_id >= token_num) continue;

//       // Get input token using cal_id
//       auto input_token = a_cont.select(0, cal_id);  // [K]

//       // Get A scale for this token
//       at::Tensor token_a_scale;
//       if (is_per_token) {
//         token_a_scale = a_scale_cont.select(0, cal_id);
//       } else if (is_per_expert_channel) {
//         token_a_scale = expert_a_scale;
//       } else {
//         token_a_scale = expert_a_scale;
//       }

//       // For W8A8 quantization, we need to simulate the CPU's approach more
//       // closely
//       at::Tensor result;
//       if (is_per_expert_channel) {
//         // Per-channel A scale: CPU版本有特殊处理，直接计算而不做int8量化
//         result = torch::zeros({n}, input_token.options());
//         // Get original B in int8 format for this calculation
//         auto B_int8_cont = B.contiguous().to(torch::kInt8);
//         auto expert_weights_int8 =
//             B_int8_cont.select(0, expert_id);  // [N, K] int8

//         for (int64_t n = 0; n < N; n++) {
//           float scaled_sum = 0.0f;
//           for (int64_t k = 0; k < K; k++) {
//             float a_val = input_token[k].item<float>();
//             float w_val = static_cast<float>(
//                 expert_weights_int8.select(0, n)[k].item<int8_t>());
//             float a_scale_val = token_a_scale[k].item<float>();
//             scaled_sum += a_val * w_val * a_scale_val;
//           }
//           result[n] = scaled_sum;
//         }
//       } else {
//         // Scalar or per-token A scale: 模拟真正的int8量化过程
//         auto B_int8_cont = B.contiguous().to(torch::kInt8);
//         auto expert_weights_int8 =
//             B_int8_cont.select(0, expert_id);  // [N, K] int8

//         // 量化输入A到int8
//         at::Tensor input_quantized = torch::zeros({K}, torch::kInt8);
//         float a_scale_val = token_a_scale.numel() == 1
//                                 ? token_a_scale.item<float>()
//                                 : token_a_scale[0].item<float>();

//         for (int64_t k = 0; k < K; k++) {
//           float quantized_val = input_token[k].item<float>() / a_scale_val;
//           // 模拟fp32_to_int8_cpu函数
//           int32_t i32 = static_cast<int32_t>(std::round(quantized_val));
//           if (i32 > 127)
//             i32 = 127;
//           else if (i32 < -127)
//             i32 = -127;
//           input_quantized[k] = static_cast<int8_t>(i32);
//         }

//         // 执行int8矩阵乘法
//         result = torch::zeros({N}, torch::kFloat32);
//         for (int64_t n = 0; n < N; n++) {
//           int32_t sum = 0;
//           for (int64_t k = 0; k < K; k++) {
//             sum += static_cast<int32_t>(input_quantized[k].item<int8_t>()) *
//                    static_cast<int32_t>(
//                        expert_weights_int8.select(0, n)[k].item<int8_t>());
//           }

//           // 反量化：result * a_scale
//           float dequant_result = static_cast<float>(sum) * a_scale_val;
//           result[n] = dequant_result;
//         }
//       }

//       // Apply weight scale
//       result = result * expert_w_scale;

//       // Add bias if available
//       if (expert_bias.defined()) {
//         result = result + expert_bias;
//       }

//       // Store result in output tensor using store_token_id and
//       store_expert_id if (store_token_id < c_cont.size(0) && store_expert_id
//       < c_cont.size(1)) {
//         c_cont.select(0, store_token_id)
//             .select(0, store_expert_id)
//             .copy_(result);
//       }

//       // Apply routed weight if needed
//       if (mul_routed_weight) {
//         int64_t token_idx = sorted_id / actual_topk;
//         int64_t expert_idx = sorted_id % actual_topk;
//         int64_t weight_idx = token_idx * actual_topk + expert_idx;
//         if (weight_idx < topk_weights_cont.numel()) {
//           auto weight = topk_weights_cont.view({-1})[weight_idx];
//           // Apply weight to the stored result
//           auto stored_result =
//               c_cont.select(0, store_token_id).select(0, store_expert_id);
//           stored_result.mul_(weight);
//         }
//       }
//     }
//   }

//   // Copy result back to original C tensor if needed
//   if (!c.is_same(c_cont)) {
//     c.copy_(c_cont.to(c.dtype()));
//   }
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// inline void extsSiluMulPerTokenGroupQuant(at::Tensor& out, at::Tensor& scale,
//                                           const at::Tensor& in,
//                                           const at::Tensor& size,
//                                           const int32_t group_size) {
//   // Get real size
//   int32_t real_size = size.item<int32_t>();

//   // Get tensor dimensions
//   auto sizes = in.sizes();

//   // Calculate dimensions following the same logic as CPU implementation
//   int dim0 = real_size;
//   int dim1;
//   int lowest_dim_index = sizes.size() - 1;

//   for (int i = 1; i < lowest_dim_index; i++) {
//     dim0 *= sizes[i];
//   }
//   dim1 = sizes[lowest_dim_index];

//   int out_dim1 = dim1 / 2;
//   int out_elem_num = dim0 * out_dim1;
//   int group_num = out_elem_num / group_size;

//   // Convert input to float32 for computation (like CPU implementation)
//   auto in_f32 = in.to(at::kFloat);

//   // Split input tensor into two halves along the last dimension
//   auto in1 = in_f32.narrow(-1, 0, out_dim1);         // First half
//   auto in2 = in_f32.narrow(-1, out_dim1, out_dim1);  // Second half

//   // Apply SiLU to first half: x * sigmoid(x)
//   auto silu_result = in1 * at::sigmoid(in1);

//   // Multiply with second half
//   auto silu_mul = silu_result * in2;

//   // Only process the real_size portion along the first dimension
//   auto silu_mul_real = silu_mul.narrow(0, 0, real_size);

//   // Flatten for group processing - this should match the CPU implementation
//   // exactly
//   auto silu_mul_flat = silu_mul_real.contiguous().view(-1);

//   // Verify we have the expected number of elements
//   if (silu_mul_flat.numel() != out_elem_num) {
//     throw std::runtime_error("Element count mismatch in native
//     implementation");
//   }

//   // Reshape to groups for scale computation
//   auto silu_mul_groups = silu_mul_flat.view({group_num, group_size});

//   // Compute absolute values and find max per group
//   auto abs_values = at::abs(silu_mul_groups);
//   auto group_max = std::get<0>(at::max(abs_values, /*dim=*/1));

//   // Determine FP8 max value based on output tensor type
//   float fp8_max;
//   if (out.scalar_type() == at::kFloat8_e4m3fn) {
//     fp8_max = 448.0f;
//   } else {
//     fp8_max = 57344.0f;  // e5m2
//   }

//   // Compute scales
//   auto scales = group_max / fp8_max;
//   scale.copy_(scales);

//   // Apply quantization: divide by scale
//   auto scales_expanded = scales.unsqueeze(1).expand({group_num, group_size});
//   auto quantized = silu_mul_groups / scales_expanded;

//   // Flatten back to 1D
//   auto quantized_flat = quantized.view(-1);

//   // Convert to fp8 - this is the critical step
//   // We need to ensure the conversion happens correctly
//   auto quantized_fp8 = quantized_flat.to(out.scalar_type());

//   // Copy to output tensor - ensure we're copying to the right location
//   auto out_real = out.narrow(0, 0, real_size);
//   auto out_flat = out_real.contiguous().view(-1);
//   out_flat.copy_(quantized_fp8);
// }
// /**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// namespace detail {

// // Helper function to perform per-token group quantization using torch
// // operations
// inline std::tuple<at::Tensor, at::Tensor> perTokenGroupQuantizeFP8Native(
//     const at::Tensor& input, int32_t group_size) {
//   auto input_sizes = input.sizes();
//   int64_t n = input_sizes[0];
//   int64_t d = input_sizes[1];

//   TORCH_CHECK(d % group_size == 0,
//               "Hidden dimension must be divisible by group_size");

//   int64_t group_num = d / group_size;

//   // Reshape to [n, group_num, group_size] for per-group processing
//   auto reshaped = input.view({n, group_num, group_size});

//   // Find max absolute value per group: [n, group_num, 1]
//   auto abs_values = at::abs(reshaped);
//   auto group_max =
//       std::get<0>(at::max(abs_values, /*dim=*/-1, /*keepdim=*/true));

//   // Determine FP8 max based on output type - for now assume E4M3
//   constexpr float E4M3_MAX = 448.0f;
//   constexpr float E5M2_MAX = 57344.0f;
//   float fp8_max = E4M3_MAX;  // Default to E4M3

//   // Compute scales: scale = group_max / fp8_max
//   auto scales = group_max / fp8_max;

//   // Avoid division by zero
//   auto safe_scales = at::where(scales > 0, scales, at::ones_like(scales));

//   // Quantize: quantized = input / scale
//   auto quantized = reshaped / safe_scales;

//   // Reshape back to original shape
//   auto output = quantized.view(input_sizes);

//   // Return both quantized values and scales
//   // Note: scales shape is [n, group_num]
//   return std::make_tuple(output, scales.squeeze(-1));
// }

// }  // namespace detail

// /**
//  * @brief Fused Add and RMS Normalization Dyn Per Token Group Quantize FP8
//  * Native implementation.
//  *
//  * @param output          Output tensor of shape [N, hidden_size].
//  * @param residual_update Update residual tensor of shape [N, hidden_size].
//  * @param scale           Scale tensor of shape [N, hidden_size /
//  group_size].
//  * @param input           Input tensor of shape [N, hidden_size].
//  * @param residual        Residual tensor of shape [N, hidden_size].
//  * @param weight          Weight tensor of shape [hidden_size].
//  * @param epsilon         Value added to denominator for numerical stability.
//  * @param group_size      The group size used for quantization.
//  */
// inline void vllmFusedAddRmsNormPerTokenGroupQuantFp8(
//     at::Tensor& output, at::Tensor& residual_update, at::Tensor& scale,
//     const at::Tensor& input, const at::Tensor& residual,
//     const at::Tensor& weight, float epsilon, int32_t group_size) {
//   // Convert all inputs to float32 for computation
//   auto input_f32 = input.to(at::kFloat);
//   auto residual_f32 =
//       residual.numel() > 0 ? residual.to(at::kFloat) : at::Tensor();
//   auto weight_f32 = weight.numel() > 0 ? weight.to(at::kFloat) :
//   at::Tensor();

//   // Step 1: Fused Add - Add input and residual
//   at::Tensor x;
//   at::Tensor residual_update_computed;
//   if (residual_f32.numel() > 0) {
//     x = input_f32 + residual_f32;
//     residual_update_computed = x.clone();  // Store the updated residual
//   } else {
//     x = input_f32;
//     residual_update_computed = at::Tensor();  // Empty tensor
//   }

//   // Step 2: RMS Normalization
//   // variance = mean(x^2, dim=-1, keepdim=True)
//   auto variance = at::mean(x * x, /*dim=*/-1, /*keepdim=*/true);

//   // Apply RMS normalization: x = x * rsqrt(variance + epsilon)
//   x = x * at::rsqrt(variance + epsilon);

//   // Step 3: Apply weight scaling if provided
//   if (weight_f32.numel() > 0) {
//     x = x * weight_f32;
//   }

//   // Step 4: Dynamic Per-Token Group Quantization FP8
//   auto input_sizes = x.sizes();
//   int64_t n = 1;
//   for (int i = 0; i < input_sizes.size() - 1; ++i) {
//     n *= input_sizes[i];
//   }
//   int64_t d = input_sizes[input_sizes.size() - 1];

//   TORCH_CHECK(d % group_size == 0,
//               "Hidden dimension must be divisible by group_size");

//   int64_t group_num = d / group_size;

//   // Reshape to [n, group_num, group_size] for per-group processing
//   auto x_reshaped = x.view({n, group_num, group_size});

//   // Find max absolute value per group: [n, group_num, 1]
//   auto abs_values = at::abs(x_reshaped);
//   auto group_max =
//       std::get<0>(at::max(abs_values, /*dim=*/-1, /*keepdim=*/true));

//   // Determine FP8 max values
//   constexpr float E4M3_MAX = 448.0f;
//   constexpr float E5M2_MAX = 57344.0f;
//   float fp8_max = E4M3_MAX;  // Default to E4M3 for now

//   // Compute scales: scale = group_max / fp8_max
//   auto scales = group_max / fp8_max;

//   // Avoid division by zero
//   auto safe_scales = at::where(scales > 0, scales, at::ones_like(scales));

//   // Quantize: quantized = x / scale
//   auto quantized = x_reshaped / safe_scales;

//   // Reshape back to original shape
//   auto quantized_output = quantized.view(input_sizes);

//   // Reshape scales to [n, group_num]
//   auto output_scales = scales.squeeze(-1);

//   // Copy results to output tensors
//   output.copy_(quantized_output.to(output.scalar_type()));

//   // Copy residual update if provided
//     if (residual_update_computed.numel() > 0 && residual_update.num/**
//  * Copyright 2025 Enflame. All Rights Reserved.
//  */

// // Helper function for stable topk to ensure deterministic results
// inline std::tuple<at::Tensor, at::Tensor> stableTopkMultiDim(
//     const at::Tensor& input, int64_t k, int64_t dim = -1, bool largest =
//     true) {
//     // Parameter validation
//     TORCH_CHECK(input.numel() > 0, "Input tensor cannot be empty");
//     TORCH_CHECK(input.dim() >= 1,
//                 "Input tensor must have at least 1 dimension");

//     dim = dim < 0 ? input.dim() + dim : dim;
//     TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
//     TORCH_CHECK(dim == input.dim() - 1, "Only support topk on last
//     dimension");

//     auto original_shape = input.sizes();
//     int64_t num_dim = original_shape[dim];
//     k = std::min(k, num_dim);

//     // Use PyTorch's built-in topk which handles device placement correctly
//     auto topk_result = torch::topk(input, k, dim, largest, /*sorted=*/true);
//     auto values = std::get<0>(topk_result);
//     auto indices = std::get<1>(topk_result);

//     return std::make_tuple(values, indices);
// }

// /**
//  * @brief Performs a grouped TopK expert selection operation for
//  *        sparse gating in mixture-of-experts models (Native torch
//  implementation with bias).
//  *
//  * @param topk_weights          Output tensor for TopK weights,
//  *                              with shape [num_tokens, topk] and data type
//  FP32.
//  * @param topk_ids              Output tensor for TopK indices,
//  *                              with shape [num_tokens, topk] and data type
//  INT32.
//  * @param gating                Input gating output tensor,
//  *                              with shape [num_tokens, num_experts] and
//  *                              data type FP16/BF16.
//  * @param topk                  Number of experts to select per token
//  *                              (must satisfy topk <= num_experts).
//  * @param renormalize           Whether to perform L1 normalization on the
//  *                              TopK weights.
//  * @param num_expert_group      Number of expert groups
//  *                              (num_experts is divided into this many
//  groups).
//  * @param topk_group            Number of TopK groups to select per token
//  *                              (must satisfy topk_group <=
//  num_expert_group).
//  * @param e_score_correction_bias   Expert score correction bias tensor,
//  *                              with shape [num_experts] (optional)
//  * @param scoring_func          Scoring function to use: "softmax" or
//  "sigmoid"
//  *
//  * @return std::tuple<at::Tensor, at::Tensor> - TopK weights and indices
//  */
// inline std::tuple<at::Tensor, at::Tensor> vllmGroupedTopk(
//     const at::Tensor& gating,
//     const int32_t topk, const bool renormalize, const int32_t
//     num_expert_group, const int32_t topk_group, const at::Tensor&
//     e_score_correction_bias, const char* scoring_func = "softmax") {
//     // Parameter validation
//     std::string scoring_func_str(scoring_func);
//     TORCH_CHECK(scoring_func_str == "softmax" || scoring_func_str ==
//     "sigmoid",
//                 "Unsupported scoring function: ", scoring_func_str);

//     using namespace torch::indexing;

//     // Compute scores using torch operations
//     at::Tensor scores;
//     if (scoring_func_str == "softmax") {
//       scores = torch::softmax(gating, -1);
//     } else {
//       scores = torch::sigmoid(gating);
//     }

//     // Handle score correction bias
//     at::Tensor original_scores;
//     if (e_score_correction_bias.defined()) {
//       original_scores = scores.clone();
//       scores = scores + e_score_correction_bias.unsqueeze(0);
//     }

//     const int64_t num_token = scores.size(0);
//     at::Tensor group_scores;

//     // Compute group scores using torch operations
//     if (e_score_correction_bias.defined()) {
//       // Use top-2 sum for each group: [num_token, num_expert_group,
//       // experts_per_group] -> [num_token, num_expert_group, 2] ->
//       [num_token,
//       // num_expert_group]
//       auto grouped_scores = scores.view({num_token, num_expert_group, -1});
//       auto result = stableTopkMultiDim(grouped_scores, 2, -1, true);
//       auto top2 = std::get<0>(result);
//       group_scores = top2.sum(-1);
//     } else {
//       // Use max score for each group: [num_token, num_expert_group,
//       // experts_per_group] -> [num_token, num_expert_group]
//       auto grouped_scores = scores.view({num_token, num_expert_group, -1});
//       group_scores = std::get<0>(grouped_scores.max(-1));
//     }

//     // Select topk groups using torch operations
//     auto group_idx =
//         std::get<1>(stableTopkMultiDim(group_scores, topk_group, -1, true));
//     auto group_mask = torch::zeros_like(group_scores, torch::kBool);
//     group_mask.scatter_(-1, group_idx, 1);

//     // Generate score mask using torch operations
//     auto score_mask = group_mask.unsqueeze(-1)
//                           .expand({num_token, num_expert_group,
//                                    scores.size(-1) / num_expert_group})
//                           .reshape(scores.sizes());

//     // Apply mask (set non-selected groups to -inf) using torch operations
//     auto tmp_scores = scores.masked_fill(
//         ~score_mask, -std::numeric_limits<float>::infinity());

//     // Select topk experts using torch operations
//     at::Tensor topk_weights, topk_ids;
//     if (e_score_correction_bias.defined()) {
//       // Use corrected scores for selection, but original scores for weights
//       auto result = stableTopkMultiDim(tmp_scores, topk, -1, true);
//       topk_ids = std::get<1>(result);
//       topk_weights = original_scores.gather(-1, topk_ids);
//     } else {
//       // Use scores directly for both selection and weights
//       auto topk_result = stableTopkMultiDim(tmp_scores, topk, -1, true);
//       topk_weights = std::get<0>(topk_result);
//       topk_ids = std::get<1>(topk_result);
//     }

//     // Renormalize if requested using torch operations
//     if (renormalize) {
//       auto sum_weights = topk_weights.sum(-1, true);
//       topk_weights = topk_weights / sum_weights;
//     }

//     // Convert to appropriate types (FP32 for weights, INT32 for indices)
//     return std::make_tuple(topk_weights.to(torch::kFloat32),
//                            topk_ids.to(torch::kInt32));
// }

// /**
//  * @brief Performs a grouped TopK expert selection operation for
//  *        sparse gating in mixture-of-experts models (Native torch
//  implementation without bias).
//  *
//  * @param topk_weights          Output tensor for TopK weights,
//  *                              with shape [num_tokens, topk] and data type
//  FP32.
//  * @param topk_ids              Output tensor for TopK indices,
//  *                              with shape [num_tokens, topk] and data type
//  INT32.
//  * @param gating                Input gating output tensor,
//  *                              with shape [num_tokens, num_experts] and
//  *                              data type FP16/BF16.
//  * @param topk                  Number of experts to select per token
//  *                              (must satisfy topk <= num_experts).
//  * @param renormalize           Whether to perform L1 normalization on the
//  *                              TopK weights.
//  * @param num_expert_group      Number of expert groups
//  *                              (num_experts is divided into this many
//  groups).
//  * @param topk_group            Number of TopK groups to select per token
//  *                              (must satisfy topk_group <=
//  num_expert_group).
//  * @param scoring_func          Scoring function to use: "softmax" or
//  "sigmoid"
//  *
//  * @return std::tuple<at::Tensor, at::Tensor> - TopK weights and indices
//  */
// inline std::tuple<at::Tensor, at::Tensor> vllmGroupedTopk(
//     const at::Tensor& gating,
//     const int32_t topk, const bool renormalize, const int32_t
//     num_expert_group, const int32_t topk_group, const char* scoring_func =
//     "softmax") {
//     // Create empty bias tensor for the overloaded function
//     at::Tensor empty_bias;
//     return vllmGroupedTopk(gating, topk, renormalize, num_expert_group,
//                            topk_group, empty_bias, scoring_func);
// }

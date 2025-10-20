/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mha_fwd_kvcache_mla.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include <tuple>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

// #define CHECK_DEVICE(x) TORCH_CHECK(x.is_gcu(), #x " must be on GCU")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace vllm_gcu::llm_ops {

std::tuple<at::Tensor, at::Tensor> mha_fwd_kvcache_mla(
    at::Tensor &q, const at::Tensor &kcache, const int64_t head_size_v,
    const at::Tensor &seqlens_k, const at::Tensor &block_table,
    const double softmax_scale, bool is_causal,
    const at::Tensor &tile_scheduler_metadata, const at::Tensor &num_splits,
    const c10::optional<at::Tensor> &descale_q,
    const c10::optional<at::Tensor> &descale_k) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size = sizes[3];
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

  if (seqlen_q == 1) {
    is_causal = false;
  }

  auto q_dtype = q.dtype();
  auto opts = q.options();
  caffe2::TypeMeta out_type;
  if (q_dtype == torch::kFloat8_e4m3fn) {
    out_type = torch::kBFloat16;
  } else {
    out_type = q_dtype;
  }
  at::Tensor out = torch::empty(
    {batch_size, seqlen_q, num_heads_q, head_size_v}, opts.dtype(out_type));
  at::Tensor softmax_lse =
      torch::empty({batch_size, num_heads_q, seqlen_q}, opts.dtype(at::kFloat));

  at::Scalar head_size_v_scalar(head_size_v);
  at::Scalar softmax_scale_scalar(softmax_scale);

  std::vector<at::Tensor> out_vector = {out, softmax_lse};

  auto k_dtype = kcache.dtype();
  if (k_dtype == torch::kFloat8_e4m3fn) {
    TORCH_CHECK(descale_k.has_value(),
                "descale_k must be provided for k_dtype fp8_e4m3fn");
    TORCH_CHECK(descale_k.value().dtype() == torch::kFloat);

    at::Tensor descale_q_tensor;
    if (descale_q.has_value()) {
      descale_q_tensor = descale_q.value();
      if (descale_q_tensor.dim() == 0) {
        descale_q_tensor = descale_q_tensor.unsqueeze(0);
      }
    } else {
      // quant inside attention
      TORCH_CHECK(q_dtype == torch::kBFloat16 || q_dtype == torch::kHalf);
    }

    at::Tensor descale_k_tensor = descale_k.value();
    if (descale_k_tensor.dim() == 0) {
      descale_k_tensor = descale_k_tensor.unsqueeze(0);
    }

    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFwdKvcacheMla)(
        out_vector, q, kcache, head_size_v_scalar, seqlens_k, block_table,
        softmax_scale_scalar, is_causal, tile_scheduler_metadata, num_splits,
        descale_q_tensor, descale_k_tensor, stream));
  } else {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmFwdKvcacheMla)(
        out_vector, q, kcache, head_size_v_scalar, seqlens_k, block_table,
        softmax_scale_scalar, is_causal, tile_scheduler_metadata, num_splits,
        stream));
  }

  return {out, softmax_lse};
}
}  // namespace vllm_gcu::llm_ops

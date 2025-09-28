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

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kBFloat16 || q_dtype == torch::kHalf ||
              q_dtype == torch::kFloat8_e4m3fn);

  if (q_dtype == torch::kFloat8_e4m3fn) {
    TORCH_CHECK(descale_q.has_value(),
                "descale_q must be provided for q_dtype fp8_e4m3fn");
    TORCH_CHECK(descale_q.value().dtype() == torch::kFloat);
  }
  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  auto k_dtype = kcache.dtype();
  if (k_dtype == torch::kFloat8_e4m3fn) {
    TORCH_CHECK(descale_k.has_value(),
                "descale_k must be provided for k_dtype fp8_e4m3fn");
    TORCH_CHECK(descale_k.value().dtype() == torch::kFloat);
  }
  TORCH_CHECK(kcache.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  // CHECK_DEVICE(block_table);
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must have dtype torch.int32");
  TORCH_CHECK(block_table.stride(-1) == 1,
              "block_table must have contiguous last dimension");

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  const int seqlen_q_ori = sizes[1];
  const int num_heads_q = sizes[2];
  const int head_size = sizes[3];
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

  const int max_num_blocks_per_seq = block_table.size(1);
  const int num_blocks = kcache.size(0);
  const int page_block_size = kcache.size(1);
  const int num_heads_k = kcache.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      num_heads_q % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  if (seqlen_q_ori == 1) {
    is_causal = false;
  }

  const int num_q_heads_per_hk = num_heads_q / num_heads_k;
  const int seqlen_q = seqlen_q_ori * num_q_heads_per_hk;
  const int num_heads = num_heads_k;
  q = q.view({batch_size, seqlen_q_ori, num_heads_k, num_q_heads_per_hk,
              head_size})
          .transpose(2, 3)
          .reshape({batch_size, seqlen_q, num_heads, head_size});
  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);

  int head_size_k = head_size;
  CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
  CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

  TORCH_CHECK(seqlens_k.dtype() == torch::kInt32,
              "seqlens_k must have dtype int32");
  // CHECK_DEVICE(seqlens_k);
  CHECK_CONTIGUOUS(seqlens_k);
  CHECK_SHAPE(seqlens_k, batch_size);

  auto opts = q.options();
  caffe2::TypeMeta out_type;
  if (q_dtype == torch::kFloat8_e4m3fn) {
    out_type = torch::kBFloat16;
  } else {
    out_type = q_dtype;
  }

  at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v},
                                opts.dtype(out_type));
  at::Tensor softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

  at::Scalar head_size_v_scalar(head_size_v);
  at::Scalar softmax_scale_scalar(softmax_scale);

  std::vector<at::Tensor> out_vector = {out, softmax_lse};

  if (k_dtype == torch::kFloat8_e4m3fn) {
    at::Tensor descale_q_tensor;
    if (descale_q.has_value()) {
      descale_q_tensor = descale_q.value();
      if (descale_q_tensor.dim() == 0) {
        descale_q_tensor = descale_q_tensor.unsqueeze(0);
      }
    }
    descale_q_tensor = descale_q_tensor
                           .view({batch_size, seqlen_q_ori, num_heads_k,
                                  num_q_heads_per_hk, 1})
                           .transpose(2, 3)
                           .reshape({batch_size, seqlen_q, num_heads, 1});

    at::Tensor descale_k_tensor;
    if (descale_k.has_value()) {
      descale_k_tensor = descale_k.value();
      if (descale_k_tensor.dim() == 0) {
        descale_k_tensor = descale_k_tensor.unsqueeze(0);
      }
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

  out = out.view({batch_size, seqlen_q_ori, num_q_heads_per_hk, num_heads_k,
                  head_size_v})
            .transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size_v});
  softmax_lse =
      softmax_lse
          .view({batch_size, num_heads_k, seqlen_q_ori, num_q_heads_per_hk})
          .transpose(2, 3)
          .reshape({batch_size, num_heads_q, seqlen_q_ori});

  return {out, softmax_lse};
}
}  // namespace vllm_gcu::llm_ops

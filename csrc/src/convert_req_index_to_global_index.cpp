/*
 * Copyright 2026 Enflame. All Rights Reserved.
 *
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

#include "convert_req_index_to_global_index.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void convert_req_index_to_global_index(
    at::Tensor& output, const at::Tensor& req_id, const at::Tensor& block_table,
    const at::Tensor& token_indices,
    const c10::optional<at::Tensor>& prefill_workspace_request_ids,
    const c10::optional<at::Tensor>& prefill_workspace_starts,
    int64_t block_size, int64_t num_topk_tokens, int64_t block_n,
    bool has_prefill_workspace, const c10::optional<at::Tensor>& seq_lens) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  at::Tensor prefill_workspace_request_ids_tensor;
  at::Tensor prefill_workspace_starts_tensor;
  if (has_prefill_workspace) {
    assert(prefill_workspace_request_ids.has_value());
    assert(prefill_workspace_starts.has_value());
    prefill_workspace_request_ids_tensor =
        prefill_workspace_request_ids.value();
    prefill_workspace_starts_tensor = prefill_workspace_starts.value();
  }
  at::Tensor seq_lens_tensor;
  if (seq_lens.has_value()) {
    seq_lens_tensor = seq_lens.value();
  }

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmConvertReqIndexToGlobalIndex)(
          output, req_id, block_table, token_indices,
          prefill_workspace_request_ids_tensor, prefill_workspace_starts_tensor,
          seq_lens_tensor, block_size, num_topk_tokens, block_n,
          has_prefill_workspace, -1, stream));
}

}  // namespace vllm_gcu::llm_ops

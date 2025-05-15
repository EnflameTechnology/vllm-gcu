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
#include "fused_grouped_topk.h"
#include <topsaten/topsaten_vllm.h>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"
namespace vllm_gcu::llm_ops {
void fused_grouped_topk(at::Tensor &topk_weights, at::Tensor &topk_ids,
            const at::Tensor &gating_output, const int64_t topk, 
            bool renormalize, const int64_t num_expert_group, 
            const int64_t topk_group, 
            const at::Tensor &e_score_correction_bias,
            c10::string_view scoring_func) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(gating_output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (gating_output.numel() == 0) return;

  const char *scoring_func_name = scoring_func.data();
  if(!e_score_correction_bias.defined()) {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGroupedTopk)(
      topk_weights, topk_ids, gating_output, topk, renormalize,
      num_expert_group, topk_group, scoring_func_name, stream));
  } else {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmGroupedTopk)(
      topk_weights, topk_ids, gating_output, topk, renormalize,
      num_expert_group, topk_group, e_score_correction_bias,
      scoring_func_name, stream));
  }
}
}  // namespace vllm_gcu::llm_ops

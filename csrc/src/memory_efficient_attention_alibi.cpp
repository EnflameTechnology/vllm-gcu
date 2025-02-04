/*
 * Copyright 2024 Enflame. All Rights Reserved.

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

#include "memory_efficient_attention_alibi.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void memory_efficient_attention_alibi(at::Tensor &output,
                                      const at::Tensor &query,
                                      const at::Tensor &key,
                                      const at::Tensor &value,
                                      const at::Tensor &alibi_slopes,
                                      double dropout_p, double scale) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(query));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Tensor attn_bias;
  at::Scalar scale_scalar(scale);
  at::Scalar dropout_p_scalar(dropout_p);
  at::Scalar mask_mode = 2;
  at::Scalar sliding_window = 0;

  ATEN_ATENOP_CALL(topsvllm::topsvllmMemoryEfficientAttentionV1)
  (output, query, key, value, attn_bias, dropout_p_scalar, scale_scalar,
   mask_mode, alibi_slopes, sliding_window, stream);
}

}  // namespace vllm_gcu::llm_ops

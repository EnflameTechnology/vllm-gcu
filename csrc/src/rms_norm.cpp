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

#include "rms_norm.h"

#include <topsaten/topsaten_te.h>
#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void rms_norm(at::Tensor &out, const at::Tensor &input,
              const at::Tensor &weight, double epsilon) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (input.numel() == 0) return;

  at::Tensor view_out = out.view({-1, out.size(-1)});
  at::Tensor view_input = input.view({-1, input.size(-1)});
  at::Scalar xeps(epsilon);

  at::Tensor device_weight;
  if (!weight.device().is_privateuseone()) {
    device_weight = weight.to(at::kPrivateUse1);
  } else {
    device_weight = weight;
  }

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmRmsNorm)(
      view_out, view_input, device_weight, xeps, stream));
}

}  // namespace vllm_gcu::llm_ops

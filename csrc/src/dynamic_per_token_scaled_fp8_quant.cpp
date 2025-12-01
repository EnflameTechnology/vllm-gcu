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

#include "dynamic_per_token_scaled_fp8_quant.h"
#include <topsaten/topsaten_vllm.h>
#include <tuple>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void dynamic_per_token_scaled_fp8_quant(
    at::Tensor &result, const at::Tensor &input, at::Tensor &scale,
    const ::std::optional<at::Tensor> &scale_ub) {
        if (result.numel() == 0) return;
        const torch_gcu::OptionalGCUGuard device_guard(device_of(result));
        const topsStream_t stream = torch_gcu::getCurrentGCUStream();

        ATEN_ATENOP_CHECK(
            ATEN_ATENOP_CALL(topsvllm::topsvllmDynamicPerTokenScaledFP8Quant)(
                result, scale, input, stream));
    }

}  // namespace vllm_gcu::llm_ops

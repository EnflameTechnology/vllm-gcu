/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "rms_norm_per_token_group_quant_fp8.h"

#include <topsaten/topsaten_vllm.h>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

    void rms_norm_per_token_group_quant_fp8(at::Tensor &out, at::Tensor &scale,
        const at::Tensor &input,
        const at::Tensor &weight,
        double epsilon,
        int64_t group_size) {
    const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();

    ATEN_ATENOP_CHECK(
        ATEN_ATENOP_CALL(topsvllm::topsvllmRmsNormPerTokenGroupQuantFp8)(
            out, scale, input, weight, epsilon, group_size, stream));
}

} // namespace vllm_gcu::llm_ops

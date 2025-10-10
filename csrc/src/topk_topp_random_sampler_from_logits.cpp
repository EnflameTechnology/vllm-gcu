/**
* Copyright 2024 Enflame. All Rights Reserved.
*/
#include "topk_topp_random_sampler_from_logits.h"
#include <topsaten/topsaten_vllm.h>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"
namespace vllm_gcu::llm_ops {
    void topk_topp_random_sampler_from_logits(
                    at::Tensor &output_token_ids,
                    const at::Tensor &logits,
                    const c10::optional<at::Tensor> &k,
                    const c10::optional<at::Tensor> &p,
                    const at::Tensor &exponential,
                    const int64_t dim = -1) {
        const torch_gcu::OptionalGCUGuard device_guard(device_of(logits));
        const topsStream_t stream = torch_gcu::getCurrentGCUStream();
        at::Tensor k_tensor;
        if (k.has_value()) {
            k_tensor = k.value();
        }

        at::Tensor p_tensor;
        if (p.has_value()) {
            p_tensor = p.value();
        }

        ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
            topsvllm::topsvllmTopkToppRandomSamplerFromLogits)(
            output_token_ids, logits, k_tensor,
            p_tensor, exponential, dim, stream));
    }
} // namespace vllm_gcu::llm_ops

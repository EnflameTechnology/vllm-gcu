/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "rms_norm_per_token_group_quant.h"

#include <topsaten/topsaten_vllm.h>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

    void rms_norm_per_token_group_quant(at::Tensor &out, at::Tensor &scale,
        const at::Tensor &input,
        const at::Tensor &weight,
        double epsilon,
        int64_t group_size) {
}

} // namespace vllm_gcu::llm_ops

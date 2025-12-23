/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "apply_repetition_penalties.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/tops/Context.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void apply_repetition_penalties(at::Tensor& logits,
                                const at::Tensor& prompt_mask,
                                const at::Tensor& output_mask,
                                const at::Tensor& repetition_penalties) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(logits));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  if (get_device_capability() == 3) {
    at::Tensor repetition_penalties_repeat =
        repetition_penalties.unsqueeze(1).repeat({1, logits.size(1)});

    auto penalties = torch::where(prompt_mask.logical_or(output_mask),
                                    repetition_penalties_repeat, 1.0);

    auto scaling = torch::where(logits > 0, 1.0 / penalties, penalties);
    logits.mul_(scaling);
  } else {
    ATEN_ATENOP_CHECK(
        ATEN_ATENOP_CALL(topsvllm::topsvllmApplyRepetitionPenalties)(
        logits, prompt_mask, output_mask, repetition_penalties));
  }
}

}  // namespace vllm_gcu::llm_ops

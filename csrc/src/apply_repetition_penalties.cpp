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
  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmApplyRepetitionPenalties)(
      logits, prompt_mask, output_mask, repetition_penalties, stream));
}
}  // namespace vllm_gcu::llm_ops

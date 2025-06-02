/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "advance_step_flashattn.h"
#include <stdexcept>

#include <topsaten/topsaten_vllm.h>

#include "advance_step_xformers.h"
#include "tops_extension/tops/Context.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu {
namespace llm_ops {

void advance_step_flashattn(int64_t num_seqs, int64_t num_queries,
                            int64_t block_size, at::Tensor& input_tokens,
                            at::Tensor& sampled_token_ids,
                            at::Tensor& input_positions, at::Tensor& seq_lens,
                            at::Tensor& slot_mapping,
                            at::Tensor& block_tables) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(input_tokens));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  int major = get_device_capability();

  if (major == 3) {
    advance_step_xformers(num_seqs, num_queries, block_size, input_tokens,
                          sampled_token_ids, input_positions, seq_lens,
                          slot_mapping, block_tables);
  } else if (major == 4) {
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmAdvanceStepFlashAttn)(
        num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
        input_positions, seq_lens, slot_mapping, block_tables, stream));
  } else {
    throw std::runtime_error("Not support platform");
  }
}

}  // namespace llm_ops
}  // namespace vllm_gcu

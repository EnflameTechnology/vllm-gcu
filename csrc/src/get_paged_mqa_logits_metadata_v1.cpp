/* Copyright 2026 Enflame. All Rights Reserved. */

#include "get_paged_mqa_logits_metadata_v1.h"

#include <topsaten/topsaten_deepgemm.h>
#include <torch/all.h>

#include <tuple>
#include <vector>

#include "fp8_paged_mqa_logits_v1.h"
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

at::Tensor get_paged_mqa_logits_metadata_v1(const at::Tensor& context_lens,
                                            int64_t block_kv, int64_t num_sms,
                                            int64_t threshold) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(context_lens));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  auto out = torch::empty({num_sms + 1, 2},
                          context_lens.options().dtype(torch::kInt32));

  at::Scalar block_kv_scalar(block_kv);
  at::Scalar num_sms_scalar(num_sms);
  at::Scalar threshold_scalar(threshold);

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsdeepgemm::topsdeepgemmGetPagedMqaLogitsMetaData_V1)(
          out, context_lens, block_kv_scalar, num_sms_scalar, threshold_scalar,
          stream));

  return out;
}

}  // namespace vllm_gcu::llm_ops

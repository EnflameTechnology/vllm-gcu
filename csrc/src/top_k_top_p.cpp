/**
* Copyright 2024 Enflame. All Rights Reserved.
*/
#include "top_k_top_p.h"

#include <topsaten/topsaten_ops.h>
#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void top_k_top_p(at::Tensor &logits, const at::Tensor &k, const at::Tensor &p,
                  const int64_t dim = -1, const bool descending = false) {
   const torch_gcu::OptionalGCUGuard device_guard(device_of(logits));
   const topsStream_t stream = torch_gcu::getCurrentGCUStream();

   ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenTop_k_top_p)(
       logits, k, p, dim, descending, stream));
}
}
// namespace vllm_gcu::llm_ops


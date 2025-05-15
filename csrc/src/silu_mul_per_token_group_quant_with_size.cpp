/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "silu_mul_per_token_group_quant_with_size.h"

#include <topsaten/topsaten_extensions.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void silu_mul_per_token_group_quant_with_size(at::Tensor &out,
                                             at::Tensor &scale,
                                             const at::Tensor &input,
                                             const at::Tensor &size,
                                             int64_t group_size) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  if (input.numel() == 0) return;

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsSiluMulPerTokenGroupQuant)(
          out, scale, input, size, group_size, stream));
}

} // namespace vllm_gcu::llm_ops

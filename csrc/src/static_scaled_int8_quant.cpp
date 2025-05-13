/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "static_scaled_int8_quant.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void static_scaled_int8_quant(at::Tensor& output, const at::Tensor& input,
                              const at::Tensor& scales,
                              const ::std::optional<at::Tensor>& azp) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  const at::Tensor scales_dtype = scales.reciprocal()
                                      .to(input.dtype())
                                      .unsqueeze(0);

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenQuantize)(
      output, input, scales_dtype, stream));
}

}  // namespace vllm_gcu::llm_ops

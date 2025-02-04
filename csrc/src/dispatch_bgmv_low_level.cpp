/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "dispatch_bgmv_low_level.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void dispatch_bgmv_low_level(at::Tensor &y, const at::Tensor &x,
                             const at::Tensor &w, const at::Tensor &indicies,
                             const int64_t layer_idx, const double scale,
                             const int64_t h_in, const int64_t h_out,
                             const int64_t y_offset) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(x));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsvllm::topsvllmBgmvLowLevel)(
      y, y, x, w, indicies, layer_idx, scale, h_in, h_out, y_offset, stream));
}

}  // namespace vllm_gcu::llm_ops
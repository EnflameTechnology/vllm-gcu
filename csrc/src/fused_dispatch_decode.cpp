/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "fused_dispatch_decode.h"

#include <topsaten/topsaten_extensions.h>
#include <vector>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void fused_dispatch_decode(at::TensorList outputs,
                         const at::Tensor &recv_packed,
                         const at::Tensor &sp_split_size,
                         at::IntArrayRef split_sizes) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(outputs[0]));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();


  topsatenSize_t topsaten_split_sizes(split_sizes.data(),
                                   static_cast<int64_t>(split_sizes.size()));

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsexts::topsextsFusedDispatchDecode)(
          outputs, recv_packed, sp_split_size,
          topsaten_split_sizes, stream));
}

} // namespace vllm_gcu::llm_ops

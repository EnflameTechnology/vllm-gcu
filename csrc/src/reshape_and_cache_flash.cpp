/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "reshape_and_cache_flash.h"

#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu {
namespace llm_ops {

void reshape_and_cache_flash(const at::Tensor& key,
                            const at::Tensor& value,
                            at::Tensor& key_cache,
                            at::Tensor& value_cache,
                           const at::Tensor& slot_mapping) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(key_cache));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  ATEN_ATENOP_CHECK(
      ATEN_ATENOP_CALL(topsvllm::topsvllmReshapeAndCacheFlash)(
          key_cache, value_cache, key, value, slot_mapping, stream));
}

} // namespace llm_ops
} // namespace vllm_gcu

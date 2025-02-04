#include <tops/tops_runtime.h>
#include <torch/all.h>

#include <map>

#include "cache_ops.h"

namespace vllm_gcu::llm_ops {

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  topsMemcpyKind memcpy_type;
  if (src_device.is_privateuseone() && dst_device.is_privateuseone()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GCU");
    memcpy_type = topsMemcpyDeviceToDevice;
  } else if (src_device.is_privateuseone() && dst_device.is_cpu()) {
    memcpy_type = topsMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_privateuseone()) {
    memcpy_type = topsMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const c10::OptionalDeviceGuard device_guard(src_device);
  const size_t pair_num = block_mapping.size(0);
  for (size_t pair = 0; pair < pair_num; ++pair) {
    int64_t src_block_number = block_mapping[pair][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[pair][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    topsMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type);
  }
}

void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping) {
  TORCH_CHECK(key_caches.size() == value_caches.size());
  for (size_t i = 0; i < key_caches.size(); i++) {
    auto k = key_caches[i];
    auto v = value_caches[i];
    const size_t pair_num = block_mapping.size(0);
    for (size_t pair = 0; pair < pair_num; ++pair) {
      int64_t src = block_mapping[pair][0].item<int64_t>();
      int64_t dst = block_mapping[pair][1].item<int64_t>();
      k[dst] = k[src];
      v[dst] = v[src];
    }
  }
}

}  // namespace vllm_gcu::llm_ops

/**
 * Copyright 2025 Enflame. All Rights Reserved.
 */
#include "rotary_embedding_with_kv_cache.h"

#include <topsaten/topsaten_extensions.h>

#include <tuple>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void rotary_embedding_with_kv_cache_gcu(
    at::Tensor &q_out, at::Tensor &kv_cache,
    const c10::optional<at::Tensor> &k_pe_out,
    const c10::optional<at::Tensor> &k_c_normed, const at::Tensor &q,
    const at::Tensor &kv, const at::Tensor &positions,
    const at::Tensor &cos_sin_cache, const at::Tensor &weight,
    const at::Tensor &slot_mapping, const at::Tensor &scale_tensor, double eps,
    at::IntArrayRef split_size, const char *kv_dtype) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(q_out));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  topsatenSize_t topsaten_split_sizes(split_size.data(),
                                      static_cast<int64_t>(split_size.size()));

  auto view_q = q.view({-1, q.size(-1)});
  auto view_q_out = q_out.view({-1, q_out.size(-1)});
  auto view_positions = positions.view({-1});

  if (k_pe_out.has_value()) {
    assert(k_c_normed.has_value());
    at::Tensor k_pe_out_tensor;
    at::Tensor k_c_normed_tensor;
    k_pe_out_tensor = k_pe_out.value();
    k_pe_out_tensor = k_pe_out_tensor.view({-1, k_pe_out_tensor.size(-1)});
    k_c_normed_tensor = k_c_normed.value();
    k_c_normed_tensor =
        k_c_normed_tensor.view({-1, k_c_normed_tensor.size(-1)});
    ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(
        topsexts::topsextsRotaryEmbeddingWithKVCache)(
        view_q_out, kv_cache, k_pe_out_tensor, k_c_normed_tensor, view_q, kv,
        view_positions, cos_sin_cache, weight, slot_mapping, scale_tensor,
        static_cast<double>(eps), topsaten_split_sizes, kv_dtype, stream));
  } else {
    ATEN_ATENOP_CHECK(
        ATEN_ATENOP_CALL(topsexts::topsextsRotaryEmbeddingWithKVCache)(
            view_q_out, kv_cache, view_q, kv, view_positions, cos_sin_cache,
            weight, slot_mapping, scale_tensor, static_cast<double>(eps),
            topsaten_split_sizes, kv_dtype, stream));
  }
}

void rotary_embedding_with_kv_cache(
    at::Tensor &q_out, at::Tensor &kv_cache,
    const c10::optional<at::Tensor> &k_pe_out,
    const c10::optional<at::Tensor> &k_c_normed, const at::Tensor &q,
    const at::Tensor &kv, const at::Tensor &positions,
    const at::Tensor &cos_sin_cache, const at::Tensor &weight,
    const at::Tensor &slot_mapping, const at::Tensor &scale, double eps,
    at::IntArrayRef split_size, c10::string_view kv_cache_dtype) {
  const char *kv_dtype = kv_cache_dtype.data();
  at::Tensor scale_tensor = scale;
  if (scale.dim() == 0) {
    scale_tensor = scale.unsqueeze(0);
  }

#ifndef NDEBUG
  auto fallback_ops = c10::utils::get_env("VLLM_GCU_FALLBACK_CPU");
  bool is_fallback = false;
  at::Tensor q_out_cpu, kv_cache_cpu, q_cpu, kv_cpu, positions_cpu;
  at::Tensor cos_sin_cache_cpu, weight_cpu, slot_mapping_cpu, scale_tensor_cpu;

  if (fallback_ops.has_value()) {
    if (fallback_ops->find("rotary_embedding_with_kv_cache") !=
            std::string::npos ||
        (*fallback_ops) == "all") {
      is_fallback = true;

      // Log fallback CPU usage
      VLLM_FALLBACK_CPU_LOG("rotary_embedding_with_kv_cache",
                            "Using CPU fallback implementation");

      // Convert tensors to CPU for native implementation
      q_out_cpu = q_out.to(at::kCPU);
      kv_cache_cpu = kv_cache.to(at::kCPU);
      q_cpu = q.to(at::kCPU);
      kv_cpu = kv.to(at::kCPU);
      positions_cpu = positions.to(at::kCPU);
      cos_sin_cache_cpu = cos_sin_cache.to(at::kCPU);
      weight_cpu = weight.to(at::kCPU);
      slot_mapping_cpu = slot_mapping.to(at::kCPU);
      scale_tensor_cpu = scale_tensor.to(at::kCPU);

      // Convert split_size to vector for native call
      std::vector<int64_t> split_size_vec(split_size.begin(), split_size.end());

      // Call native implementation on CPU tensors
      extsRotaryEmbeddingWithKVCache(
          q_out_cpu, kv_cache_cpu, q_cpu, kv_cpu, positions_cpu,
          cos_sin_cache_cpu, weight_cpu, slot_mapping_cpu, scale_tensor_cpu,
          eps, split_size_vec, std::string(kv_cache_dtype));

      VLLM_FALLBACK_CPU_LOG("rotary_embedding_with_kv_cache",
                            "CPU fallback computation completed");
    }
  }
#endif

  rotary_embedding_with_kv_cache_gcu(
      q_out, kv_cache, k_pe_out, k_c_normed, q, kv, positions, cos_sin_cache,
      weight, slot_mapping, scale_tensor, eps, split_size, kv_dtype);

#ifndef NDEBUG
  if (is_fallback) {
    const topsStream_t stream = torch_gcu::getCurrentGCUStream();
    topsStreamSynchronize(stream);

    VLLM_FALLBACK_CPU_LOG("rotary_embedding_with_kv_cache",
                          "Starting result verification");

    auto cpu_output = std::make_tuple(q_out_cpu, kv_cache_cpu);
    auto device_outputs =
        std::make_tuple(q_out.to(at::kCPU), kv_cache.to(at::kCPU));
    EXPECT_TRUE(extsRotaryEmbeddingWithKVCacheCheck(cpu_output, device_outputs),
                "rotary_embedding_with_kv_cache");
    q_out.copy_(q_out_cpu);
    kv_cache.copy_(kv_cache_cpu);

    VLLM_FALLBACK_CPU_LOG("rotary_embedding_with_kv_cache",
                          "Fallback CPU results copied back to device");
  }
#endif
}

}  // namespace vllm_gcu::llm_ops

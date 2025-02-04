
#pragma once

#include <ATen/ATen.h>

namespace vllm_gcu::llm_ops {

void multi_query_kv_attention_v2(at::Tensor &output, const at::Tensor &query,
                                 const at::Tensor &key, const at::Tensor &value,
                                 c10::optional<const at::Tensor> attn_bias,
                                 //  const at::Tensor &attn_bias,
                                 float dropout_p,  // const float dropout_p,
                                 //  c10::optional<const float> scale,
                                 float scale,
                                 const at::ArrayRef<int32_t> &seqlens);

}  // namespace vllm_gcu::llm_ops

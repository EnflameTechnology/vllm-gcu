
#include "multi_query_kv_attention_v2.h"

#include <topsaten/topsaten_vllm.h>
#include <torch/all.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void multi_query_kv_attention_v2(at::Tensor &output, const at::Tensor &query,
                                 const at::Tensor &key, const at::Tensor &value,
                                 c10::optional<const at::Tensor> attn_bias,
                                 float dropout_p, float scale,
                                 const at::ArrayRef<int32_t> &seqlens) {
  // std::cout << "topsvllmMemEfficientAttention" << " :\n"
  //           << "output: " << tensorToString(output) << "\n"
  //           << "query: " << tensorToString(query) << "\n"
  //           << "key: " << tensorToString(key) << "\n"
  //           << "value: " << tensorToString(value) << "\n"
  //           << "attn_bias: " << (attn_bias
  //                               ? tensorToString(*attn_bias)
  //                               : "undefine tensor") << "\n"
  //           << "dropout_p: " << xdropout_p << "\n"
  //           << "scale: " << (scale ? std::to_string(*scale) : "none") <<
  //           "\n"
  //           << "seqlens: " << seqlens << "\n"
  //           << "stream: " << (topsStream_t)stream << "\n";
  const torch_gcu::OptionalGCUGuard device_guard(device_of(query));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  at::Scalar scale_scalar(scale);
  at::Scalar dropout_p_scalar(dropout_p);

  at::Tensor attn_bias_tensor;
  if (attn_bias.has_value()) {
    attn_bias_tensor = attn_bias.value();
  }
  ATEN_ATENOP_CALL(topsvllm::topsvllmMemEfficientAttention)
  (output, query, key, value, attn_bias_tensor, dropout_p_scalar, scale_scalar,
   seqlens.vec(), stream);
}
}  // namespace vllm_gcu::llm_ops

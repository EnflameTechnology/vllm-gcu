/**
 * Copyright 2024 Enflame. All Rights Reserved.
 */
#include "layer_norm_static_int8_quant.h"

#include <ATen/AccumulateType.h>
#include <topsaten/topsaten_vllm.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {
void layer_norm_static_int8_quant(at::Tensor &output, const at::Tensor &input,
                                  const at::Tensor &scaling,
                                  at::IntArrayRef normalized_shape,
                                  const c10::optional<at::Tensor> &weight_opt,
                                  const c10::optional<at::Tensor> &bias_opt,
                                  double epsilon) {
  const torch_gcu::OptionalGCUGuard device_guard(device_of(output));
  const topsStream_t stream = torch_gcu::getCurrentGCUStream();
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor &weight = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor &bias = *bias_maybe_owned;
  at::Scalar scalar_epsilon(epsilon);
  topsatenSize_t xshape = {normalized_shape.data(),
                           static_cast<int64_t>(normalized_shape.size())};
  auto input_shape = input.sizes();
  std::vector<int64_t> stat_shape;
  const size_t axis = input.dim() - normalized_shape.size();
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  at::Tensor mean = at::empty(stat_shape, input.options().dtype(acc_type));
  at::Tensor rstd = at::empty(stat_shape, input.options().dtype(acc_type));
  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topsaten::topsatenNativeLayerNormQuant)(
      output, mean, rstd, input, scaling, xshape, weight, bias, scalar_epsilon,
      stream));
}

}  // namespace vllm_gcu::llm_ops

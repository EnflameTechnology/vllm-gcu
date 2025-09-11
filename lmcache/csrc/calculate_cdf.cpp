/*
 * Copyright 2022-2025 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <topslmc/topslmc.h>
#include <torch/torch.h>

#include "tops_extension/torch/GCUAten.h"
#include "torch_gcu.h"  // NOLINT(build/include_subdir)

namespace lmcache {

#define MAX_BINS_SUPPORTED 64

at::Tensor calculate_cdf(const at::Tensor& input, int64_t max_bins) {
  TORCH_CHECK(input.is_privateuseone(), "Input must be a device tensor");
  TORCH_CHECK(max_bins < MAX_BINS_SUPPORTED, "Max bins must be less than ",
              MAX_BINS_SUPPORTED);

  const topsStream_t stream = torch_gcu::getCurrentGCUStream();

  const auto input_shape = input.sizes();
  const int nlayers = input_shape[0];
  const int nchannels = input_shape[2];
  auto output = torch::zeros({nlayers, nchannels, max_bins + 1},
                             input.options().dtype(at::kShort));

  ATEN_ATENOP_CHECK(ATEN_ATENOP_CALL(topslmc::topslmcCalculateCdf)(
      output, input, max_bins, stream));

  return output;
}

}  // namespace lmcache

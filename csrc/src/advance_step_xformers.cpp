/*
 * Copyright 2021-2025 Enflame. All Rights Reserved.

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

#include "advance_step_xformers.h"

#include <torch/all.h>

#include "torch_gcu.h"

namespace vllm_gcu::llm_ops {

void launch_advance_step_kernel(int64_t num_queries, int64_t block_size,
                                int *seq_lens_ptr, int *slot_mapping_ptr,
                                int *block_tables_ptr,
                                int64_t const block_table_stride,
                                int64_t num_blocks, int64_t num_threads);

inline void verify_tensor(std::string const &name, torch::Tensor &t,
                          int64_t const size_0, int64_t const size_1,
                          c10::ScalarType const type) {
  bool size_0_cond = true;
  if (size_0 != -1) {
    size_0_cond = t.size(0) == size_0;
  }

  bool size_1_cond = true;
  if (size_1 != -1) {
    size_1_cond = t.size(1) == size_1;
  }

  bool is_contiguous = t.is_contiguous();
  bool same_type = t.dtype() == type;

  bool pass = size_0_cond && size_1_cond && is_contiguous && same_type;
  if (!pass) {
    TORCH_CHECK(false, "tensor: name = ", name, ", shape = ", t.sizes(),
                " is_cont = ", t.is_contiguous(), ", type = ", t.dtype(),
                " is not as expected: shape = [", size_0, ", ", size_1,
                "], type = ", type);
  }
}

void advance_step_xformers(
    int64_t num_seqs, int64_t num_queries, int64_t block_size,
    at::Tensor &input_tokens,       // type: long, [num_seqs]
    at::Tensor &sampled_token_ids,  // type: long, [num_queries, 1]
    at::Tensor &input_positions,    // type: long, [num_seqs]
    at::Tensor &seq_lens,           // type: int, [num_seqs]
    at::Tensor &slot_mapping,       // type: long, [num_seqs]
    at::Tensor
        &block_tables) {  // type: int, [num_seqs, max_seq_len / block_size]
  int dev = sampled_token_ids.get_device();
  const topsStream_t stream = torch_gcu::getCurrentGCUStream(dev);

  verify_tensor("input_tokens", input_tokens, num_seqs, -1, at::kLong);
  verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,
                at::kLong);
  verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
  verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
  verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kLong);
  verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);

  TORCH_CHECK(num_queries <= num_seqs, "num_queries must less than num_seqs");

  // update input_tokens
  topsMemcpy(input_tokens.data_ptr(), sampled_token_ids.data_ptr(),
             num_queries * sizeof(int),
             topsMemcpyKind::topsMemcpyDeviceToDevice);
  topsMemcpy(input_positions.data_ptr(), seq_lens.data_ptr(),
             num_queries * sizeof(int),
             topsMemcpyKind::topsMemcpyDeviceToDevice);

  launch_advance_step_kernel(num_queries, block_size,
                             reinterpret_cast<int *>(seq_lens.data_ptr()),
                             reinterpret_cast<int *>(slot_mapping.data_ptr()),
                             reinterpret_cast<int *>(block_tables.data_ptr()),
                             block_tables.stride(0), 1, 1);
}

}  // namespace vllm_gcu::llm_ops

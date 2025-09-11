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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

// NOLINTBEGIN
#include "calculate_cdf.h"
#include "decode_cuda_new.h"
#include "decode_cuda_prefsum.h"
#include "encode_cuda_new.h"
#include "load_and_reshape_flash.h"
#include "multi_layer_kv_transfer.h"
#include "multi_layer_kv_transfer_unilateral.h" 
#include "reshape_and_cache_back_flash.h"
#include "rotary_embedding_k_fused.h"
#include "single_layer_kv_transfer.h"
// NOLINTEND

std::string get_gpu_pci_bus_id(int device);
uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);
void free_pinned_ptr(uintptr_t ptr);
uintptr_t alloc_pinned_numa_ptr(size_t size, int node);
void free_pinned_numa_ptr(uintptr_t ptr, size_t size);

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("multi_layer_kv_transfer", &lmcache::multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &lmcache::multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &lmcache::single_layer_kv_transfer);
  m.def("load_and_reshape_flash", &lmcache::load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &lmcache::reshape_and_cache_back_flash);
  m.def("encode_fast_new", &lmcache::encode_cuda_new);
  m.def("decode_fast_new", &lmcache::decode_cuda_new);
  m.def("decode_fast_prefsum", &lmcache::decode_cuda_prefsum);
  m.def("calculate_cdf", &lmcache::calculate_cdf);
  m.def("rotary_embedding_k_fused", &lmcache::rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
}

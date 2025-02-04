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

#pragma once

#include <tops/tops_runtime.h>
#include <tops/topscc_types.h>

#define QACC_SIZE 512
#define VDMEM_SIZE 0x100000
#define ALIGN(a, b) (a / b * b)
#define ALIGNUP(a, b) (((a + b - 1) / b) * b)
#define DIV_CEIL(a, b) ((a + b - 1) / b)

__device__ __forceinline__ int GetBlockNum(void) {
  return (gridDim.x * gridDim.y * gridDim.z);
}

__device__ __forceinline__ int GetBlockIdx(void) {
  return (blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x +
          blockIdx.x);
}

__device__ __forceinline__ int GetThreadNumEachBlock(void) {
  return (blockDim.x * blockDim.y * blockDim.z);
}

__device__ __forceinline__ int GetThreadIdxInBlock(void) {
  return threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x +
         threadIdx.x;
}

__device__ __forceinline__ int GetThreadIdx(void) {
  int blockIdx = GetBlockIdx();
  int threadNumEachBlock = GetThreadNumEachBlock();

  return blockIdx * threadNumEachBlock + GetThreadIdxInBlock();
}

__device__ __forceinline__ int GetThreadNum(void) {
  return GetBlockNum() * GetThreadNumEachBlock();
}

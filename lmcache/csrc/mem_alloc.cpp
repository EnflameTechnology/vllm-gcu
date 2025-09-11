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

#include <errno.h>
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include <sys/mman.h>
#include <sys/syscall.h>
#include <tops/tops_runtime.h>
#include <unistd.h>

#include <cstring>  // for strerror
#include <stdexcept>
#include <string>

std::string get_gpu_pci_bus_id(int device) {
  char pciBusId[13];
  topsError_t err = topsDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
  if (err != topsSuccess) {
    throw std::runtime_error(std::string("topsDeviceGetPCIBusId failed: ") +
                             topsGetErrorString(err));
  }
  return std::string(pciBusId);
}

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = malloc(size);
  topsError_t err = topsHostRegister(ptr, size, flags);
  if (err != topsSuccess) {
    free(ptr);
    throw std::runtime_error(std::string("alloc pinned ptr failed: ") +
                             topsGetErrorString(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  topsError_t err = topsHostUnregister(reinterpret_cast<void*>(ptr));
  if (err != topsSuccess) {
    throw std::runtime_error(std::string("free pinned ptr failed: ") +
                             topsGetErrorString(err));
  }
  free(reinterpret_cast<void*>(ptr));
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);  // NOLINT(runtime/int)
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

// NOLINTBEGIN
static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}
// NOLINTEND


uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;  // NOLINT(runtime/int)
  long maxnode = 8 * sizeof(mask);  // NOLINT(runtime/int)
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  topsError_t st = topsHostRegister(ptr, size, 0);
  if (st != topsSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("topsHostRegister failed: ") +
                             topsGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  topsError_t st = topsHostUnregister(p);
  if (st != topsSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("topsHostUnregister failed: ") +
                             topsGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

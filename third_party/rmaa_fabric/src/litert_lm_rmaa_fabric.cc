// Copyright 2026 RMAA Holdings & The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cinttypes>
#include <cstdio>
#include "litert_lm_rmaa_fabric.h"

#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/prctl.h>
#endif

constexpr uint32_t kNJITMagic = 0x54494A4E; // 'NJIT' in little-endian
constexpr size_t kNJITHeaderSize = 36;
constexpr size_t kHugePageAlignment = 2 * 1024 * 1024; // 2 MiB THP
constexpr size_t kDirectIOAlignment = 4096;            // 4 KiB O_DIRECT

#pragma pack(push, 1)
struct NJITHeader {
  uint32_t magic;           // 0x54494A4E
  uint16_t version;         // 0x0100
  uint16_t tensor_type;     // 1=F16, 2=Q8_0, 3=Q4_0
  uint64_t content_id;      // 64-bit Canonical Content Hash
  uint32_t layers;          // Number of Transformer layers
  uint32_t head_dim;        // Attention head dimension
  uint32_t sequence_length; // Sequence length of pre-computed KV cache
  uint64_t payload_bytes;   // Size of serialized attention tensor
};
#pragma pack(pop)
static_assert(sizeof(NJITHeader) == kNJITHeaderSize, "NJITHeader must be exactly 36 bytes");

struct JITExecutablePage {
  void* code_ptr = nullptr;
  size_t code_size = 0;
};

struct LiteRtLmRmaaFabricInstance {
  LiteRtLmRmaaFabricConfig config;
  std::mutex mutex;
  
  struct MappedFile {
    void* addr = nullptr;
    size_t size = 0;
    int fd = -1;
  };
  std::vector<MappedFile> mappings;

  std::unordered_map<uint64_t, JITExecutablePage> jit_cache;

  uint64_t transmitted_packets = 0;
  uint64_t accepted_tokens = 0;
};

LiteRtLmRmaaFabricConfig litert_lm_rmaa_fabric_config_default(void) {
  LiteRtLmRmaaFabricConfig cfg;
  cfg.enable_smtf_v3_hugepages = true;
  cfg.enable_njit_memory_grafting = true;
  cfg.enable_localjit_dispatch = true;
  cfg.enable_upaas_quic_transport = false;
  cfg.shm_namespace = "/litert_njit_graft";
  cfg.quic_mtu_bytes = 1200;
  cfg.mtp_speculative_gamma = 2;
  return cfg;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_create(
    const LiteRtLmRmaaFabricConfig* config,
    LiteRtLmRmaaFabricHandle* out_handle) {
  if (!out_handle) {
    return kLiteRtLmStatusInvalidArgument;
  }
  
  auto* instance = new (std::nothrow) LiteRtLmRmaaFabricInstance();
  if (!instance) {
    return kLiteRtLmStatusResourceExhausted;
  }

  if (config) {
    instance->config = *config;
  } else {
    instance->config = litert_lm_rmaa_fabric_config_default();
  }

  *out_handle = instance;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_destroy(LiteRtLmRmaaFabricHandle handle) {
  if (!handle) {
    return kLiteRtLmStatusInvalidArgument;
  }

  std::lock_guard<std::mutex> lock(handle->mutex);

  for (auto& m : handle->mappings) {
    if (m.addr && m.addr != MAP_FAILED) {
      munmap(m.addr, m.size);
    }
    if (m.fd >= 0) {
      close(m.fd);
    }
  }
  handle->mappings.clear();

  for (auto& [hash, page] : handle->jit_cache) {
    if (page.code_ptr && page.code_ptr != MAP_FAILED) {
      munmap(page.code_ptr, page.code_size);
    }
  }
  handle->jit_cache.clear();

  delete handle;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_map_smtf_v3(
    LiteRtLmRmaaFabricHandle handle,
    const char* file_path,
    void** out_mapped_base,
    size_t* out_mapped_size) {
  if (!handle || !file_path || !out_mapped_base || !out_mapped_size) {
    return kLiteRtLmStatusInvalidArgument;
  }

  int flags = O_RDONLY;
  int fd = open(file_path, flags);
  if (fd < 0) {
    return kLiteRtLmStatusNotFound;
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    return kLiteRtLmStatusInternal;
  }

  size_t file_size = static_cast<size_t>(st.st_size);

  void* addr = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    return kLiteRtLmStatusResourceExhausted;
  }

#if defined(MADV_HUGEPAGE)
  if (handle->config.enable_smtf_v3_hugepages) {
    madvise(addr, file_size, MADV_HUGEPAGE);
    madvise(addr, file_size, MADV_WILLNEED);
  }
#endif

  {
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->mappings.push_back({addr, file_size, fd});
  }

  *out_mapped_base = addr;
  *out_mapped_size = file_size;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_graft_kv_cache(
    LiteRtLmRmaaFabricHandle handle,
    uint64_t content_id,
    void* engine_kv_cache_ptr,
    size_t max_cache_bytes,
    size_t* out_grafted_sequence_pos) {
  if (!handle || !engine_kv_cache_ptr || !out_grafted_sequence_pos) {
    return kLiteRtLmStatusInvalidArgument;
  }

  if (!handle->config.enable_njit_memory_grafting) {
    return kLiteRtLmStatusUnavailable;
  }

  char shm_path[128];
  snprintf(shm_path, sizeof(shm_path), "%s_%" PRIx64,
           handle->config.shm_namespace ? handle->config.shm_namespace : "/litert_njit_graft",
           content_id);

  int shm_fd = shm_open(shm_path, O_RDONLY, 0666);
  if (shm_fd < 0) {
    return kLiteRtLmStatusNotFound;
  }

  struct stat st;
  if (fstat(shm_fd, &st) < 0 || st.st_size < static_cast<off_t>(sizeof(NJITHeader))) {
    close(shm_fd);
    return kLiteRtLmStatusDataLoss;
  }

  void* shm_map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, shm_fd, 0);
  close(shm_fd);

  if (shm_map == MAP_FAILED) {
    return kLiteRtLmStatusResourceExhausted;
  }

  const auto* header = reinterpret_cast<const NJITHeader*>(shm_map);
  if (header->magic != kNJITMagic || header->content_id != content_id) {
    munmap(shm_map, st.st_size);
    return kLiteRtLmStatusInvalidArgument;
  }

  if (header->payload_bytes > max_cache_bytes) {
    munmap(shm_map, st.st_size);
    return kLiteRtLmStatusResourceExhausted;
  }

  const uint8_t* payload_src = reinterpret_cast<const uint8_t*>(shm_map) + sizeof(NJITHeader);
  std::memcpy(engine_kv_cache_ptr, payload_src, header->payload_bytes);

  *out_grafted_sequence_pos = header->sequence_length;

  munmap(shm_map, st.st_size);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_dispatch_kernel(
    LiteRtLmRmaaFabricHandle handle,
    uint64_t kernel_signature_hash,
    void* execution_context,
    bool* out_cache_hit) {
  if (!handle || !out_cache_hit) {
    return kLiteRtLmStatusInvalidArgument;
  }

  if (!handle->config.enable_localjit_dispatch) {
    *out_cache_hit = false;
    return kLiteRtLmStatusOk;
  }

  std::lock_guard<std::mutex> lock(handle->mutex);
  auto it = handle->jit_cache.find(kernel_signature_hash);
  if (it != handle->jit_cache.end()) {
    typedef void (*KernelFunc)(void*);
    auto func = reinterpret_cast<KernelFunc>(it->second.code_ptr);
    if (func && execution_context) {
      func(execution_context);
    }
    *out_cache_hit = true;
    return kLiteRtLmStatusOk;
  }

  *out_cache_hit = false;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_stream_activations(
    LiteRtLmRmaaFabricHandle handle,
    uint32_t layer_idx,
    const void* bf16_activations,
    size_t tensor_bytes) {
  if (!handle || !bf16_activations || tensor_bytes == 0) {
    return kLiteRtLmStatusInvalidArgument;
  }

  if (!handle->config.enable_upaas_quic_transport) {
    return kLiteRtLmStatusUnavailable;
  }

  const size_t mtu = handle->config.quic_mtu_bytes > 0 ? handle->config.quic_mtu_bytes : 1200;
  size_t offset = 0;
  while (offset < tensor_bytes) {
    size_t chunk_len = std::min(tensor_bytes - offset, mtu);
    offset += chunk_len;
    handle->transmitted_packets++;
  }

  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_rmaa_fabric_verify_mtp_draft(
    LiteRtLmRmaaFabricHandle handle,
    uint64_t seed,
    uint64_t seq_id,
    uint32_t position,
    const int32_t* draft_tokens,
    size_t gamma,
    size_t* out_accepted_count) {
  if (!handle || !draft_tokens || !out_accepted_count) {
    return kLiteRtLmStatusInvalidArgument;
  }

  size_t accepted = 0;
  for (size_t i = 0; i < gamma; ++i) {
    uint64_t state = seed ^ (seq_id + (position + i) * 0x9E3779B97F4A7C15ULL);
    state ^= state >> 30;
    state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27;
    state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;

    if (draft_tokens[i] >= 0) {
      accepted++;
    } else {
      break;
    }
  }

  handle->accepted_tokens += accepted;
  *out_accepted_count = accepted;
  return kLiteRtLmStatusOk;
}

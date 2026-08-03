// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// The KV cache interface including all K and V buffers for a model.
class KVCacheInterface {
 public:
  virtual ~KVCacheInterface() = default;

  // Returns the total number of entries in the KV cache per block.
  virtual int GetNumEntries() const = 0;

  // Returns the batch size of the KV cache.
  virtual int GetBatchSize() const = 0;

  // Serializes the KV cache to a byte string.
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Loads the KV cache from a serialized byte string.
  virtual absl::Status Load(absl::string_view serialized_kv_cache) = 0;

  // Selects a single batch from the other KV cache and copies it to this KV
  // cache.
  // Example:
  //   This has shape [1, ...] and other has shape [3, ...]. Then we can select
  //   batch x from other and copy it to this
  //   (i.e., other[x, :, ...] -> this[0, :, ...]).
  virtual absl::Status SelectAndCopyFrom(KVCacheInterface& other,
                                         int batch_index) = 0;

  // Broadcasts the source KV with batch size 1 to this KV cache with batch size
  // > 1.
  // Example:
  //   This has shape [3, ...] and other has shape [1, ...]. Then we can copy
  //   other[0, :, ...] -> this[0, :, ...], this[1, :, ...], this[2, :, ...].
  virtual absl::Status BroadcastAndCopyFrom(KVCacheInterface& other) = 0;

  // Deep copies the KV cache. This is an expensive operation. Use sparingly.
  virtual absl::StatusOr<std::unique_ptr<KVCacheInterface>> DeepCopy()
      const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_KV_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_KV_CACHE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/kv_cache_interface.h"

namespace litert::lm {

class LitertKVCache : public KVCacheInterface {
 public:
  static absl::StatusOr<std::unique_ptr<LitertKVCache>> Create(
      Environment& env, const Model& model, absl::string_view signature_name,
      CompiledModel& compiled_model, bool inplace_update);

  int GetNumEntries() const override { return num_entries_; };

  int GetBatchSize() const override { return batch_size_; };

  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status Load(absl::string_view serialized_kv_cache) override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status SelectAndCopyFrom(KVCacheInterface& other,
                                 int batch_index) override;

  absl::Status BroadcastAndCopyFrom(KVCacheInterface& other) override;

  absl::StatusOr<std::unique_ptr<KVCacheInterface>> DeepCopy() const override;

  // Resizes the KV cache to the given number of entries (sequence length).
  // Note: Resize is a no-op if the requested size is smaller than the current
  // size.
  absl::Status Resize(int num_entries);

  struct KVCacheBuffers {
    absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  };

  // For backends that support inplace update, this returns a single set of KV
  // cache buffers that can be used for both input and output (i.e,
  // input_buffers and output_buffers point to the same data).
  // For backends that don't support inplace update, this returns two distinct
  // sets of KV cache buffers, one for input and one for output. On each call,
  // the input/output buffers will be swapped.
  absl::StatusOr<KVCacheBuffers> GetKVCacheBuffers();

 private:
  LitertKVCache(
      int batch_size, int num_entries, std::optional<int> k_dynamic_dim,
      std::optional<int> v_dynamic_dim, Environment& env,
      absl::flat_hash_map<std::string, TensorBuffer> bank_1_key_cache_buffers,
      absl::flat_hash_map<std::string, TensorBuffer> bank_1_value_cache_buffers,
      std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
          bank_2_key_cache_buffers,
      std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
          bank_2_value_cache_buffers)
      : batch_size_(batch_size),
        num_entries_(num_entries),
        k_dynamic_dim_(std::move(k_dynamic_dim)),
        v_dynamic_dim_(std::move(v_dynamic_dim)),
        env_(env),
        bank_1_key_cache_buffers_(std::move(bank_1_key_cache_buffers)),
        bank_1_value_cache_buffers_(std::move(bank_1_value_cache_buffers)),
        bank_2_key_cache_buffers_(std::move(bank_2_key_cache_buffers)),
        bank_2_value_cache_buffers_(std::move(bank_2_value_cache_buffers)) {}

  // Batch size of the KV cache buffers.
  int batch_size_;
  // Number of entries in the KV cache.
  int num_entries_;
  // Dynamic dimension index of the KV cache buffers (i.e., sequence dimension).
  std::optional<int> k_dynamic_dim_;
  std::optional<int> v_dynamic_dim_;
  // Environment to create new TensorBuffers (required for resizing).
  Environment& env_;
  // Primary KV cache buffers.
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_key_cache_buffers_;
  absl::flat_hash_map<std::string, TensorBuffer> bank_1_value_cache_buffers_;
  // Secondary KV cache buffers - only used when inplace update is not
  // supported.
  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_key_cache_buffers_;
  std::optional<absl::flat_hash_map<std::string, TensorBuffer>>
      bank_2_value_cache_buffers_;

  bool bank_1_is_input_ = true;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_KV_CACHE_H_

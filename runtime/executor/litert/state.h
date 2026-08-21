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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_

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
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/state_interface.h"
#include "runtime/proto/executor_metadata.pb.h"

namespace litert::lm {

class LitertState : public StateInterface {
 public:
  enum class AllocationPolicy {
    // In-place update. The same buffer is used for both input and output KV
    // cache tensors.
    kInplace = 0,
    // Ping-pong update. Two sets of buffers are used and swapped between
    // inputs and outputs at each step. Required when the backend/model
    // does not support updating buffers in-place.
    kPingPong = 1,
    // GPU-optimized in-place update. Similar to kInplace, but optimized for
    // GPU backends that handle KV cache updates internally. In this mode,
    // KV cache buffers are allocated as GPU buffers (outputs) and we skip
    // passing them as inputs during execution to avoid overhead.
    kGpuOptimizedInplace = 2,
  };

  static absl::StatusOr<std::unique_ptr<LitertState>> Create(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata* executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill = true);

  int GetNumEntries() const override { return num_entries_; }

  int GetBatchSize() const override { return batch_size_; }

  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status Load(absl::string_view serialized_state) override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status SelectAndCopyFrom(StateInterface& other,
                                 int batch_index) override;

  absl::Status BroadcastAndCopyFrom(StateInterface& other) override;

  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const override;

  bool Contains(absl::string_view tensor_name) const override;

  absl::Status Clear() override;

  // Resizes the KV cache to the given number of entries (sequence length).
  // Note: Resize is a no-op if the requested size is smaller than the current
  // size.
  absl::Status Resize(CompiledModel& compiled_model,
                      absl::string_view signature_name, int num_entries);

  struct StateBuffers {
    absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  };

  // For backends that support inplace update, this returns a single set of
  // state cache buffers that can be used for both input and output (i.e,
  // input_buffers and output_buffers point to the same data).
  // For backends that don't support inplace update, this returns two distinct
  // sets of state cache buffers, one for input and one for output. On each
  // call, the input/output buffers will be swapped.
  absl::StatusOr<StateBuffers> GetStateBuffers(
      CompiledModel& compiled_model, absl::string_view signature_name);

 private:
  struct StateBuffer {
    TensorBuffer buffer;
    proto::StateBuffer::Type type = proto::StateBuffer::TYPE_UNSPECIFIED;
    std::optional<int> dynamic_dim;
  };

  absl::Status SyncShapes(CompiledModel& compiled_model,
                          absl::string_view signature_name);

  static absl::StatusOr<std::unique_ptr<LitertState>> HeuristicBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name, AllocationPolicy allocation_policy,
      int batch_size, bool clear_kv_cache_before_prefill);

  static absl::StatusOr<std::unique_ptr<LitertState>> MetadataBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata& executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill);

  LitertState(
      int batch_size, int num_entries, Environment& env,
      absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers,
      std::optional<absl::flat_hash_map<std::string, StateBuffer>>
          bank_2_state_buffers,
      AllocationPolicy allocation_policy)
      : batch_size_(batch_size),
        num_entries_(num_entries),
        env_(env),
        bank_1_state_buffers_(std::move(bank_1_state_buffers)),
        bank_2_state_buffers_(std::move(bank_2_state_buffers)),
        allocation_policy_(allocation_policy) {}

  // Batch size of the KV cache buffers.
  int batch_size_;
  // Number of entries in the KV cache.
  int num_entries_;
  // Environment to create new TensorBuffers (required for resizing).
  Environment& env_;
  // Primary state buffers.
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers_;
  // Secondary state buffers - only used when inplace update is not
  // supported.
  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers_;

  bool bank_1_is_input_ = true;

  AllocationPolicy allocation_policy_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_

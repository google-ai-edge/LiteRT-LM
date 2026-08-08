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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LEGACY_MAP_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LEGACY_MAP_STATE_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/state_interface.h"

namespace litert::lm {

// A simple implementation of StateInterface that holds an owned map of
// TensorBuffers. Used by executors that do not use specialized LiteRT state.
// TODO: mheydary - Deprecate LegacyMapState in favor of specialized state.
class LegacyMapState : public StateInterface {
 public:
  explicit LegacyMapState(
      absl::flat_hash_map<std::string, ::litert::TensorBuffer> buffers)
      : buffers_(std::move(buffers)) {}

  int GetNumEntries() const override { return 0; }
  int GetBatchSize() const override { return 0; }

  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError(
        "Serialize not implemented in LegacyMapState");
  }

  absl::Status Load(absl::string_view serialized_state) override {
    return absl::UnimplementedError("Load not implemented in LegacyMapState");
  }

  absl::Status SelectAndCopyFrom(StateInterface& other,
                                 int batch_index) override {
    return absl::UnimplementedError(
        "SelectAndCopyFrom not implemented in LegacyMapState");
  }

  absl::Status BroadcastAndCopyFrom(StateInterface& other) override {
    return absl::UnimplementedError(
        "BroadcastAndCopyFrom not implemented in LegacyMapState");
  }

  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const override {
    absl::flat_hash_map<std::string, ::litert::TensorBuffer> cloned_buffers;
    cloned_buffers.reserve(buffers_.size());
    for (const auto& [name, buffer] : buffers_) {
      auto buffer_dup = buffer.Duplicate();
      if (!buffer_dup) {
        return absl::InternalError(
            "Failed to duplicate buffer in LegacyMapState");
      }
      cloned_buffers[name] = std::move(*buffer_dup);
    }
    return std::make_unique<LegacyMapState>(std::move(cloned_buffers));
  }

  bool Contains(absl::string_view tensor_name) const override {
    return buffers_.contains(tensor_name);
  }

  absl::Status Clear() override {
    for (auto& [_, buffer] : buffers_) {
      LITERT_RETURN_IF_ERROR(buffer.Clear());
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<std::string, ::litert::TensorBuffer>& buffers() {
    return buffers_;
  }

  const absl::flat_hash_map<std::string, ::litert::TensorBuffer>& buffers()
      const {
    return buffers_;
  }

 private:
  absl::flat_hash_map<std::string, ::litert::TensorBuffer> buffers_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LEGACY_MAP_STATE_H_

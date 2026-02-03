// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LORA_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LORA_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/lora.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/lora_data.h"

namespace litert::lm {

// The class managing LoRA weights for LiteRT-LM.
// It is responsible for loading LoRA weights, creating LoRA objects, and
// managing the current LoRA ID to use.
// It will create LoRA objects on the backend (e.g. GPU) lazily, only when
// UseLoRA() is called with the corresponding LoRA ID.
class LoraManager {
 public:
  // Args:
  // compiled_model: The CompiledModel object containing model and environment
  // information. It is used for creating backend resources for model buffers.
  // It also contains the LoRA input signature information
  static absl::StatusOr<std::unique_ptr<LoraManager>> Create(
      const litert::CompiledModel& compiled_model);

  // Returns the current LoRA ID.
  std::optional<uint32_t> GetCurrentLoRAId() const { return current_lora_id_; }

  // Loads the LoRA model into tensor loader, but does not use it.
  // To use the lora weights, call `UseLoRA()` with the lora_id.
  // Args:
  // lora_id: The unique id to assign to the loaded LoRA model.
  // model_assets: Contains the LoRA model to load.
  absl::Status LoadLoRA(uint32_t lora_id, const ModelAssets& model_assets);

  // Sets the current LoRA ID to use. If the LoRA object for the given ID
  // doesn't exist, it will be created.
  absl::Status UseLoRA(uint32_t lora_id);

  // Clears the current LoRA ID so no LoRA is active.
  void ClearLoRA() { current_lora_id_ = std::nullopt; }

  // Returns a map of all the LoRA tensor names to their duplicated
  // TensorBuffers for the current LoRA ID.
  absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
  GetLoRABuffers() const;

 private:
  explicit LoraManager(const litert::CompiledModel& compiled_model);

  const litert::CompiledModel& compiled_model_;

  absl::flat_hash_map<uint32_t, std::unique_ptr<LoraData>> lora_data_;
  absl::flat_hash_map<uint32_t, std::unique_ptr<LoRA>> loras_;
  std::optional<uint32_t> current_lora_id_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_LORA_MANAGER_H_

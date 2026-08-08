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

#include "runtime/components/lora_manager.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/lora.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/lora_data.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<LoraManager>> LoraManager::Create(
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name) {
  return absl::WrapUnique(new LoraManager(compiled_model, signature_name));
}

LoraManager::LoraManager(const litert::CompiledModel& compiled_model,
                         absl::string_view signature_name)
    : compiled_model_(compiled_model), signature_name_(signature_name) {}

absl::Status LoraManager::LoadLoRA(uint32_t lora_id,
                                   const ModelAssets& model_assets) {
  if (lora_data_.contains(lora_id)) {
    return absl::AlreadyExistsError("LoRA ID already exists");
  }
  ABSL_ASSIGN_OR_RETURN(auto scoped_file, model_assets.GetOrCreateScopedFile());
  ABSL_ASSIGN_OR_RETURN(auto lora_data,
                        LoraData::CreateFromScopedFile(scoped_file));
  lora_data_[lora_id] = std::move(lora_data);
  return absl::OkStatus();
}

absl::Status LoraManager::UseLoRA(uint32_t lora_id) {
  if (!lora_data_.contains(lora_id) && !loras_.contains(lora_id)) {
    return absl::NotFoundError("LoRA ID not found");
  }
  if (!loras_.contains(lora_id)) {
    ABSL_ASSIGN_OR_RETURN(
        auto lora, LoRA::Create(std::move(lora_data_[lora_id]), compiled_model_,
                                signature_name_));
    loras_[lora_id] = std::move(lora);
    lora_data_.erase(lora_id);
  }
  current_lora_id_ = lora_id;
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
LoraManager::GetLoRABuffers() const {
  if (!current_lora_id_.has_value()) {
    return absl::FailedPreconditionError("No LoRA ID is set");
  }
  if (!loras_.contains(*current_lora_id_)) {
    return absl::NotFoundError("LoRA ID not found");
  }
  return loras_.at(*current_lora_id_)->GetLoRABuffers();
}

}  // namespace litert::lm

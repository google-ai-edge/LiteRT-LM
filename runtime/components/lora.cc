#include "runtime/components/lora.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
// TODO: b/467362164 Move tflite_lora_utils to an OSS directory to support open
// sourcing LoRA.
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/lora_data.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

// Names of the signature runners, used to get the signature runners from the
// interpreter.
// TODO: b/450616365 - Consolidate constant definitions.
constexpr char kDecodeSignatureRunner[] = "decode";

}  // namespace

absl::StatusOr<std::unique_ptr<LoRA>> LoRA::Create(
    std::unique_ptr<LoraData> lora_data,
    const litert::CompiledModel& compiled_model) {
  auto lora = absl::WrapUnique(new LoRA(std::move(lora_data), compiled_model));
  RETURN_IF_ERROR(lora->Init());
  return lora;
}

absl::Status LoRA::Init() {
  // Get the input names from the decode signature. When the model has separate
  // LoRA signatures (e.g. "decode_lora_r4"), try those first.
  std::string decode_sig = kDecodeSignatureRunner;

  // Try 1: Use the LoRA rank from LoraData metadata to construct the signature
  // name directly.
  bool found_lora_sig = false;
  if (lora_data_) {
    auto rank = lora_data_->GetLoRARank();
    if (rank.ok() && *rank > 0) {
      std::string lora_sig =
          absl::StrCat(kDecodeSignatureRunner, "_lora_r", *rank);
      auto lora_names = compiled_model_.GetSignatureInputNames(lora_sig);
      if (lora_names.HasValue()) {
        decode_sig = std::move(lora_sig);
        found_lora_sig = true;
      }
    }
  }

  // Try 2: If LoRA metadata didn't provide the rank (e.g. litert-torch
  // generated LoRA files), scan the model's signatures for a decode_lora_r*
  // pattern.
  if (!found_lora_sig) {
    auto sig_keys = compiled_model_.GetSignatureKeys();
    if (sig_keys.HasValue()) {
      constexpr absl::string_view kLoraPrefix = "decode_lora_r";
      for (const auto& key : sig_keys.Value()) {
        if (absl::StartsWith(key, kLoraPrefix)) {
          // Verify this signature actually has LoRA inputs.
          auto names = compiled_model_.GetSignatureInputNames(key);
          if (names.HasValue()) {
            decode_sig = std::string(key);
            break;
          }
        }
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      compiled_model_.GetSignatureInputNames(decode_sig));

  for (const auto& input_name : input_names) {
    if (!IsLoRAInputName(input_name)) {
      continue;
    }
    // Create the input buffer for the LoRA tensor.
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBuffer tensor_buffer,
        compiled_model_.CreateInputBuffer(decode_sig, input_name));

    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create(
                                tensor_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_size,
                                 tensor_buffer.PackedSize());

    if (lora_data_->HasTensor(input_name)) {
      // Read the tensor data from LoraData.
      ASSIGN_OR_RETURN(auto lora_tensor_data,
                       lora_data_->ReadTensor(input_name));

      // Copy the data from LoraData to the TensorBuffer.
      RET_CHECK_EQ(tensor_buffer_size, lora_tensor_data->Size())
          << "LoRA tensor size mismatch between model input and Lora Data: "
          << tensor_buffer_size << " vs. " << lora_tensor_data->Size();
      std::memcpy(lock_and_addr.second, lora_tensor_data->Data(),
                  lora_tensor_data->Size());
    } else {
      // Fill the buffer with zeros if the tensor is not in LoraData.
      std::memset(lock_and_addr.second, 0, tensor_buffer_size);
    }

    lora_buffers_[input_name] = std::move(tensor_buffer);
  }
  return absl::OkStatus();
}

absl::StatusOr<litert::TensorBuffer> LoRA::GetLoRABuffer(
    const std::string& name) const {
  auto it = lora_buffers_.find(name);
  if (it == lora_buffers_.end()) {
    return absl::NotFoundError("LoRA tensor not found.");
  }
  LITERT_ASSIGN_OR_RETURN(auto duplicated_buffer, it->second.Duplicate());
  return duplicated_buffer;
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
LoRA::GetLoRABuffers() const {
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> buffers;
  for (const auto& [name, buffer] : lora_buffers_) {
    LITERT_ASSIGN_OR_RETURN(buffers[name], buffer.Duplicate());
  }
  return buffers;
}

}  // namespace litert::lm

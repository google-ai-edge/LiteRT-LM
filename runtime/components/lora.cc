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

#include "runtime/components/lora.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
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

// Input of GPU models exported with runtime-BMM LoRA. It holds the channel
// range of the LoRA BMM ops (see ml_drift::LlmRuntimeParams) and is not a LoRA
// weight.
constexpr absl::string_view kLoRAParamTensorName = "lora_param_tensor";

}  // namespace

absl::StatusOr<std::unique_ptr<LoRA>> LoRA::Create(
    std::unique_ptr<LoraData> lora_data,
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name) {
  auto lora = absl::WrapUnique(
      new LoRA(std::move(lora_data), compiled_model, signature_name));
  ABSL_RETURN_IF_ERROR(lora->Init());
  return lora;
}

absl::Status LoRA::Init() {
  // Get the input names from the default signature.
  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      compiled_model_.GetSignatureInputNames(signature_name_));

  // LoRA inputs of the model that have no matching tensor in the LoRA data.
  // They are filled with zeros, i.e. the LoRA has no effect on them. Only
  // collected for the signature the LoRA was created with.
  std::vector<std::string> unmatched_input_names;

  // Creates a TensorBuffer for `input_name` of `signature` and fills it with
  // the LoRA weights for that input.
  auto create_filled_buffer =
      [&](absl::string_view signature,
          absl::string_view input_name) -> absl::StatusOr<TensorBuffer> {
    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer tensor_buffer,
        compiled_model_.CreateInputBuffer(signature, input_name));

    LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                            litert::TensorBufferScopedLock::Create(
                                tensor_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_size,
                            tensor_buffer.PackedSize());

    if (lora_data_->HasTensor(input_name)) {
      // Read the tensor data from LoraData.
      ABSL_ASSIGN_OR_RETURN(auto lora_tensor_data,
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
      if (signature == signature_name_) {
        unmatched_input_names.push_back(std::string(input_name));
      }
    }
    return tensor_buffer;
  };

  // GPU models exported with runtime-BMM LoRA read the channel range of every
  // LoRA BMM op from `lora_param_tensor`: element 0 is the start index and the
  // others are end indices. Without this the range is unset, so the LoRA ops
  // compute on garbage ranges (no effect, wrong output or out-of-bounds
  // access). Set the ends to the largest non-rank extent of any LoRA input
  // (all dimensions but the rank, multiplied, since the ops may reshape e.g.
  // [rank, heads, head_dim] to [rank, heads * head_dim]), so that every op
  // covers all of its channels. The rank is the model's LoRA rank (a LoRA with
  // a smaller rank is zero padded to it), i.e. the dimension that every LoRA
  // input has.
  int64_t lora_param_end = 0;
  if (std::find(input_names.begin(), input_names.end(),
                kLoRAParamTensorName) != input_names.end()) {
    std::vector<std::vector<int32_t>> lora_dims;
    for (const auto& input_name : input_names) {
      if (!IsLoRAInputName(input_name)) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          auto tensor_type,
          compiled_model_.GetInputTensorType(signature_name_, input_name));
      const auto dims = tensor_type.Layout().Dimensions();
      lora_dims.emplace_back(dims.begin(), dims.end());
    }
    RET_CHECK(!lora_dims.empty())
        << kLoRAParamTensorName << " is an input but there are no LoRA inputs.";

    // Candidates for the rank: dimensions of the first input that every other
    // input also has. Use the smallest one.
    int32_t model_rank = 0;
    for (int32_t candidate : lora_dims[0]) {
      if (candidate <= 0 || (model_rank > 0 && candidate >= model_rank)) {
        continue;
      }
      if (std::all_of(lora_dims.begin(), lora_dims.end(),
                      [candidate](const std::vector<int32_t>& dims) {
                        return std::find(dims.begin(), dims.end(),
                                         candidate) != dims.end();
                      })) {
        model_rank = candidate;
      }
    }
    RET_CHECK_GT(model_rank, 0)
        << "Could not find a LoRA rank shared by all LoRA inputs.";

    for (const auto& dims : lora_dims) {
      int64_t num_elements = 1;
      for (int32_t dim : dims) {
        num_elements *= dim;
      }
      lora_param_end = std::max(lora_param_end, num_elements / model_rank);
    }
  }

  // Creates a TensorBuffer for `input_name` of `signature`: the channel range
  // for `lora_param_tensor`, the LoRA weights otherwise.
  auto create_buffer =
      [&](absl::string_view signature,
          absl::string_view input_name) -> absl::StatusOr<TensorBuffer> {
    if (input_name != kLoRAParamTensorName) {
      return create_filled_buffer(signature, input_name);
    }
    LITERT_ASSIGN_OR_RETURN(
        TensorBuffer param_buffer,
        compiled_model_.CreateInputBuffer(signature, kLoRAParamTensorName));
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          litert::TensorBufferScopedLock::Create(
              param_buffer, TensorBuffer::LockMode::kWrite));
      LITERT_ASSIGN_OR_RETURN(auto param_size, param_buffer.PackedSize());
      // Element 0 is the start index; at least one end index must follow.
      RET_CHECK_GE(param_size, 2 * sizeof(int32_t))
          << "Unexpected size of " << kLoRAParamTensorName;
      int32_t* params = static_cast<int32_t*>(lock_and_addr.second);
      std::fill(params, params + param_size / sizeof(int32_t),
                static_cast<int32_t>(lora_param_end));
      params[0] = 0;
    }
    return param_buffer;
  };

  // Whether `input_name` gets a buffer from this class.
  auto is_lora_buffer_input = [](absl::string_view input_name) {
    return IsLoRAInputName(input_name) || input_name == kLoRAParamTensorName;
  };

  // Create the buffers for the signature the LoRA was created with. These are
  // kept in a local map and moved into `lora_buffers_by_signature_` at the end:
  // inserting other signatures below may rehash `lora_buffers_by_signature_`,
  // which would invalidate a reference into it.
  absl::flat_hash_map<std::string, TensorBuffer> primary_buffers;
  for (const auto& input_name : input_names) {
    if (!is_lora_buffer_input(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(primary_buffers[input_name],
                            create_buffer(signature_name_, input_name));
  }

  // A LoRA that matches none of the model's LoRA inputs has no effect at all,
  // which is almost certainly a naming mismatch between the LoRA file and the
  // model (e.g. a LoRA converted for a different model or backend).
  int num_lora_inputs = 0;
  for (const auto& input_name : input_names) {
    if (IsLoRAInputName(input_name)) {
      ++num_lora_inputs;
    }
  }
  if (num_lora_inputs > 0 &&
      unmatched_input_names.size() == static_cast<size_t>(num_lora_inputs)) {
    const std::vector<std::string> lora_tensor_names_in_data =
        lora_data_->GetAllTensorNames();
    return absl::InvalidArgumentError(absl::StrFormat(
        "None of the %d LoRA inputs of signature '%s' has a matching tensor "
        "in the LoRA data (e.g. model input '%s', LoRA tensor '%s'). Check "
        "that the LoRA was converted with tensor names for this model.",
        num_lora_inputs, signature_name_, unmatched_input_names.front(),
        lora_tensor_names_in_data.empty() ? "<none>"
                                          : lora_tensor_names_in_data.front()));
  }
  if (!unmatched_input_names.empty()) {
    constexpr int kMaxNamesToLog = 5;
    ABSL_LOG(WARNING)
        << unmatched_input_names.size() << " of " << num_lora_inputs
        << " LoRA inputs of signature '" << signature_name_
        << "' have no matching tensor in the LoRA data and are filled with "
           "zeros, e.g. "
        << absl::StrJoin(
               absl::MakeConstSpan(unmatched_input_names)
                   .subspan(0, kMaxNamesToLog),
               ", ");
  }

  // Other signatures may require different buffer types for the same LoRA
  // input. For example, on GPU a LoRA input that has no consumer in "prefill"
  // only accepts host memory there, while "decode" uses a GPU buffer. Reuse an
  // existing buffer when its type is supported, otherwise create one of a
  // supported type holding the same weights.
  // LoRA input name -> buffers created for signatures other than the primary.
  absl::flat_hash_map<std::string, std::vector<TensorBuffer>> extra_buffers;
  LITERT_ASSIGN_OR_RETURN(auto signature_keys,
                          compiled_model_.GetSignatureKeys());
  for (const auto& signature_key : signature_keys) {
    if (signature_key == signature_name_) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto signature_input_names,
        compiled_model_.GetSignatureInputNames(signature_key));
    absl::flat_hash_map<std::string, TensorBuffer> signature_buffers;
    bool signature_runnable = true;
    for (const auto& input_name : signature_input_names) {
      if (!is_lora_buffer_input(input_name)) {
        continue;
      }
      auto requirements =
          compiled_model_.GetInputBufferRequirements(signature_key, input_name);
      if (!requirements) {
        // Requirements are unavailable for signatures that are not active in
        // the compiled model; such signatures cannot be run, so skip them.
        // GetLoRABuffers() returns NotFound for a skipped signature, so log
        // why it was skipped.
        ABSL_LOG(WARNING) << "No LoRA buffers for signature '" << signature_key
                          << "': buffer requirements of input '" << input_name
                          << "' are unavailable: "
                          << requirements.Error().Message();
        signature_runnable = false;
        break;
      }
      LITERT_ASSIGN_OR_RETURN(auto supported_types,
                              requirements->SupportedTypes());
      auto is_supported = [&supported_types](const TensorBuffer& buffer) {
        auto buffer_type = buffer.BufferType();
        return buffer_type &&
               std::find(supported_types.begin(), supported_types.end(),
                         *buffer_type) != supported_types.end();
      };

      const TensorBuffer* reusable = nullptr;
      if (auto it = primary_buffers.find(input_name);
          it != primary_buffers.end() && is_supported(it->second)) {
        reusable = &it->second;
      } else {
        for (const auto& buffer : extra_buffers[input_name]) {
          if (is_supported(buffer)) {
            reusable = &buffer;
            break;
          }
        }
      }
      if (reusable == nullptr) {
        LITERT_ASSIGN_OR_RETURN(auto buffer,
                                create_buffer(signature_key, input_name));
        auto& buffers = extra_buffers[input_name];
        buffers.push_back(std::move(buffer));
        // Only valid until the next push_back into `buffers`; it is duplicated
        // right below, before any other buffer is added.
        reusable = &buffers.back();
      }
      LITERT_ASSIGN_OR_RETURN(signature_buffers[input_name],
                              reusable->Duplicate());
    }
    if (signature_runnable) {
      lora_buffers_by_signature_[signature_key] = std::move(signature_buffers);
    }
  }
  lora_buffers_by_signature_[signature_name_] = std::move(primary_buffers);
  return absl::OkStatus();
}

absl::StatusOr<litert::TensorBuffer> LoRA::GetLoRABuffer(
    const std::string& name) const {
  const auto& buffers = lora_buffers_by_signature_.at(signature_name_);
  auto it = buffers.find(name);
  if (it == buffers.end()) {
    return absl::NotFoundError("LoRA tensor not found.");
  }
  LITERT_ASSIGN_OR_RETURN(auto duplicated_buffer, it->second.Duplicate());
  return duplicated_buffer;
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
LoRA::GetLoRABuffers() const {
  return GetLoRABuffers(signature_name_);
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
LoRA::GetLoRABuffers(absl::string_view signature_name) const {
  auto signature_it = lora_buffers_by_signature_.find(signature_name);
  if (signature_it == lora_buffers_by_signature_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No LoRA buffers for signature: %s", signature_name));
  }
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> buffers;
  for (const auto& [name, buffer] : signature_it->second) {
    LITERT_ASSIGN_OR_RETURN(buffers[name], buffer.Duplicate());
  }
  return buffers;
}

}  // namespace litert::lm

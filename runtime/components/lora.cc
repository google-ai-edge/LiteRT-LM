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
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
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

  for (const auto& input_name : input_names) {
    if (!IsLoRAInputName(input_name)) {
      continue;
    }
    // Create the input buffer for the LoRA tensor.
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBuffer tensor_buffer,
        compiled_model_.CreateInputBuffer(signature_name_, input_name));

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
    }

    lora_buffers_[input_name] = std::move(tensor_buffer);
  }

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

    int64_t max_extent = 0;
    for (const auto& dims : lora_dims) {
      int64_t num_elements = 1;
      for (int32_t dim : dims) {
        num_elements *= dim;
      }
      max_extent = std::max(max_extent, num_elements / model_rank);
    }

    LITERT_ASSIGN_OR_RETURN(TensorBuffer param_buffer,
                            compiled_model_.CreateInputBuffer(
                                signature_name_, kLoRAParamTensorName));
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
                static_cast<int32_t>(max_extent));
      params[0] = 0;
    }
    lora_buffers_[kLoRAParamTensorName] = std::move(param_buffer);
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

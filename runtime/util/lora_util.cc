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

#include "runtime/util/lora_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

constexpr LazyRE2 kLoRAInputNamePattern = {
    "^(?:(?:query|key|value|post)_w_prime_(?:left|right)_?\\d+|"
    "lora_atten_(?:q|k|v|o)_(?:a|b)_prime_weight_\\d+)$|"
    "^transformer\\.layer_\\d+\\.attn\\."
    "(?:q|k|v|attn_vec_einsum)\\.w_prime_(?:left|right)$"};

uint64_t AlignByN(uint64_t number, uint64_t n) {
  const uint64_t q = number / n;
  return (number % n == 0 ? q : q + 1) * n;
}

}  // namespace

// Gets an offset and size which will be valid to pass to
// MemoryMappedFile.
absl::StatusOr<std::unique_ptr<MemoryMappedFileWithAutoAlignment>>
MemoryMappedFileWithAutoAlignment::Create(ScopedFile::PlatformFile file,
                                          uint64_t offset, uint64_t size,
                                          absl::string_view key) {
  const size_t kAlignment = MemoryMappedFile::GetOffsetAlignment();
  uint64_t aligned_offset = (offset / kAlignment) * kAlignment;

  uint64_t map_size;
  if (size == 0) {
    map_size = 0;  // Map to the end of the file.
  } else {
    map_size = AlignByN(offset - aligned_offset + size, kAlignment);
  }

  ABSL_ASSIGN_OR_RETURN(auto region, MemoryMappedFile::Create(
                                         file, aligned_offset, map_size, key));

  uint64_t internal_offset = offset - aligned_offset;
  uint64_t final_size;
  if (size == 0) {
    final_size = region->length() - internal_offset;
  } else {
    final_size = size;
  }

  return absl::WrapUnique(new MemoryMappedFileWithAutoAlignment(
      std::move(region), internal_offset, final_size));
}

// Returns true if the given name is a LoRA input name for the model.
// The LoRA name is in the format of
// "(query|key|value|post)_w_prime_(left|right)_[0-num_layers)" or
// "lora_atten_(q|k|v|o)_(a|b)_prime_weight_[0-num_layers)".
bool IsLoRAInputName(absl::string_view name) {
  return RE2::FullMatch(name, *kLoRAInputNamePattern);
}

bool IsLoRABufferInputName(absl::string_view name) {
  return IsLoRAInputName(name) || name == kLoRAParamTensorName;
}

// GPU models exported with runtime-BMM LoRA read the channel range of every
// LoRA BMM op from `lora_param_tensor`: element 0 is the start index and the
// others are end indices. Without this the range is unset, so the LoRA ops
// compute on garbage ranges (no effect, wrong output or out-of-bounds access).
// Set the ends to the largest non-rank extent of any LoRA input (all dimensions
// but the rank, multiplied, since the ops may reshape e.g. [rank, heads,
// head_dim] to [rank, heads * head_dim]), so that every op covers all of its
// channels. The rank is the model's LoRA rank (a LoRA with a smaller rank is
// zero padded to it), i.e. the dimension that every LoRA input has.
absl::StatusOr<int32_t> ComputeLoRAParamTensorEnd(
    absl::Span<const std::vector<int32_t>> lora_input_dims) {
  RET_CHECK(!lora_input_dims.empty())
      << kLoRAParamTensorName << " is an input but there are no LoRA inputs.";

  auto find_shared_rank = [&](int32_t min_candidate) -> int32_t {
    int32_t rank = 0;
    for (int32_t candidate : lora_input_dims[0]) {
      if (candidate < min_candidate || (rank > 0 && candidate >= rank)) {
        continue;
      }
      if (std::all_of(lora_input_dims.begin(), lora_input_dims.end(),
                      [candidate](const std::vector<int32_t>& dims) {
                        return std::find(dims.begin(), dims.end(), candidate) !=
                               dims.end();
                      })) {
        rank = candidate;
      }
    }
    return rank;
  };

  // Prefer shared dimensions > 1 so singleton batch/spatial dimensions (e.g.
  // [1, 32, 1, 1536]) do not shadow the actual LoRA rank.
  int32_t model_rank = find_shared_rank(/*min_candidate=*/2);
  if (model_rank == 0) {
    model_rank = find_shared_rank(/*min_candidate=*/1);
  }
  RET_CHECK_GT(model_rank, 0)
      << "Could not find a LoRA rank shared by all LoRA inputs.";

  int64_t max_extent = 0;
  for (const auto& dims : lora_input_dims) {
    RET_CHECK(!dims.empty()) << "LoRA input has empty dimensions.";
    int64_t num_elements = 1;
    for (int32_t dim : dims) {
      RET_CHECK_GT(dim, 0) << "LoRA input has non-positive dimension: " << dim;
      num_elements *= dim;
    }
    max_extent = std::max(max_extent, num_elements / model_rank);
  }
  RET_CHECK_LE(max_extent, std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(max_extent);
}

absl::Status FillLoRAParamTensorValues(
    absl::Span<const std::vector<int32_t>> lora_input_dims,
    absl::Span<int32_t> params) {
  // Element 0 is the start index; at least one end index must follow.
  RET_CHECK_GE(params.size(), 2)
      << "Unexpected size of " << kLoRAParamTensorName;
  ABSL_ASSIGN_OR_RETURN(const int32_t lora_param_end,
                        ComputeLoRAParamTensorEnd(lora_input_dims));
  std::fill(params.begin(), params.end(), lora_param_end);
  params[0] = 0;
  return absl::OkStatus();
}

absl::Status PopulateLoRAParamTensor(
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name, litert::TensorBuffer& param_buffer) {
  LITERT_ASSIGN_OR_RETURN(
      auto input_names, compiled_model.GetSignatureInputNames(signature_name));
  std::vector<std::vector<int32_t>> lora_dims;
  for (const auto& input_name : input_names) {
    if (!IsLoRAInputName(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto tensor_type,
        compiled_model.GetInputTensorType(signature_name, input_name));
    const auto dims = tensor_type.Layout().Dimensions();
    lora_dims.emplace_back(dims.begin(), dims.end());
  }
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      litert::TensorBufferScopedLock::Create<int32_t>(
          param_buffer, litert::TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, param_buffer.PackedSize());
  return FillLoRAParamTensorValues(
      lora_dims,
      absl::MakeSpan(lock_and_addr.second, packed_size / sizeof(int32_t)));
}

absl::Status InitLoRAParamTensorIfPresent(
    const litert::CompiledModel& compiled_model,
    absl::string_view signature_name,
    absl::flat_hash_map<absl::string_view, litert::TensorBuffer>&
        input_buffers_map) {
  auto param_it = input_buffers_map.find(kLoRAParamTensorName);
  if (param_it == input_buffers_map.end()) {
    return absl::OkStatus();
  }
  return PopulateLoRAParamTensor(compiled_model, signature_name,
                                 param_it->second);
}

}  // namespace litert::lm

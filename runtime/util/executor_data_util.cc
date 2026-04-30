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

#include "runtime/util/executor_data_util.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

template <typename T>
absl::StatusOr<T> CombineExecutorDataImpl(std::vector<T>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    // If there is only one image, we can just move it to the combined image
    // data.
    return std::move(executor_data[0]);
  }
  // If there are multiple executor data, we need to first combine them into a
  // TensorBuffer, then create a single ExecutorVisionData from the
  // TensorBuffer.
  int num_executor_data = executor_data.size();
  ASSIGN_OR_RETURN(const auto* first_tensor,
                   executor_data[0].GetEmbeddingsPtr());
  LITERT_ASSIGN_OR_RETURN(auto first_tensor_type, first_tensor->TensorType());
  ASSIGN_OR_RETURN(auto first_tensor_dims, TensorBufferDims(*first_tensor));
  int total_token_num = 0;
  int total_packed_size = 0;
  std::vector<int> combined_token_num;
  for (const auto& executor_data : executor_data) {
    ASSIGN_OR_RETURN(const auto* embeddings_ptr,
                     executor_data.GetEmbeddingsPtr());
    ASSIGN_OR_RETURN(auto dims, TensorBufferDims(*embeddings_ptr));
    if (dims.size() != 3 && dims.size() != 4) {
      return absl::InvalidArgumentError(
          "The embedding tensor type must have 3 or 4 dimensions.");
    }
    combined_token_num.push_back(dims[dims.size() - 2]);
    total_token_num += dims[dims.size() - 2];
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, embeddings_ptr->PackedSize());
    total_packed_size += packed_size;
  }
  Layout combined_layout;
  if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 3) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], 1, total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 4) {
    combined_layout =
        Layout(Dimensions({first_tensor_dims[0], first_tensor_dims[1],
                           total_token_num, first_tensor_dims[3]}));
  }
  ::litert::RankedTensorType combined_tensor_type(
      first_tensor_type.ElementType(), std::move(combined_layout));

  LITERT_ASSIGN_OR_RETURN(auto combined_tensor_buffer,
                          TensorBuffer::CreateManagedHostMemory(
                              combined_tensor_type, total_packed_size));
  LITERT_ASSIGN_OR_RETURN(
      auto combined_embeddings_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(combined_tensor_buffer,
                                               TensorBuffer::LockMode::kWrite));
  char* combined_tensor_buffer_ptr =
      static_cast<char*>(combined_embeddings_lock_and_addr.second);
  for (int i = 0; i < num_executor_data; ++i) {
    ASSIGN_OR_RETURN(auto embeddings_ptr,
                     executor_data[i].GetMutableEmbeddingsPtr());
    LITERT_ASSIGN_OR_RETURN(auto embeddings_size, embeddings_ptr->PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto embeddings_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            *embeddings_ptr, TensorBuffer::LockMode::kRead));
    memcpy(combined_tensor_buffer_ptr, embeddings_lock_and_addr.second,
           embeddings_size);
    combined_tensor_buffer_ptr += embeddings_size;
  }
  if constexpr (std::is_same_v<T, ExecutorVisionData>) {
    return ExecutorVisionData(std::move(combined_tensor_buffer),
                              /*per_layer_embeddings=*/std::nullopt);
  } else if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    int num_audio_tokens = 0;
    for (const auto& executor_data : executor_data) {
      num_audio_tokens += executor_data.GetValidTokens();
    }
    return ExecutorAudioData(std::move(combined_tensor_buffer),
                             /*per_layer_embeddings=*/std::nullopt,
                             num_audio_tokens);
  } else {
    return absl::InvalidArgumentError("Executor data type is not supported.");
  }
}

}  // namespace

absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

}  // namespace litert::lm

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

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
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

struct CombinedTensorBuffer {
  ::litert::TensorBuffer buffer;
  int total_tokens = 0;
};

// Returns the token capacity along the sequence dimension of an embedding
// tensor of shape:
// - 3D: [B, T, D] -> dims[1]
// - 4D: [B, dim1, T, D] -> dims[2]
// - 5D (Artisan BHWDC): [B, 1, W, 1, C] or [B, 1, 1, D, C] -> dims[2] * dims[3]
absl::StatusOr<int> GetTensorTokenCapacity(const std::vector<int>& dims) {
  if (dims.size() != 3 && dims.size() != 4 && dims.size() != 5) {
    return absl::InvalidArgumentError(
        "The embedding tensor type must have 3, 4, or 5 dimensions.");
  }
  for (int dim : dims) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          "Embedding tensor dimensions must be non-negative.");
    }
  }
  if (dims.size() == 3 || dims.size() == 4) {
    return dims[dims.size() - 2];
  }
  if (dims[1] != 1 || (dims[2] != 1 && dims[3] != 1)) {
    return absl::InvalidArgumentError(
        "5D audio embedding tensor must have H=1 and either W=1 or D=1.");
  }
  return dims[2] * dims[3];
}

// Concatenates `tensors` along their token sequence axis. If
// `valid_token_counts` is non-empty, copies `valid_token_counts[i]` tokens from
// `tensors[i]` (slicing off any trailing padding tokens, e.g. from
// non-streaming GpuArtisanAudioExecutor).
absl::StatusOr<CombinedTensorBuffer> CombineTensorBuffers(
    const std::vector<::litert::TensorBuffer*>& tensors,
    const std::vector<int>& valid_token_counts, bool is_audio) {
  if (tensors.empty()) {
    return absl::InvalidArgumentError("Tensor list is empty.");
  }
  if (!valid_token_counts.empty() &&
      valid_token_counts.size() != tensors.size()) {
    return absl::InvalidArgumentError(
        "valid_token_counts size must match tensors size.");
  }
  if (tensors[0] == nullptr) {
    return absl::InvalidArgumentError("Tensor pointer cannot be null.");
  }
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType first_tensor_type,
                          tensors[0]->TensorType());
  ABSL_ASSIGN_OR_RETURN(std::vector<int> first_tensor_dims,
                        TensorBufferDims(*tensors[0]));
  ABSL_RETURN_IF_ERROR(GetTensorTokenCapacity(first_tensor_dims).status());
  if (first_tensor_dims[0] <= 0) {
    return absl::InvalidArgumentError("Batch size must be positive.");
  }
  const int embedding_dim = first_tensor_dims.back();
  if (embedding_dim <= 0) {
    return absl::InvalidArgumentError("Embedding dimension must be positive.");
  }

  int total_token_num = 0;
  size_t total_packed_size = 0;
  std::vector<size_t> copy_byte_sizes;
  copy_byte_sizes.reserve(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i] == nullptr) {
      return absl::InvalidArgumentError("Tensor pointer cannot be null.");
    }
    LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType tensor_type,
                            tensors[i]->TensorType());
    if (tensor_type.Layout().HasStrides()) {
      return absl::InvalidArgumentError(
          "Strided embedding TensorBuffers are not supported.");
    }
    if (tensor_type.ElementType() != first_tensor_type.ElementType()) {
      return absl::InvalidArgumentError(
          "Mismatched element types across embedding tensors.");
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<int> dims, TensorBufferDims(*tensors[i]));
    if (!is_audio) {
      if (dims.size() != 3 && dims.size() != 4) {
        return absl::InvalidArgumentError(
            "Vision embedding tensor must have 3 or 4 dimensions.");
      }
      const int first_dim1 =
          (first_tensor_dims.size() == 4) ? first_tensor_dims[1] : 1;
      const int dim1 = (dims.size() == 4) ? dims[1] : 1;
      if (dim1 != first_dim1) {
        return absl::InvalidArgumentError(
            "Mismatched leading dimension across vision tensors.");
      }
    } else {
      if (dims.size() == 4 && dims[1] != 1) {
        return absl::InvalidArgumentError(
            "4D audio embedding tensor must have dims[1]=1.");
      }
      if ((dims.size() == 5) != (first_tensor_dims.size() == 5)) {
        return absl::InvalidArgumentError(
            "Cannot mix 5D and non-5D audio embedding tensors.");
      }
    }
    ABSL_ASSIGN_OR_RETURN(int tensor_tokens, GetTensorTokenCapacity(dims));
    if (dims[0] != first_tensor_dims[0] || dims.back() != embedding_dim) {
      return absl::InvalidArgumentError(
          "Mismatched batch size or embedding dimension across tensors.");
    }
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, tensors[i]->PackedSize());
    if (tensor_tokens > 0 &&
        packed_size % static_cast<size_t>(tensor_tokens) != 0) {
      return absl::InvalidArgumentError(
          "Tensor packed byte size is not divisible by token capacity.");
    }
    int tokens_to_copy = tensor_tokens;
    if (!valid_token_counts.empty() && valid_token_counts[i] >= 0) {
      tokens_to_copy = std::min(valid_token_counts[i], tensor_tokens);
    }
    if (tokens_to_copy < tensor_tokens && dims[0] != 1) {
      return absl::InvalidArgumentError(
          "Slicing valid tokens requires batch size dims[0]=1.");
    }
    if (tokens_to_copy > std::numeric_limits<int>::max() - total_token_num) {
      return absl::InvalidArgumentError("Combined token count overflows int.");
    }
    const size_t bytes_to_copy =
        (tensor_tokens > 0)
            ? (packed_size / static_cast<size_t>(tensor_tokens)) *
                  static_cast<size_t>(tokens_to_copy)
            : 0;
    copy_byte_sizes.push_back(bytes_to_copy);
    total_token_num += tokens_to_copy;
    total_packed_size += bytes_to_copy;
  }

  Layout combined_layout;
  if (is_audio) {
    if (first_tensor_dims.size() == 5) {
      combined_layout = Layout(Dimensions(
          {first_tensor_dims[0], 1, total_token_num, 1, embedding_dim}));
    } else {
      combined_layout = Layout(
          Dimensions({first_tensor_dims[0], total_token_num, embedding_dim}));
    }
  } else if (first_tensor_dims.size() == 3) {
    combined_layout = Layout(
        Dimensions({first_tensor_dims[0], 1, total_token_num, embedding_dim}));
  } else if (first_tensor_dims.size() == 4) {
    combined_layout =
        Layout(Dimensions({first_tensor_dims[0], first_tensor_dims[1],
                           total_token_num, embedding_dim}));
  } else {
    return absl::InvalidArgumentError(
        "Vision embedding tensor must have 3 or 4 dimensions.");
  }

  ::litert::RankedTensorType combined_tensor_type(
      first_tensor_type.ElementType(), std::move(combined_layout));
  LITERT_ASSIGN_OR_RETURN(::litert::TensorBuffer combined_tensor_buffer,
                          TensorBuffer::CreateManagedHostMemory(
                              combined_tensor_type, total_packed_size));
  LITERT_ASSIGN_OR_RETURN(
      auto combined_embeddings_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(combined_tensor_buffer,
                                               TensorBuffer::LockMode::kWrite));
  char* dst_ptr = static_cast<char*>(combined_embeddings_lock_and_addr.second);
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (copy_byte_sizes[i] == 0) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto src_lock_and_addr,
                            ::litert::TensorBufferScopedLock::Create(
                                *tensors[i], TensorBuffer::LockMode::kRead));
    std::memcpy(dst_ptr, src_lock_and_addr.second, copy_byte_sizes[i]);
    dst_ptr += copy_byte_sizes[i];
  }
  return CombinedTensorBuffer{
      .buffer = std::move(combined_tensor_buffer),
      .total_tokens = total_token_num,
  };
}

}  // namespace

absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    return std::move(executor_data[0]);
  }
  std::vector<::litert::TensorBuffer*> tensors;
  tensors.reserve(executor_data.size());
  for (ExecutorVisionData& data : executor_data) {
    ABSL_ASSIGN_OR_RETURN(::litert::TensorBuffer * ptr,
                          data.GetMutableEmbeddingsPtr());
    tensors.push_back(ptr);
  }
  ABSL_ASSIGN_OR_RETURN(CombinedTensorBuffer combined_tensor,
                        CombineTensorBuffers(tensors, /*valid_token_counts=*/{},
                                             /*is_audio=*/false));
  return ExecutorVisionData(std::move(combined_tensor.buffer),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    return std::move(executor_data[0]);
  }
  std::vector<::litert::TensorBuffer*> projected_tensors;
  std::vector<::litert::TensorBuffer*> unprojected_tensors;
  std::vector<int> valid_token_counts;
  projected_tensors.reserve(executor_data.size());
  unprojected_tensors.reserve(executor_data.size());
  valid_token_counts.reserve(executor_data.size());

  for (ExecutorAudioData& data : executor_data) {
    ABSL_ASSIGN_OR_RETURN(::litert::TensorBuffer * projected_ptr,
                          data.GetMutableProjectedAudioEmbeddingsPtr());
    projected_tensors.push_back(projected_ptr);
    valid_token_counts.push_back(data.GetValidTokens());
    if (absl::StatusOr<::litert::TensorBuffer*> unprojected_ptr =
            data.GetMutableAudioEmbeddingsPtr();
        unprojected_ptr.ok() && *unprojected_ptr != nullptr) {
      unprojected_tensors.push_back(*unprojected_ptr);
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      CombinedTensorBuffer combined_projected,
      CombineTensorBuffers(projected_tensors, valid_token_counts,
                           /*is_audio=*/true));
  ExecutorAudioData combined_audio(std::move(combined_projected.buffer),
                                   /*per_layer_embeddings=*/std::nullopt,
                                   combined_projected.total_tokens);
  if (unprojected_tensors.size() == executor_data.size()) {
    ABSL_ASSIGN_OR_RETURN(
        CombinedTensorBuffer combined_unprojected,
        CombineTensorBuffers(unprojected_tensors, valid_token_counts,
                             /*is_audio=*/true));
    combined_audio.SetAudioEmbeddings(std::move(combined_unprojected.buffer));
  }
  return combined_audio;
}

}  // namespace litert::lm

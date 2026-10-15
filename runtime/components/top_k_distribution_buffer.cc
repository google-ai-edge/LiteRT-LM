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

#include "runtime/components/top_k_distribution_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {
namespace {

// Returns the size of the last dimension of `tensor` after checking its
// element type and that it has `rows * K` elements.
absl::StatusOr<int> GetLastDimension(const TensorBuffer& tensor,
                                     ElementType element_type, int rows,
                                     absl::string_view name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  if (tensor_type.ElementType() != element_type) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The top-k ", name, " tensor has the wrong element type."));
  }
  const auto dims = tensor_type.Layout().Dimensions();
  if (dims.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The top-k ", name, " tensor must not be a scalar."));
  }
  int64_t num_elements = 1;
  for (const auto dim : dims) {
    num_elements *= dim;
  }
  const int last_dim = dims[dims.size() - 1];
  if (last_dim <= 0 || num_elements != static_cast<int64_t>(rows) * last_dim) {
    return absl::InvalidArgumentError(
        absl::StrCat("The top-k ", name,
                     " tensor must have shape [batch_size, "
                     "sequence_size, K] with ",
                     rows, " rows, but has ", num_elements,
                     " elements and K = ", last_dim, "."));
  }
  return last_dim;
}

}  // namespace

absl::StatusOr<int> GetTopKDistributionBufferSize(
    const TensorBuffer& probs_tensor, const TensorBuffer& ids_tensor,
    int batch_size, int sequence_size) {
  if (batch_size <= 0 || sequence_size <= 0) {
    return absl::InvalidArgumentError(
        "batch_size and sequence_size must be positive.");
  }
  const int rows = batch_size * sequence_size;
  ABSL_ASSIGN_OR_RETURN(
      const int probs_k,
      GetLastDimension(probs_tensor, ElementType::Float32, rows, "probs"));
  ABSL_ASSIGN_OR_RETURN(
      const int ids_k,
      GetLastDimension(ids_tensor, ElementType::Int32, rows, "ids"));
  if (probs_k != ids_k) {
    return absl::InvalidArgumentError(
        absl::StrCat("The top-k probs and ids tensors must have the same K, "
                     "but got ",
                     probs_k, " and ", ids_k, "."));
  }
  return probs_k;
}

absl::Status WriteTopKDistribution(absl::Span<const float> probs,
                                   absl::Span<const int> ids, int src_k,
                                   int num_valid, int batch_size,
                                   int sequence_size,
                                   TensorBuffer& probs_tensor,
                                   TensorBuffer& ids_tensor) {
  ABSL_ASSIGN_OR_RETURN(const int out_k, GetTopKDistributionBufferSize(
                                             probs_tensor, ids_tensor,
                                             batch_size, sequence_size));
  const int rows = batch_size * sequence_size;
  if (src_k <= 0 || num_valid < 0 || num_valid > src_k ||
      probs.size() != static_cast<size_t>(rows) * src_k ||
      ids.size() != probs.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid top-k source distribution: src_k = ", src_k,
                     ", num_valid = ", num_valid, ", ", probs.size(),
                     " probs and ", ids.size(), " ids for ", rows, " rows."));
  }
  if (out_k < num_valid) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The top-k tensors have K = ", out_k, ", which is less than the ",
        num_valid, " entries of the distribution."));
  }
  LITERT_ASSIGN_OR_RETURN(auto probs_lock,
                          TensorBufferScopedLock::Create<float>(
                              probs_tensor, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(auto ids_lock,
                          TensorBufferScopedLock::Create<int>(
                              ids_tensor, TensorBuffer::LockMode::kWrite));
  for (int r = 0; r < rows; ++r) {
    float* row_probs = probs_lock.second + static_cast<size_t>(r) * out_k;
    int* row_ids = ids_lock.second + static_cast<size_t>(r) * out_k;
    std::copy_n(probs.begin() + static_cast<size_t>(r) * src_k, num_valid,
                row_probs);
    std::fill_n(row_probs + num_valid, out_k - num_valid, 0.0f);
    std::copy_n(ids.begin() + static_cast<size_t>(r) * src_k, num_valid,
                row_ids);
    std::fill_n(row_ids + num_valid, out_k - num_valid, -1);
  }
  return absl::OkStatus();
}

}  // namespace litert::lm

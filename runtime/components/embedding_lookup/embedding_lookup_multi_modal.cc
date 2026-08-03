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

#include "runtime/components/embedding_lookup/embedding_lookup_multi_modal.h"

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

absl::Status EmbeddingLookupMultiModal::LookupDecode(
    int token, std::vector<float>& output_vector) {
  // Multimodal lookup is not supported for single token case because decode
  // does not use multimodal embedding lookup.
  return absl::UnimplementedError(
      "Multimodal embedding lookup is not supported for single token decode "
      "case.");
}

absl::Status EmbeddingLookupMultiModal::LookupDecode(
    int token, litert::TensorBuffer* output_tensor) {
  // Multimodal lookup is not supported for single token case because decode
  // does not use multimodal embedding lookup.
  return absl::UnimplementedError(
      "Multimodal embedding lookup is not supported for single token decode "
      "case.");
}

absl::Status EmbeddingLookupMultiModal::LookupPrefill(
    int token, std::vector<float>& output_vector) {
  // Support this case because it is used for the case where the
  // llm_litert_executor needs to look up embeddings for the current step and
  // then use the result for the next step. At that point, it does not have a
  // TfLiteTensor to store the result in.
  if (token != special_token_) {
    return absl::OkStatus();
  }

  if (embedding_.size() < output_vector.size()) {
    return absl::InvalidArgumentError(
        "The embedding buffer is not large enough to contain the number of "
        "requested tokens.");
  }

  // Copy the embedding data to the output vector.
  std::memcpy(output_vector.data(), embedding_.data(),
              output_vector.size() * sizeof(float));
  // Remove used embeddings from the buffer.
  embedding_ = embedding_.subspan(output_vector.size());

  return absl::OkStatus();
}

absl::Status EmbeddingLookupMultiModal::LookupPrefill(
    absl::Span<const int> tokens, litert::TensorBuffer* output_tensor,
    size_t byte_offset) {
  if (output_tensor == nullptr) {
    return absl::InvalidArgumentError("Output tensor is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto output_tensor_type, output_tensor->TensorType());
  const auto& output_tensor_layout = output_tensor_type.Layout();

  // Embedding lookup only supports float32 output tensor type right now.
  if (output_tensor_type.ElementType() != litert::ElementType::Float32) {
    return absl::UnimplementedError(
        "The output tensor type for multimodal embedding lookup must be "
        "float32.");
  }

  if (output_tensor_layout.Rank() < 3) {
    return absl::UnimplementedError(
        "The output tensor provided to the Embedding LookupPrefill function "
        "must have at least 3 dimensions.");
  }

  if (output_tensor_layout.Dimensions()[0] != 1) {
    return absl::UnimplementedError(
        "The output tensor to fill with the multimodal embeddings must be have "
        "the 0th dimension as 1. Other sizes are not supported yet.");
  }

  if (output_tensor_layout.Dimensions()[1] < tokens.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor to fill from the multimodal embeddings must have a "
        "1st dimension that is at least the same size as the number of tokens. "
        "Requested tensor 1st dim: ",
        output_tensor_layout.Dimensions()[1], " but the number of tokens is ",
        tokens.size()));
  }

  size_t floats_per_token = 1;
  for (size_t i = 2; i < output_tensor_layout.Rank(); ++i) {
    floats_per_token *= output_tensor_layout.Dimensions()[i];
  }

  const size_t size_of_float = sizeof(float);
  const size_t bytes_per_token = floats_per_token * size_of_float;

  LITERT_ASSIGN_OR_RETURN(auto output_tensor_size, output_tensor->Size());

  if (byte_offset + bytes_per_token * tokens.size() > output_tensor_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The byte offset and the total number of bytes to be "
                     "written must not exceed the size of the output "
                     "tensor. Byte offset: ",
                     byte_offset, ". Bytes per token: ", bytes_per_token,
                     ". Number of tokens: ", tokens.size(),
                     ". Output tensor bytes: ", output_tensor->Size()));
  }

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *output_tensor, TensorBuffer::LockMode::kWrite);
  auto output_tensor_ptr =
      reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

  output_tensor_ptr += byte_offset;
  for (int token : tokens) {
    if (token == special_token_) {
      // Check if we have enough embeddings left to be read to cover the next
      // token.
      if (embedding_.size() < floats_per_token) {
        return absl::InvalidArgumentError(
            "The embedding buffer is not large enough to contain the number of "
            "requested tokens.");
      }
      // Copy the embedding data to the output tensor.
      std::memcpy(output_tensor_ptr, embedding_.data(), bytes_per_token);
      // Remove used embeddings from the buffer.
      embedding_ = embedding_.subspan(floats_per_token);
    }
    output_tensor_ptr += bytes_per_token;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupMultiModal>>
EmbeddingLookupMultiModal::Create(
    const ::litert::TensorBuffer* embedding_buffer, int special_token) {
  auto handler = std::make_unique<EmbeddingLookupMultiModal>();
  RETURN_IF_ERROR(handler->Initialize(embedding_buffer, special_token));
  return handler;
}

absl::Status EmbeddingLookupMultiModal::Initialize(
    const ::litert::TensorBuffer* embedding_buffer, int special_token) {
  if (embedding_buffer == nullptr) {
    return absl::InvalidArgumentError(
        "Cannot initialize embedding lookup with an embedding buffer that is "
        "null.");
  }
  LITERT_ASSIGN_OR_RETURN(
      embedding_,
      ::litert::lm::ReferTensorBufferAsSpan<float>(*embedding_buffer));
  special_token_ = special_token;
  return absl::OkStatus();
}

}  // namespace litert::lm

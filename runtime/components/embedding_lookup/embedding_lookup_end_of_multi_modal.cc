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

#include "runtime/components/embedding_lookup/embedding_lookup_end_of_multi_modal.h"

#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/status_macros.h"  //NOLINT
#if defined(__ANDROID__)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif

namespace litert::lm {

absl::Status EndOfMultiModalEmbedding::LookupPrefill(
    int token, std::vector<float>& output_vector) {
  if (token != special_token_) {
    return absl::OkStatus();
  }
  if (output_vector.size() != end_of_multi_modal_embedding_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output vector is not the correct size for the end of multi-modal "
        "embedding. Output vector size: ",
        output_vector.size(), ". End of multi-modal embedding size: ",
        end_of_multi_modal_embedding_.size()));
  }
  memcpy(output_vector.data(), end_of_multi_modal_embedding_.data(),
         output_vector.size() * sizeof(float));
  return absl::OkStatus();
}

absl::Status EndOfMultiModalEmbedding::LookupPrefill(
    absl::Span<const int> tokens, litert::TensorBuffer* prefill_output,
    size_t byte_offset) {
  if (prefill_output == nullptr) {
    return absl::InvalidArgumentError("Output tensor is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto prefill_output_type,
                          prefill_output->TensorType());
  const auto& prefill_output_layout = prefill_output_type.Layout();

  if (prefill_output_layout.Rank() != output_buffer_layout_.Rank()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor from the EndOfMultiModalEmbedding model must be "
        "have the same number of dimensions as the requested tensor. Requested "
        "tensor dims: ",
        prefill_output_layout.Rank(),
        ". Output tensor dims: ", prefill_output_layout.Rank()));
  }

  if (prefill_output_layout.Rank() < 3) {
    return absl::UnimplementedError(
        "The output tensor provided to the EndOfMultiModalEmbedding Lookup "
        "function must have  at least 3 dimensions.");
  }

  if (output_buffer_layout_.Rank() < 3) {
    return absl::UnimplementedError(
        "The output tensor from the EndOfMultiModalEmbedding model must have "
        "at least 3 dimensions.");
  }

  if (prefill_output_layout.Dimensions()[0] != 1) {
    return absl::UnimplementedError(
        "The output tensor to fill from the EndOfMultiModalEmbedding model "
        "must be have the 0th dimension as 1. Other sizes are not supported "
        "yet.");
  }

  if (prefill_output_layout.Dimensions()[1] < tokens.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor to fill from the EndOfMultiModalEmbedding model "
        "must have a 1st dimension that is at least the same size as the "
        "number of tokens. Requested tensor 1st dim: ",
        prefill_output_layout.Dimensions()[1], " but the number of tokens is ",
        tokens.size()));
  }

  for (size_t i = 2; i < prefill_output_layout.Rank(); ++i) {
    if (prefill_output_layout.Dimensions()[i] !=
        output_buffer_layout_.Dimensions()[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The output tensor from the EndOfMultiModalEmbedding model must be "
          "have the same dimensions as the requested tensor for dims > 1. "
          "Requested tensor dim for ",
          i, ": ", prefill_output_layout.Dimensions()[i],
          ". Output tensor dims: ", prefill_output_layout.Dimensions()[i]));
    }
  }

  const size_t bytes_per_token =
      end_of_multi_modal_embedding_.size() * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(auto prefill_output_size, prefill_output->Size());

  if (byte_offset + bytes_per_token * tokens.size() > prefill_output_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The byte offset and the total number of bytes to be "
                     "written must not exceed the size of the output "
                     "tensor. Byte offset: ",
                     byte_offset, ". Bytes per token: ", bytes_per_token,
                     ". Number of tokens: ", tokens.size(),
                     ". Output tensor bytes: ", prefill_output_size));
  }

  auto prefill_output_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *prefill_output, TensorBuffer::LockMode::kWrite);
  auto prefill_output_ptr =
      reinterpret_cast<uint8_t*>(prefill_output_lock_and_addr->second);
  prefill_output_ptr += byte_offset;
  for (int token : tokens) {
    if (token == special_token_) {
      memcpy(prefill_output_ptr, end_of_multi_modal_embedding_.data(),
             bytes_per_token);
    }
    prefill_output_ptr += bytes_per_token;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<EndOfMultiModalEmbedding>>
EndOfMultiModalEmbedding::Create(litert::Environment& env,
                                 const litert::Model* absl_nonnull model,
                                 int special_token) {
  auto handler = std::unique_ptr<EndOfMultiModalEmbedding>(
      new EndOfMultiModalEmbedding(env, model, special_token));
  RETURN_IF_ERROR(  // IWYU pragma: keep as is included by status_macros.h
      handler->Initialize());
  return handler;
}

absl::Status EndOfMultiModalEmbedding::Initialize() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
#if defined(__ANDROID__)
  options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                  litert::HwAccelerators::kCpu);
#else
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#endif
#if defined(__ANDROID__)
  LITERT_ASSIGN_OR_RETURN(::litert::qualcomm::QualcommOptions & qnn_opts,
                          options.GetQualcommOptions());
  qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kOff);
  qnn_opts.SetHtpPerformanceMode(
      ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::
          kSustainedHighPerformance);
#endif

  LITERT_ASSIGN_OR_RETURN(
      litert::CompiledModel compiled_model,
      litert::CompiledModel::Create(env_, model_.Get(), options));
  if (auto num_signatures = model_.GetNumSignatures(); num_signatures != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Embedding model must have exactly one signature but got ",
        num_signatures));
  }

  LITERT_ASSIGN_OR_RETURN(auto input_buffers, compiled_model.CreateInputBuffers(
                                                  /*signature_index=*/0));

  if (!input_buffers.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The Embedding model must have 0 input tensors but got ",
                     input_buffers.size()));
  }

  LITERT_ASSIGN_OR_RETURN(std::vector<litert::TensorBuffer> output_buffers,
                          compiled_model.CreateOutputBuffers(
                              /*signature_index=*/0));
  LITERT_ASSIGN_OR_RETURN(auto output_buffer_type,
                          output_buffers[0].TensorType());
  output_buffer_layout_ = output_buffer_type.Layout();

  if (output_buffers.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Embedding model must have exactly one output tensor but got ",
        output_buffers.size()));
  }

  if (output_buffer_type.ElementType() != litert::ElementType::Float32) {
    return absl::InvalidArgumentError(
        "The output tensor from the Embedding model must be of type float32.");
  }

  size_t floats_per_token = 1;
  for (size_t i = 2; i < output_buffer_layout_.Rank(); ++i) {
    floats_per_token *= output_buffer_layout_.Dimensions()[i];
  }
  end_of_multi_modal_embedding_.resize(floats_per_token);

  LITERT_ASSIGN_OR_RETURN(auto output_buffer_size, output_buffers[0].Size());
  if (output_buffer_size !=
      end_of_multi_modal_embedding_.size() * sizeof(float)) {
    return absl::InternalError(absl::StrCat(
        "The output tensor from the Embedding model must be the correct size "
        "for the end of multi-modal embedding. Output tensor size: ",
        output_buffers[0].Size(), ". End of multi-modal embedding size: ",
        end_of_multi_modal_embedding_.size()));
  }

  // Run the model and cache its output.
  LITERT_RETURN_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  uint8_t* data_ptr =
      reinterpret_cast<uint8_t*>(end_of_multi_modal_embedding_.data());
  size_t bytes = end_of_multi_modal_embedding_.size() * sizeof(float);
  output_buffers[0].Read(absl::MakeSpan(data_ptr, bytes));

  return absl::OkStatus();
}

}  // namespace litert::lm

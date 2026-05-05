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

#include "runtime/components/embedding_lookup/embedding_lookup_text.h"

#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/status_macros.h"  // NOLINT
#if defined(__ANDROID__)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif

namespace litert::lm {

using ::litert::TensorBuffer;

absl::Status EmbeddingLookupText::LookupInternal(int token,
                                                 absl::Span<uint8_t> buffer) {
  if (!compiled_model_.has_value() || input_buffers_.size() != 1 ||
      output_buffers_.size() != 1) {
    return absl::InvalidArgumentError(
        "The Embedding model must be initialized before being used.");
  }

  if (token < 0) {
    memcpy(buffer.data(), default_embedding_vector_.data(), buffer.size());
    return absl::OkStatus();
  }

  // The input tensor size was verified when the model was loaded.
  input_buffers_[0].Write(absl::MakeSpan(const_cast<const int*>(&token), 1));

  compiled_model_->Run(signature_key_.value(), input_buffers_, output_buffers_);

  LITERT_ASSIGN_OR_RETURN(auto output_buffer_size, output_buffers_[0].Size());

  if (buffer.size() != output_buffer_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor from the Embedding model must be have the same "
        "number of bytes as the requested tensor. Requested tensor bytes: ",
        buffer.size(), ". Output tensor bytes: ", output_buffer_size));
  }

  // Copy the output buffer to the requested buffer.
  output_buffers_[0].Read(buffer);

  return absl::OkStatus();
}

absl::Status EmbeddingLookupText::LookupDecode(
    int token, std::vector<float>& decode_output_vector) {
  // For text embedding, looking up a single token during decode is the same as
  // prefill.
  return LookupPrefill(token, decode_output_vector);
}

absl::Status EmbeddingLookupText::LookupDecode(int token,
                                               TensorBuffer* decode_output) {
  if (decode_output == nullptr) {
    return absl::InvalidArgumentError("Decode output tensor buffer is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_output_type, decode_output->TensorType());
  const auto& decode_output_layout = decode_output_type.Layout();
  const auto& output_buffer_layout = output_buffer_type_.value().Layout();

  if (decode_output_layout.Rank() != output_buffer_layout.Rank()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor from the Embedding model must be have the same "
        "number of dimensions as the requested tensor. Requested tensor dims: ",
        decode_output_layout.Rank(),
        ". Output tensor dims: ", output_buffer_layout.Rank()));
  }

  for (int i = 0; i < decode_output_layout.Rank(); ++i) {
    if (decode_output_layout.Dimensions()[i] !=
        output_buffer_layout.Dimensions()[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The output tensor from the Embedding model must be have the same "
          "dimensions as the requested tensor. Requested tensor dim for ",
          i, ": ", decode_output_layout.Dimensions()[i],
          ". Output tensor dims: ", output_buffer_layout.Dimensions()[i]));
    }
  }

  auto decode_output_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *decode_output, TensorBuffer::LockMode::kWrite);
  auto decode_output_ptr =
      reinterpret_cast<uint8_t*>(decode_output_lock_and_addr->second);

  LITERT_ASSIGN_OR_RETURN(auto decode_output_size, decode_output->Size());

  return LookupInternal(
      token, absl::Span<uint8_t>(decode_output_ptr, decode_output_size));
}

absl::Status EmbeddingLookupText::LookupPrefill(
    int token, std::vector<float>& prefill_output_vector) {
  if (prefill_output_vector.size() != floats_per_token_output_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The text embedding lookup output vector must be of size ",
        floats_per_token_output_, " but got ", prefill_output_vector.size()));
  }

  const size_t bytes_per_token = GetFloatsPerToken() * sizeof(float);
  uint8_t* output_ptr =
      reinterpret_cast<uint8_t*>(prefill_output_vector.data());
  return LookupInternal(token, absl::MakeSpan(output_ptr, bytes_per_token));
}

size_t EmbeddingLookupText::GetFloatsPerToken() {
  return floats_per_token_output_;
}

absl::Status EmbeddingLookupText::LookupPrefill(absl::Span<const int> tokens,
                                                TensorBuffer* prefill_output,
                                                size_t byte_offset) {
  if (prefill_output == nullptr) {
    return absl::InvalidArgumentError("Prefill output tensor buffer is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto prefill_output_type,
                          prefill_output->TensorType());
  const auto& prefill_output_layout = prefill_output_type.Layout();
  const auto& output_buffer_layout = output_buffer_type_.value().Layout();

  if (prefill_output_layout.Rank() != output_buffer_layout.Rank()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor from the Embedding model must be have the same "
        "number of dimensions as the requested tensor. Requested tensor dims: ",
        prefill_output_layout.Rank(),
        ". Output tensor dims: ", output_buffer_layout.Rank()));
  }

  if (prefill_output_layout.Rank() < 3) {
    return absl::UnimplementedError(
        "The output tensor provided to the Embedding Lookup function must have "
        " at least 3 dimensions.");
  }

  if (output_buffer_layout.Rank() < 3) {
    return absl::UnimplementedError(
        "The output tensor from the Embedding model must have at least 3 "
        "dimensions.");
  }

  if (prefill_output_layout.Dimensions()[0] != 1) {
    return absl::UnimplementedError(
        "The output tensor to fill from the Embedding model must be have the "
        "0th dimension as 1. Other sizes are not supported yet.");
  }

  if (prefill_output_layout.Dimensions()[1] < tokens.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output tensor to fill from the Embedding model must have a "
        "1st dimension that is at least the same size as the number of tokens. "
        "Requested tensor 1st dim: ",
        prefill_output_layout.Dimensions()[1], " but the number of tokens is ",
        tokens.size()));
  }

  for (size_t i = 2; i < prefill_output_layout.Rank(); ++i) {
    if (prefill_output_layout.Dimensions()[i] !=
        output_buffer_layout.Dimensions()[i]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The output tensor from the Embedding model must be have the same "
          "dimensions as the requested tensor for dims > 1. Requested tensor "
          "dim for ",
          i, ": ", prefill_output_layout.Dimensions()[i],
          ". Output tensor dims: ", output_buffer_layout.Dimensions()[i]));
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto prefill_output_size, prefill_output->Size());
  const size_t bytes_per_token = GetFloatsPerToken() * sizeof(float);

  if (byte_offset + bytes_per_token * tokens.size() > prefill_output_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The byte offset and the total number of bytes to be "
                     "written must not exceed the size of the output "
                     "tensor. Byte offset: ",
                     byte_offset, ". Bytes per token: ", bytes_per_token,
                     ". Number of tokens: ", tokens.size(),
                     ". Output tensor bytes: ", prefill_output->Size()));
  }

  auto prefill_output_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *prefill_output, TensorBuffer::LockMode::kWrite);
  auto prefill_output_ptr =
      reinterpret_cast<uint8_t*>(prefill_output_lock_and_addr->second);

  prefill_output_ptr += byte_offset;
  for (int token : tokens) {
    absl::Span<uint8_t> output_buffer(
        reinterpret_cast<uint8_t*>(prefill_output_ptr), bytes_per_token);
    RETURN_IF_ERROR(LookupInternal(token, output_buffer));
    prefill_output_ptr += bytes_per_token;
  }

  // If there are fewer tokens than the output tensor can hold, we need to treat
  // the remaining tokens as if they were 0.
  size_t starting_token = byte_offset / bytes_per_token + tokens.size();
  size_t num_tokens_to_fill = prefill_output_layout.Dimensions()[1];
  for (int i = starting_token; i < num_tokens_to_fill; ++i) {
    memcpy(prefill_output_ptr, default_embedding_vector_.data(),
           bytes_per_token);
    prefill_output_ptr += bytes_per_token;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupText>>
EmbeddingLookupText::Create(litert::Environment& env,
                            const litert::Model* absl_nonnull model,
                            std::optional<std::string> signature_key) {
  auto handler = std::unique_ptr<EmbeddingLookupText>(
      new EmbeddingLookupText(env, signature_key));
  RETURN_IF_ERROR(handler->Initialize(*model));
  return handler;
}

absl::Status EmbeddingLookupText::Initialize(const litert::Model& model) {
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

  LITERT_ASSIGN_OR_RETURN(compiled_model_, litert::CompiledModel::Create(
                                               env_, model.Get(), options));
  LITERT_ASSIGN_OR_RETURN(auto signatures, model.GetSignatures());

  if (signature_key_.has_value()) {
    bool found = false;
    for (const auto& signature : signatures) {
      if (signature.Key() == signature_key_.value()) {
        found = true;
        break;
      }
    }
    if (!found) {
      return absl::InvalidArgumentError(
          absl::StrCat("The provided signature key '", signature_key_.value(),
                       "' was not found in the model's signatures."));
    }
  } else {
    if (signatures.size() != 1) {
      ABSL_LOG(WARNING) << absl::StrCat(
                               "No signature key was provided. The Embedding "
                               "model is expected to "
                               "have exactly one signature but got ",
                               signatures.size())
                        << ". Using the first signature: "
                        << signatures.front().Key();
    }
    signature_key_ = signatures.front().Key();
  }

  ABSL_LOG(INFO) << "EmbeddingLookupText::Initialize Creating input buffers";
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_->CreateInputBuffers(
                                              signature_key_.value()));

  LITERT_ASSIGN_OR_RETURN(auto input_buffer_size, input_buffers_[0].Size());

  if (input_buffers_.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Embedding model must have exactly one input tensor but got ",
        input_buffers_.size()));
  }

  if (input_buffer_size != 4) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input tensor bytes must be 4 but got ", input_buffers_[0].Size()));
  }

  LITERT_ASSIGN_OR_RETURN(output_buffers_, compiled_model_->CreateOutputBuffers(
                                               signature_key_.value()));
  LITERT_ASSIGN_OR_RETURN(output_buffer_type_, output_buffers_[0].TensorType());
  const auto& output_buffer_layout = output_buffer_type_.value().Layout();

  if (output_buffers_.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Embedding model must have exactly one output tensor but got ",
        output_buffers_.size()));
  }

  if (output_buffer_type_.value().ElementType() !=
      litert::ElementType::Float32) {
    return absl::InvalidArgumentError(
        "The output tensor from the Embedding model must be of type float32.");
  }

  floats_per_token_output_ = 1;
  for (size_t i = 2; i < output_buffer_layout.Rank(); ++i) {
    floats_per_token_output_ *= output_buffer_layout.Dimensions()[i];
  }

  ABSL_LOG(INFO) << "EmbeddingLookupText initialized: "
                 << "signature=" << signature_key_.value_or("default")
                 << ", rank=" << output_buffer_layout.Rank()
                 << ", floats_per_token=" << floats_per_token_output_;

  // Initialize the default embedding vector to be the embedding of token 0.
  default_embedding_vector_.resize(floats_per_token_output_);
  RETURN_IF_ERROR(LookupInternal(
      0, absl::MakeSpan(
             reinterpret_cast<uint8_t*>(default_embedding_vector_.data()),
             floats_per_token_output_ * sizeof(float))));

  return absl::OkStatus();
}

litert::Expected<bool> EmbeddingLookupText::IsFullyAccelerated() {
  if (!compiled_model_.has_value()) {
    return litert::Error(litert::Status::kErrorRuntimeFailure,
                         "Compiled model has not been created.");
  }
  return compiled_model_->IsFullyAccelerated();
}

}  // namespace litert::lm

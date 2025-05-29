#include "runtime/components/embedding_lookup_text.h"

#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/status_macros.h"  //NOLINT

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
    return absl::InvalidArgumentError(
        "The token must be non-negative for text embedding lookup.");
  }

  // The input tensor size was verified when the model was loaded.
  input_buffers_[0].Write(absl::MakeSpan(const_cast<const int*>(&token), 1));

  compiled_model_->Run(/*signature_index=*/0, input_buffers_, output_buffers_);

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

  auto decode_output_lock_and_addr =
      ::litert::TensorBufferScopedLock::Create(*decode_output);
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

  auto prefill_output_lock_and_addr =
      ::litert::TensorBufferScopedLock::Create(*prefill_output);
  auto prefill_output_ptr =
      reinterpret_cast<uint8_t*>(prefill_output_lock_and_addr->second);

  prefill_output_ptr += byte_offset;
  for (int token : tokens) {
    // If the token is negative, it is a vision or audio token. We don't need to
    // handle those here.
    if (token >= 0) {
      absl::Span<uint8_t> output_buffer(
          reinterpret_cast<uint8_t*>(prefill_output_ptr), bytes_per_token);
      RETURN_IF_ERROR(LookupInternal(token, output_buffer));
    }
    prefill_output_ptr += bytes_per_token;
  }

  // If there are fewer tokens than the output tensor can hold, we need to treat
  // the remaining tokens as if they were 0.
  for (int i = tokens.size(); i < prefill_output_layout.Dimensions()[1]; ++i) {
    memcpy(prefill_output_ptr, default_embedding_vector_.data(),
           bytes_per_token);
    prefill_output_ptr += bytes_per_token;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<EmbeddingLookupText>>
EmbeddingLookupText::Create(litert::Model& model) {
  auto handler = std::make_unique<EmbeddingLookupText>();
  RETURN_IF_ERROR(handler->Initialize(model));
  return handler;
}

absl::Status EmbeddingLookupText::Initialize(litert::Model& model) {
  LITERT_ASSIGN_OR_RETURN(auto lrt_env, ::litert::Environment::Create({}));
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

  LITERT_ASSIGN_OR_RETURN(
      compiled_model_, litert::CompiledModel::Create(lrt_env, model, options));

  LITERT_ASSIGN_OR_RETURN(auto signatures, model.GetSignatures());

  // TODO test case?
  if (signatures.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Embedding model must have exactly one signature but got ",
        signatures.size()));
  }

  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_->CreateInputBuffers(
                                              /*signature_index=*/0));

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
                                               /*signature_index=*/0));
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

  // Initialize the default embedding vector to be the embedding of token 0.
  default_embedding_vector_.resize(floats_per_token_output_);
  RETURN_IF_ERROR(LookupInternal(
      0, absl::MakeSpan(
             reinterpret_cast<uint8_t*>(default_embedding_vector_.data()),
             floats_per_token_output_ * sizeof(float))));

  return absl::OkStatus();
}

}  // namespace litert::lm

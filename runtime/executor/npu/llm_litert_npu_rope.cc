// Copyright 2026 Google LLC.
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

#include "runtime/executor/npu/llm_litert_npu_rope.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<NpuRope> NpuRope::CreateForTest(
    const ::litert::CompiledModel* compiled_model,
    InferenceContext rope_context) {
  if (compiled_model == nullptr) {
    return absl::InvalidArgumentError(
        "Compiled model is required for NpuRope.");
  }
  return NpuRope(compiled_model, std::move(rope_context));
}

absl::StatusOr<NpuRope> NpuRope::Create(
    const ::litert::CompiledModel* npu_auxiliary_compiled_model,
    absl::string_view prefill_signature, absl::string_view decode_signature,
    absl::string_view verify_signature,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers) {
  RET_CHECK(npu_auxiliary_compiled_model != nullptr)
      << "Auxiliary compiled model cannot be null for NpuRope";
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      prefill_input_buffers, prefill_output_buffers, decode_input_buffers,
      decode_output_buffers, verify_input_buffers, verify_output_buffers;

  const std::array<absl::string_view, 4> rope_output_names = {
      RopeSignatures::kOutputPosEmbeddingLocalLow,
      RopeSignatures::kOutputPosEmbeddingHigh,
      RopeSignatures::kOutputPosEmbeddingLocalHigh,
      RopeSignatures::kOutputPosEmbeddingLow};

  auto map_rope_stage =
      [&](absl::string_view signature,
          const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              decoder_inputs,
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              in_buffers,
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
              out_buffers) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(in_buffers[RopeSignatures::kInputPos],
                            npu_auxiliary_compiled_model->CreateInputBuffer(
                                signature, RopeSignatures::kInputPos));
    in_buffers[RopeSignatures::kInputPos].Clear();
    for (const auto& name : rope_output_names) {
      if (decoder_inputs.contains(name)) {
        LITERT_ASSIGN_OR_RETURN(out_buffers[name],
                                decoder_inputs.at(name).Duplicate());
      }
    }
    return absl::OkStatus();
  };

  LITERT_RETURN_IF_ERROR(
      map_rope_stage(prefill_signature, text_decoder_prefill_input_buffers,
                     prefill_input_buffers, prefill_output_buffers));

  LITERT_RETURN_IF_ERROR(
      map_rope_stage(decode_signature, text_decoder_decode_input_buffers,
                     decode_input_buffers, decode_output_buffers));

  if (!verify_signature.empty() &&
      npu_auxiliary_compiled_model->FindSignature(verify_signature)) {
    LITERT_RETURN_IF_ERROR(
        map_rope_stage(verify_signature, text_decoder_verify_input_buffers,
                       verify_input_buffers, verify_output_buffers));
  }

  InferenceContext rope_context(
      std::move(prefill_input_buffers), std::move(prefill_output_buffers),
      std::move(decode_input_buffers), std::move(decode_output_buffers),
      std::move(verify_input_buffers), std::move(verify_output_buffers));
  return NpuRope(npu_auxiliary_compiled_model, std::move(rope_context));
}

absl::StatusOr<NpuRope> NpuRope::CreateForDrafter(
    const ::litert::CompiledModel* compiled_model,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        rope_input_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
        rope_output_buffers) {
  if (compiled_model == nullptr) {
    return absl::InvalidArgumentError(
        "Compiled model is required for NpuRope.");
  }
  InferenceContext ctx;
  ctx.decode_input_buffers = std::move(rope_input_buffers);
  ctx.decode_output_buffers = std::move(rope_output_buffers);
  return NpuRope(compiled_model, std::move(ctx));
}

absl::Status NpuRope::UpdateOutputBuffers(
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_prefill_input_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_decode_input_buffers,
    const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        text_decoder_verify_input_buffers) {
  for (const auto& [name, buf] : text_decoder_prefill_input_buffers) {
    if (rope_context_.prefill_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(rope_context_.prefill_output_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : text_decoder_decode_input_buffers) {
    if (rope_context_.decode_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(rope_context_.decode_output_buffers[name],
                              buf.Duplicate());
    }
  }
  for (const auto& [name, buf] : text_decoder_verify_input_buffers) {
    if (rope_context_.verify_output_buffers.contains(name)) {
      LITERT_ASSIGN_OR_RETURN(rope_context_.verify_output_buffers[name],
                              buf.Duplicate());
    }
  }
  return absl::OkStatus();
}

absl::Status NpuRope::RunPrefill(absl::string_view signature) const {
  RET_CHECK(compiled_model_ != nullptr) << "Compiled model is null.";
  absl::string_view sig = signature.empty() ? kPrefillRopeBase : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.prefill_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.prefill_output_buffers));
  RET_CHECK(res) << "Failed to run prefill RoPE model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuRope::RunDecode(absl::string_view signature) const {
  RET_CHECK(compiled_model_ != nullptr) << "Compiled model is null.";
  absl::string_view sig =
      signature.empty() ? RopeSignatures::kDecodeRope : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.decode_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.decode_output_buffers));
  RET_CHECK(res) << "Failed to run decode RoPE model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuRope::RunVerify(absl::string_view signature) const {
  RET_CHECK(compiled_model_ != nullptr) << "Compiled model is null.";
  absl::string_view sig =
      signature.empty() ? RopeSignatures::kVerifyRope : signature;
  auto res = compiled_model_->Run(
      sig,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.verify_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.verify_output_buffers));
  RET_CHECK(res) << "Failed to run verify RoPE model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuRope::RunDrafter() const {
  RET_CHECK(compiled_model_ != nullptr) << "Compiled model is null.";
  auto res = compiled_model_->Run(
      RopeSignatures::kMtpRope,
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.decode_input_buffers),
      const_cast<
          absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&>(
          rope_context_.decode_output_buffers));
  RET_CHECK(res) << "Failed to run drafter RoPE model: "
                 << res.Error().Message();
  return absl::OkStatus();
}

absl::Status NpuRope::SetDecodePosition(int32_t step) {
  RET_CHECK(
      rope_context_.decode_input_buffers.contains(RopeSignatures::kInputPos))
      << "RoPE decode input_pos buffer not found.";
  return SetFirstElement(
      rope_context_.decode_input_buffers[RopeSignatures::kInputPos], step);
}

absl::Status NpuRope::SetVerifyPositions(int32_t start_step,
                                         size_t num_tokens) {
  RET_CHECK(
      rope_context_.verify_input_buffers.contains(RopeSignatures::kInputPos))
      << "RoPE verify input_pos buffer not found.";
  auto& pos_buf = rope_context_.verify_input_buffers[RopeSignatures::kInputPos];
  LITERT_ASSIGN_OR_RETURN(
      auto pos_lock, ::litert::TensorBufferScopedLock::Create(
                         pos_buf, ::litert::TensorBuffer::LockMode::kWrite));
  auto* pos_ptr = static_cast<int32_t*>(pos_lock.second);
  for (size_t i = 0; i < num_tokens; ++i) {
    pos_ptr[i] = start_step + i;
  }
  return absl::OkStatus();
}

absl::Status NpuRope::SetPrefillPositions(absl::Span<const int32_t> positions) {
  RET_CHECK(
      rope_context_.prefill_input_buffers.contains(RopeSignatures::kInputPos))
      << "RoPE prefill input_pos buffer not found.";
  auto& buffer = rope_context_.prefill_input_buffers[RopeSignatures::kInputPos];
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer.PackedSize());
  RET_CHECK_GE(buffer_size, positions.size() * sizeof(int32_t));

  LITERT_ASSIGN_OR_RETURN(
      auto lock, ::litert::TensorBufferScopedLock::Create(
                     buffer, ::litert::TensorBuffer::LockMode::kWrite));
  auto* buffer_ptr = static_cast<int32_t*>(lock.second);
  std::memcpy(buffer_ptr, positions.data(), positions.size() * sizeof(int32_t));

  size_t starting_token = positions.size();
  size_t num_tokens_to_fill = buffer_size / sizeof(int32_t);
  std::memset(buffer_ptr + starting_token, 0,
              (num_tokens_to_fill - starting_token) * sizeof(int32_t));
  return absl::OkStatus();
}

}  // namespace litert::lm

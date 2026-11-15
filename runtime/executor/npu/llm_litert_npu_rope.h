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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_ROPE_H_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"

namespace litert::lm {

// Signature names for the rope signatures.
struct RopeSignatures {
  static constexpr absl::string_view kDecodeRope = "decode_rope";
  static constexpr absl::string_view kVerifyRope = "verify_rope";
  static constexpr absl::string_view kMtpRope = "rope";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kInputPos = "input_pos";
  static constexpr absl::string_view kOutputPosEmbeddingLocalLow =
      "pos_emb_local_cos";
  static constexpr absl::string_view kOutputPosEmbeddingHigh = "pos_emb_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLocalHigh =
      "pos_emb_local_sin";
  static constexpr absl::string_view kOutputPosEmbeddingLow = "pos_emb_cos";
};

// =============================================================================
// NpuRope Usage Guide:
//
// 1. Regular Prefill:
//    rope_.SetPrefillPositions(positions_span);
//    rope_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    rope_.SetDecodePosition(current_step);
//    rope_.RunDecode();
//
// 3. MTP Speculative Decoding - Draft Generation (on drafter_rope):
//    drafter_rope.SetDecodePosition(draft_step);
//    drafter_rope.RunDrafter();
//
// 4. MTP Speculative Decoding - Verification (on main_rope):
//    main_rope_.SetVerifyPositions(start_step, num_tokens);
//    main_rope_.RunVerify();
//
// 5. Dynamic Context Switching:
//    When the text decoder switches from one context length to the next,
//    its input buffer bindings change. Rebind RoPE output buffers:
//    rope_.UpdateOutputBuffers(
//        active_group.text_decoder_inference_context.prefill_input_buffers,
//        active_group.text_decoder_inference_context.decode_input_buffers,
//        active_group.text_decoder_inference_context.verify_input_buffers);
// =============================================================================
class NpuRope {
 public:
  NpuRope() = default;
  NpuRope(const NpuRope&) = delete;
  NpuRope& operator=(const NpuRope&) = delete;
  NpuRope(NpuRope&&) = default;
  NpuRope& operator=(NpuRope&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuRope> Create(
      const ::litert::CompiledModel* npu_auxiliary_compiled_model,
      absl::string_view prefill_signature, absl::string_view decode_signature,
      absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  static absl::StatusOr<NpuRope> CreateForDrafter(
      const ::litert::CompiledModel* compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          rope_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          rope_output_buffers);

  static absl::StatusOr<NpuRope> CreateForTest(
      const ::litert::CompiledModel* compiled_model,
      InferenceContext rope_context);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // Updates the internal RoPE output buffer bindings (e.g. pos_emb_cos,
  // pos_emb_sin) to point to the newly active context group's text decoder
  // input buffers when switching from one context length to the next.
  absl::Status UpdateOutputBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillPositions(absl::Span<const int32_t> positions);
  absl::Status RunPrefill(absl::string_view signature = "") const;

  // --- Stage 2: Decode (Main & Drafter) ---
  absl::Status SetDecodePosition(int32_t step);
  absl::Status RunDecode(absl::string_view signature = "") const;

  // --- Stage 3: Speculative Decoding (MTP Draft & Verify) ---
  absl::Status RunDrafter() const;
  absl::Status SetVerifyPositions(int32_t start_step, size_t num_tokens);
  absl::Status RunVerify(absl::string_view signature = "") const;

  // --- Accessors ---
  const InferenceContext& Context() const { return rope_context_; }
  InferenceContext ReleaseContext() { return std::move(rope_context_); }

 private:
  explicit NpuRope(const ::litert::CompiledModel* compiled_model,
                   InferenceContext rope_context)
      : compiled_model_(compiled_model),
        rope_context_(std::move(rope_context)) {}

  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext rope_context_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_ROPE_H_

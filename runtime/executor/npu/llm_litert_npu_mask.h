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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_MASK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_MASK_H_

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

enum class MaskUpdateMethod {
  kModel,
  kWH,
};

// Signature names for the mask signatures.
struct MaskSignatures {
  static constexpr absl::string_view kDecodeMask = "decode_mask";
  static constexpr absl::string_view kVerifyMask = "verify_mask";
  static constexpr absl::string_view kMtpMask = "mask";
  // Prefill and decode use identical tensor signature names.
  static constexpr absl::string_view kMaskInputTimeStep = "time_step";
  static constexpr absl::string_view kMaskInputTokens = "input_tokens";
  static constexpr absl::string_view kMaskInputValidMask = "valid_mask";
  static constexpr absl::string_view kMaskLocal = "mask_local";
  static constexpr absl::string_view kMaskGlobal = "mask_global";
};

// =============================================================================
// NpuMask Usage Guide:
//
// 1. Regular Prefill:
//    mask_.SetPrefillInput(internal_start_step, tokens_to_embed);
//    mask_.RunPrefill(prefill_signature);
//
// 2. Regular Single-Token Decode:
//    mask_.SetDecodeInput(current_step, token_id);
//    mask_.RunDecode();
//
// 3. MTP Speculative Decoding - Draft Generation (on drafter_mask):
//    drafter_mask.SetDecodeInput(draft_step, draft_token_id);
//    drafter_mask.RunDrafter();
//
// 4. MTP Speculative Decoding - Verification (on main_mask):
//    main_mask_.SetVerifyInput(start_step, verify_ids);
//    main_mask_.RunVerify();
//
// 5. Dynamic Context Switching:
//    When the text decoder switches from one context length to the next,
//    its input buffer bindings change. Rebind Mask output buffers:
//    mask_.UpdateOutputBuffers(
//        active_group.text_decoder_inference_context.prefill_input_buffers,
//        active_group.text_decoder_inference_context.decode_input_buffers,
//        active_group.text_decoder_inference_context.verify_input_buffers);
// =============================================================================
class NpuMask {
 public:
  NpuMask() = default;
  NpuMask(const NpuMask&) = delete;
  NpuMask& operator=(const NpuMask&) = delete;
  NpuMask(NpuMask&&) = default;
  NpuMask& operator=(NpuMask&&) = default;

  // --- Lifecycle & Creation ---
  static absl::StatusOr<NpuMask> Create(
      MaskUpdateMethod method,
      const ::litert::CompiledModel* npu_auxiliary_compiled_model,
      absl::string_view prefill_signature, absl::string_view decode_signature,
      absl::string_view verify_signature,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  static absl::StatusOr<NpuMask> CreateForDrafter(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_output_buffers);

  static absl::StatusOr<NpuMask> CreateForTest(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext mask_context);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  // Updates the internal Mask output buffer bindings (e.g. mask_local,
  // mask_global) to point to the newly active context group's text decoder
  // input buffers when switching from one context length to the next.
  absl::Status UpdateOutputBuffers(
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_prefill_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_decode_input_buffers,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          text_decoder_verify_input_buffers);

  // --- Stage 1: Prefill ---
  absl::Status SetPrefillInput(int32_t start_step,
                               absl::Span<const int> token_ids);
  absl::Status RunPrefill(absl::string_view signature = "") const;

  // --- Stage 2: Decode (Main & Drafter) ---
  absl::Status SetDecodeInput(int32_t step, int32_t token_id);
  absl::Status RunDecode(absl::string_view signature = "") const;

  // --- Stage 3: Speculative Decoding (MTP Draft & Verify) ---
  absl::Status RunDrafter() const;
  absl::Status SetVerifyInput(int32_t start_step,
                              absl::Span<const int> verify_ids);
  absl::Status RunVerify(absl::string_view signature = "") const;

  // --- Accessors ---
  MaskUpdateMethod GetMethod() const { return method_; }
  const InferenceContext& Context() const { return mask_context_; }
  InferenceContext ReleaseContext() { return std::move(mask_context_); }

 private:
  explicit NpuMask(MaskUpdateMethod method,
                   const ::litert::CompiledModel* compiled_model,
                   InferenceContext mask_context)
      : method_(method),
        compiled_model_(compiled_model),
        mask_context_(std::move(mask_context)) {}

  MaskUpdateMethod method_ = MaskUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext mask_context_;
};

// Performs manual attention mask update (CPU fallback).
absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_MASK_H_

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
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.h"
#include "runtime/proto/executor_metadata.pb.h"

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
  static constexpr absl::string_view kMaskInputTimeStep = "time_step";
  static constexpr absl::string_view kMaskInputTokens = "input_tokens";
  static constexpr absl::string_view kMaskInputValidMask = "valid_mask";
  static constexpr absl::string_view kMaskLocalContextLength =
      "local_context_length";
  static constexpr absl::string_view kMaskGlobalContextLength =
      "global_context_length";
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
//    `tokens_to_embed` are the token ids of the chunk as passed to the
//    embedder, in chunk order, where image positions are
//    ExecutorVisionData::kSpecialToken. When the layout's local (or global)
//    mask type is ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL (e.g. Gemma4 12B),
//    the hand-written mask update (MaskUpdateMethod::kWH) lets each vision
//    token attend to the later vision tokens of its image within the chunk.
//    The mask subgraph of the auxiliary model (MaskUpdateMethod::kModel), as
//    produced by the converter today, only produces causal masks and does not
//    know where the vision tokens are, so RunPrefill fails for
//    vision-bidirectional masks with kModel.
//
// 2. Regular Single-Token Decode:
//    mask_.SetDecodeInput(current_step, token_id);
//    mask_.RunDecode();
//
// 3. MTP Speculative Decoding - Draft Generation (on drafter_mask):
//    drafter_mask.SetDecodeInput(draft_step, draft_token_id);
//    drafter_mask.RunDrafter(drafter_mask_signature);
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
//
// 6. Dynamic KV Cache Resizing (dynamic models only):
//    After the auxiliary model's mask signatures have been resized to the new
//    context size, re-create the shape-only context-length inputs so that the
//    buffers match the resized signatures, then point the mask at the new
//    group's layout and rebind its output buffers as in (5):
//    mask_.RecreateContextLengthInputBuffers(
//        prefill_mask_signature, decode_mask_signature, verify_mask_signature);
//    mask_.SetLayout(&new_group.layout);
//    mask_.UpdateOutputBuffers(...);
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
          text_decoder_verify_input_buffers,
      const NpuGraphLayout* layout);

  static absl::StatusOr<NpuMask> CreateForDrafter(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_input_buffers,
      absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
          mask_output_buffers,
      const NpuGraphLayout* layout);

  static absl::StatusOr<NpuMask> CreateForTest(
      MaskUpdateMethod method, const ::litert::CompiledModel* compiled_model,
      InferenceContext mask_context, const NpuGraphLayout* layout);

  void SetCompiledModel(const ::litert::CompiledModel* compiled_model) {
    compiled_model_ = compiled_model;
  }

  void SetLayout(const NpuGraphLayout* layout) { layout_ = layout; }
  const NpuGraphLayout* GetLayout() const { return layout_; }

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

  // Re-creates the context-length input buffers (`local_context_length`,
  // `global_context_length`) of the prefill/decode/verify mask signatures.
  //
  // Dynamic mask signatures take these as extra inputs whose values are never
  // read: they only carry the context size through their shape, so that the
  // mask output shape can follow a resized KV cache. After the auxiliary
  // model's mask signatures have been resized, the compiled model rejects the
  // old buffers because their shape no longer matches the signature. Only
  // buffers whose expected shape actually changed are replaced; all other
  // input buffers (and signatures without these inputs) are left untouched.
  absl::Status RecreateContextLengthInputBuffers(
      absl::string_view prefill_signature, absl::string_view decode_signature,
      absl::string_view verify_signature);

  // --- Stage 1: Prefill ---
  // `token_ids` are the token ids of the chunk as passed to the embedder, in
  // chunk order, where image positions are `ExecutorVisionData::kSpecialToken`.
  // The ids written to the `input_tokens` buffer are clamped to be
  // non-negative: the auxiliary mask model expects real token ids, and the
  // hand-written (kWH) mask update treats -1 in that buffer as padding when the
  // model has no `valid_mask` input. An unclamped copy is kept separately to
  // apply vision-bidirectional attention in the kWH mask update.
  absl::Status SetPrefillInput(int32_t start_step,
                               absl::Span<const int> token_ids);
  absl::Status RunPrefill(absl::string_view signature = "") const;

  // --- Stage 2: Decode (Main & Drafter) ---
  absl::Status SetDecodeInput(int32_t step, int32_t token_id);
  absl::Status RunDecode(absl::string_view signature = "") const;

  // --- Stage 3: Speculative Decoding (MTP Draft & Verify) ---
  // Runs the drafter mask `signature`, or `MaskSignatures::kMtpMask` when
  // empty. Multi-context models provide one drafter mask signature per context
  // size (e.g. "mask_cache_640").
  absl::Status RunDrafter(absl::string_view signature = "") const;
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
                   InferenceContext mask_context, const NpuGraphLayout* layout)
      : method_(method),
        compiled_model_(compiled_model),
        mask_context_(std::move(mask_context)),
        layout_(layout) {}

  MaskUpdateMethod method_ = MaskUpdateMethod::kModel;
  const ::litert::CompiledModel* compiled_model_ = nullptr;
  InferenceContext mask_context_;
  const NpuGraphLayout* layout_ = nullptr;
  // Unclamped token ids of the current prefill chunk (see SetPrefillInput).
  std::vector<int> prefill_token_ids_;
  // Whether the last token of the previous prefill chunk was a vision token.
  // Only used to warn when an image is split across prefill chunks (its
  // tokens then only attend bidirectionally within each chunk); the masks do
  // not depend on it.
  bool last_prefill_token_was_vision_ = false;
};

// Performs manual attention mask update (CPU fallback).
//
// `token_ids` are the optional token ids of the chunk, in chunk order, where
// image positions are `ExecutorVisionData::kSpecialToken`. They are used to
// apply vision-bidirectional attention within
// the chunk when the layout's mask type requests it.
absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    const NpuGraphLayout& layout, absl::Span<const int> token_ids = {});

absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& out_buffers,
    int64_t sliding_window_size = 512, bool uses_ringbuffer = false,
    int64_t global_capacity = 0, int64_t local_capacity = 0,
    proto::AttentionMaskType global_mask_type =
        proto::ATTENTION_MASK_TYPE_CAUSAL,
    proto::AttentionMaskType local_mask_type =
        proto::ATTENTION_MASK_TYPE_CAUSAL,
    absl::Span<const int> token_ids = {});

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_NPU_LLM_LITERT_NPU_MASK_H_

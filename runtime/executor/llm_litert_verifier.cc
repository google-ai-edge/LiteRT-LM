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

#include "runtime/executor/llm_litert_verifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/nvidia_mtp_drafter_support.h"
#include "runtime/executor/state_interface.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kLogitsName = "logits";

absl::StatusOr<int> GetVocabSizeFromLogitsTensor(TensorBuffer& logits_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  // logits tensor shape is [batch, seq, vocab].
  return logits_tensor_type.Layout().Dimensions()[2];
}

constexpr absl::string_view kVerifySignaturePrefix = "verify";

// Returns the dimensions of `type`.
std::vector<int> Dims(const RankedTensorType& type) {
  const auto& dims = type.Layout().Dimensions();
  return std::vector<int>(dims.begin(), dims.end());
}

// The facts of the decode signature that verify signatures must agree with.
struct DecodeSignatureFacts {
  ModelSignatures signatures;
  int vocab_size = 0;
  // Sorted names of the KV cache inputs and outputs, and of the LoRA inputs.
  std::vector<absl::string_view> state_inputs;
  std::vector<absl::string_view> state_outputs;
  std::vector<absl::string_view> lora_inputs;
};

// Returns the sorted names of `names` for which `keep` holds.
template <typename Predicate>
std::vector<absl::string_view> SortedNames(
    const std::vector<absl::string_view>& names, Predicate keep) {
  std::vector<absl::string_view> result;
  for (absl::string_view name : names) {
    if (keep(name)) {
      result.push_back(name);
    }
  }
  absl::c_sort(result);
  return result;
}

absl::StatusOr<DecodeSignatureFacts> GetDecodeSignatureFacts(
    const SimpleSignature& decode, const StateInterface& state) {
  DecodeSignatureFacts facts;
  ABSL_ASSIGN_OR_RETURN(facts.signatures,
                        GetModelSignaturesFromInputOutputNames(
                            decode.InputNames(), decode.OutputNames()));
  LITERT_ASSIGN_OR_RETURN(
      auto logits_type,
      decode.OutputTensorType(facts.signatures.output_logits));
  const std::vector<int> logits_dims = Dims(logits_type);
  RET_CHECK_EQ(logits_dims.size(), 3);
  facts.vocab_size = logits_dims[2];
  auto in_state = [&](absl::string_view name) { return state.Contains(name); };
  facts.state_inputs = SortedNames(decode.InputNames(), in_state);
  facts.state_outputs = SortedNames(decode.OutputNames(), in_state);
  facts.lora_inputs = SortedNames(decode.InputNames(), IsLoRABufferInputName);
  return facts;
}

// Returns OK if the input `name` of `verify` has the element type and the
// shape of decode's input `decode_name` ([1, 1, ...]), with `width` rows
// instead of one. The embedding lookups that fill these inputs in a pass are
// the ones that fill them in decode.
absl::Status CheckRowsLikeDecodeInput(
    const SimpleSignature& verify, const SimpleSignature& decode,
    absl::string_view name, const std::optional<std::string>& decode_name,
    int width) {
  if (!decode_name.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("It takes ", name, ", but decode doesn't."));
  }
  LITERT_ASSIGN_OR_RETURN(auto decode_type,
                          decode.InputTensorType(*decode_name));
  std::vector<int> expected_dims = Dims(decode_type);
  if (expected_dims.size() < 3 || expected_dims[0] != 1 ||
      expected_dims[1] != 1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Decode's ", *decode_name, " must have the shape [1, 1, ...]."));
  }
  expected_dims[1] = width;
  LITERT_ASSIGN_OR_RETURN(auto type, verify.InputTensorType(name));
  if (Dims(type) != expected_dims ||
      type.ElementType() != decode_type.ElementType()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Its ", name, " must have the shape [",
                     absl::StrJoin(expected_dims, ", "),
                     "] and the element type of decode's ", *decode_name, "."));
  }
  return absl::OkStatus();
}

// Returns the width of the verify signature `verify`, or why draft
// verification can't use it.
absl::StatusOr<int> CheckVerifySignature(
    const SimpleSignature& verify, const SimpleSignature& decode,
    const DecodeSignatureFacts& decode_facts, const StateInterface& state,
    bool has_embedding_lookup, bool has_per_layer_embedding_lookup) {
  ABSL_ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(
          verify.InputNames(), verify.OutputNames(), /*strict=*/false));

  if (signatures.input_positions.empty()) {
    return absl::FailedPreconditionError("It has no positions input.");
  }
  LITERT_ASSIGN_OR_RETURN(auto input_pos_type,
                          verify.InputTensorType(signatures.input_positions));
  const std::vector<int> input_pos_dims = Dims(input_pos_type);
  if (input_pos_dims.size() != 1 || input_pos_dims[0] < 1) {
    return absl::FailedPreconditionError(
        "Its positions must have the shape [W] with W >= 1.");
  }
  // The verifier writes the positions and tokens as int32.
  if (input_pos_type.ElementType() != ElementType::Int32) {
    return absl::FailedPreconditionError("Its positions must be int32.");
  }
  const int width = input_pos_dims[0];

  if (signatures.output_logits.empty()) {
    return absl::FailedPreconditionError("It has no logits output.");
  }
  LITERT_ASSIGN_OR_RETURN(auto logits_type,
                          verify.OutputTensorType(signatures.output_logits));
  if (Dims(logits_type) !=
      std::vector<int>{1, width, decode_facts.vocab_size}) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Its logits must have the shape [1, ", width, ", ",
        decode_facts.vocab_size, "] (the vocabulary size of decode)."));
  }

  if (!signatures.input_tokens.empty()) {
    // The verifier fills only the tokens of such signatures.
    if (signatures.input_embeddings.has_value() ||
        signatures.input_per_layer_embeddings.has_value()) {
      return absl::FailedPreconditionError(
          "It takes both tokens and embeddings.");
    }
    LITERT_ASSIGN_OR_RETURN(auto tokens_type,
                            verify.InputTensorType(signatures.input_tokens));
    if (Dims(tokens_type) != std::vector<int>{1, width}) {
      return absl::FailedPreconditionError(
          absl::StrCat("Its tokens must have the shape [1, ", width, "]."));
    }
    if (tokens_type.ElementType() != ElementType::Int32) {
      return absl::FailedPreconditionError("Its tokens must be int32.");
    }
  } else if (!signatures.input_embeddings.has_value()) {
    return absl::FailedPreconditionError(
        "It takes neither tokens nor embeddings.");
  } else if (!has_embedding_lookup) {
    return absl::FailedPreconditionError(
        "It takes embeddings, but the model has no embedding lookup.");
  } else if (signatures.input_per_layer_embeddings.has_value() &&
             !has_per_layer_embedding_lookup) {
    return absl::FailedPreconditionError(
        "It takes per-layer embeddings, but the model has no per-layer "
        "embedding lookup.");
  } else {
    ABSL_RETURN_IF_ERROR(CheckRowsLikeDecodeInput(
        verify, decode, *signatures.input_embeddings,
        decode_facts.signatures.input_embeddings, width));
    if (signatures.input_per_layer_embeddings.has_value()) {
      ABSL_RETURN_IF_ERROR(CheckRowsLikeDecodeInput(
          verify, decode, *signatures.input_per_layer_embeddings,
          decode_facts.signatures.input_per_layer_embeddings, width));
    }
  }

  if (signatures.input_attn_mask_local.has_value() ||
      decode_facts.signatures.input_attn_mask_local.has_value()) {
    return absl::FailedPreconditionError(
        "Models with a local attention mask (ring-buffer KV caches) are not "
        "supported.");
  }
  if (signatures.input_int32_param.has_value() !=
      decode_facts.signatures.input_int32_param.has_value()) {
    return absl::FailedPreconditionError(
        "It must have a param tensor if and only if decode has one.");
  }

  auto in_state = [&](absl::string_view name) { return state.Contains(name); };
  const std::vector<absl::string_view> state_inputs =
      SortedNames(verify.InputNames(), in_state);
  const std::vector<absl::string_view> state_outputs =
      SortedNames(verify.OutputNames(), in_state);
  bool same_kv_cache = state_inputs == decode_facts.state_inputs &&
                       state_outputs == decode_facts.state_outputs;
  for (absl::string_view name : state_inputs) {
    if (!same_kv_cache) {
      break;
    }
    LITERT_ASSIGN_OR_RETURN(auto type, verify.InputTensorType(name));
    LITERT_ASSIGN_OR_RETURN(auto decode_type, decode.InputTensorType(name));
    same_kv_cache = type == decode_type;
  }
  for (absl::string_view name : state_outputs) {
    if (!same_kv_cache) {
      break;
    }
    LITERT_ASSIGN_OR_RETURN(auto type, verify.OutputTensorType(name));
    LITERT_ASSIGN_OR_RETURN(auto decode_type, decode.OutputTensorType(name));
    same_kv_cache = type == decode_type;
  }
  if (!same_kv_cache) {
    return absl::FailedPreconditionError(
        "It must have the same KV cache tensors as decode.");
  }

  if (SortedNames(verify.InputNames(), IsLoRABufferInputName) !=
      decode_facts.lora_inputs) {
    return absl::FailedPreconditionError(
        "It must have the same LoRA inputs as decode.");
  }

  // Every other input must be one that the verifier fills.
  for (absl::string_view name : verify.InputNames()) {
    if (state.Contains(name) || IsLoRABufferInputName(name) ||
        name == signatures.input_positions || name == signatures.input_tokens ||
        name == signatures.input_attn_mask ||
        name == signatures.input_embeddings ||
        name == signatures.input_per_layer_embeddings ||
        name == signatures.input_int32_param) {
      continue;
    }
    return absl::FailedPreconditionError(
        absl::StrCat("It has an unknown input: ", name, "."));
  }

  if (width > state.GetNumEntries()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Its width ", width, " exceeds the KV cache size ",
                     state.GetNumEntries(), "."));
  }
  return width;
}

}  // namespace

absl::StatusOr<std::vector<VerifySignatureInfo>> FindVerifySignatures(
    CompiledModel& model, absl::string_view decode_signature_key,
    const StateInterface& state,
    absl::Span<const std::string> selected_signatures,
    bool has_embedding_lookup, bool has_per_layer_embedding_lookup) {
  LITERT_ASSIGN_OR_RETURN(auto keys, model.GetSignatureKeys());
  LITERT_ASSIGN_OR_RETURN(SimpleSignature decode,
                          model.FindSignature(decode_signature_key));
  ABSL_ASSIGN_OR_RETURN(DecodeSignatureFacts decode_facts,
                        GetDecodeSignatureFacts(decode, state));

  std::vector<VerifySignatureInfo> infos;
  for (absl::string_view key : keys) {
    if (!absl::StartsWith(key, kVerifySignaturePrefix)) {
      continue;
    }
    VerifySignatureInfo& info = infos.emplace_back();
    info.key = std::string(key);
    if (!selected_signatures.empty() &&
        !absl::c_linear_search(selected_signatures, key)) {
      info.status = absl::FailedPreconditionError(
          "It is not among the selected signatures.");
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(SimpleSignature verify, model.FindSignature(key));
    absl::StatusOr<int> width = CheckVerifySignature(
        verify, decode, decode_facts, state, has_embedding_lookup,
        has_per_layer_embedding_lookup);
    if (width.ok()) {
      info.width = *width;
    } else {
      info.status = width.status();
    }
  }
  return infos;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> LlmLiteRtVerifier::Create(
    Environment& env, CompiledModel& base_model,
    absl::string_view signature_key,
    const LlmExecutorSettings& executor_settings,
    EmbeddingLookupManager* embedding_lookup,
    EmbeddingLookupManager* per_layer_embedding_lookup,
    const proto::ExecutorMetadata* executor_metadata) {
  const ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  const Backend backend = executor_settings.GetBackend();
  // The sampler factory has no NPU sampler; GetSamplerBackend() falls back to
  // the CPU sampler there unless a sampler backend is configured explicitly.
  ABSL_ASSIGN_OR_RETURN(const Backend sampler_backend,
                        GetSamplerBackend(executor_settings));
  const bool use_fp16_precision =
      backend == Backend::GPU &&
      activation_data_type == ActivationDataType::FLOAT16;

  LITERT_ASSIGN_OR_RETURN(SimpleSignature signature,
                          base_model.FindSignature(signature_key));

  // The KV cache buffers come with each pass (see `Run()`).
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
  for (absl::string_view input_name : signature.InputNames()) {
    if (absl::StrContains(input_name, "kv")) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        base_model.CreateInputBuffer(signature.Key(), input_name));
    input_buffers[input_name] = std::move(input_buffer);
  }
  LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                          base_model.GetSignatureIndex(signature.Key()));
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
    absl::string_view output_name = signature.OutputNames()[i];
    if (absl::StrContains(output_name, "kv")) {
      continue;
    }
    if (output_name == kLogitsName && use_fp16_precision) {
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateFP16OutputBuffer(env, base_model, signature_index, output_name,
                                 i));
      output_buffers[output_name] = std::move(output_buffer);
    } else {
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          base_model.CreateOutputBuffer(signature.Key(), output_name));
      output_buffers[output_name] = std::move(output_buffer);
    }
  }

  ABSL_ASSIGN_OR_RETURN(ModelSignatures signatures,
                        GetModelSignaturesFromInputOutputNames(
                            signature.InputNames(), signature.OutputNames(),
                            /*strict=*/false));
  // Like decode, the positions input is `input_pos` or `positions`.
  if (signatures.input_positions.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Signature ", signature_key, " has no positions input."));
  }
  const std::string& positions_name = signatures.input_positions;
  LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                          signature.InputTensorType(positions_name));
  // Expected shape: [W].
  const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
  if (input_pos_dims.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ", positions_name, " input of signature ", signature_key,
        " must have rank 1, got rank ", input_pos_dims.size(), "."));
  }
  const int width = input_pos_dims[0];
  // Dynamic dimensions are negative.
  if (width < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ", positions_name, " input of signature ", signature_key,
        " must have a static positive width, got ", width, "."));
  }

  auto logits = output_buffers.find(kLogitsName);
  if (logits == output_buffers.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Signature ", signature_key, " has no ", kLogitsName, " output."));
  }
  ABSL_ASSIGN_OR_RETURN(int vocab_size,
                        GetVocabSizeFromLogitsTensor(logits->second));

  ActivationDataType logits_data_type;
  {
    LITERT_ASSIGN_OR_RETURN(auto logits_type, logits->second.TensorType());
    if (logits_type.ElementType() == ElementType::Float16) {
      logits_data_type = ActivationDataType::FLOAT16;
    } else if (logits_type.ElementType() == ElementType::Float32) {
      logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported verifier logits type");
    }
  }

  // On NVIDIA's TensorRT-RTX backend (Backend::NPU) the argmax runs on the
  // device next to the logits; elsewhere this is CreateSampler().
  ABSL_ASSIGN_OR_RETURN(
      auto sampler,
      CreateMtpGreedySampler(backend, /*batch_size=*/1, GreedySamplerParams(),
                             env, /*sequence_size=*/width, vocab_size,
                             logits_data_type));
  LITERT_ASSIGN_OR_RETURN(auto id_tensor,
                          CreateTensorBuffer<int32_t>({1, width}));
  if (signatures.input_tokens.empty()) {
    if (!signatures.input_embeddings.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Signature ", signature_key,
                       " takes neither tokens nor embeddings."));
    }
    if (embedding_lookup == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Signature ", signature_key,
                       " takes embeddings, but there is no embedding lookup."));
    }
    if (signatures.input_per_layer_embeddings.has_value() &&
        per_layer_embedding_lookup == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Signature ", signature_key,
          " takes per-layer embeddings, but there is no per-layer embedding "
          "lookup."));
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      auto mask_runner,
      LiteRtLogitMaskRunner::Create(env, GetHwAcceleratorForBackend(backend)));

  // Duplicates of the buffers to run the passes with, which avoid creating
  // the buffer maps for each pass.
  absl::flat_hash_map<absl::string_view, TensorBuffer> active_input_buffers;
  for (const auto& [name, buffer] : input_buffers) {
    LITERT_ASSIGN_OR_RETURN(active_input_buffers[name], buffer.Duplicate());
  }
  absl::flat_hash_map<absl::string_view, TensorBuffer> active_output_buffers;
  for (const auto& [name, buffer] : output_buffers) {
    LITERT_ASSIGN_OR_RETURN(active_output_buffers[name], buffer.Duplicate());
  }

  // Same default as the prefill's padding.
  const int pad_token_id = std::max(executor_settings.GetPadTokenId(), 0);
  return absl::WrapUnique(new LlmLiteRtVerifier(
      env, base_model, std::move(signature), embedding_lookup,
      per_layer_embedding_lookup, width, vocab_size, pad_token_id,
      std::move(signatures), GetAttentionMaskParams(executor_metadata),
      logits_data_type, sampler_backend, std::move(sampler),
      std::move(mask_runner), std::move(output_buffers),
      std::move(active_input_buffers), std::move(active_output_buffers),
      std::move(id_tensor)));
}

absl::Status LlmLiteRtVerifier::Run(
    int position, absl::Span<const int> ids,
    const LitertState::StateBuffers& state_buffers) {
  const int num_tokens = ids.size();
  if (num_tokens < 1 || num_tokens > width_) {
    return absl::InvalidArgumentError(
        absl::StrCat("A pass of signature ", signature_.Key(), " takes 1 to ",
                     width_, " tokens, got ", num_tokens, "."));
  }
  // Invalidates the tokens of the previous pass until this one has started.
  num_pass_tokens_ = 0;

  {
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(
            active_input_buffers_[signatures_.input_positions],
            TensorBuffer::LockMode::kWrite));
    auto* input_pos_ptr = static_cast<int32_t*>(input_pos_lock_and_addr.second);
    for (int i = 0; i < width_; ++i) {
      // Like the prefill, the padding rows take position 0.
      input_pos_ptr[i] = i < num_tokens ? position + i : 0;
    }
  }

  // Only the rows of the tokens attend to anything.
  if (signatures_.input_attn_mask.has_value()) {
    auto& mask_buf = active_input_buffers_[*signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(mask_buf, /*is_f16=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/num_tokens,
                                           attn_params_.global_type));
  }
  if (signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf = active_input_buffers_[*signatures_.input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(mask_buf, /*is_f16=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/num_tokens, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kVerify));
  }

  if (!signatures_.input_tokens.empty()) {
    ABSL_RETURN_IF_ERROR(FillTokens(ids));
  } else {
    ABSL_RETURN_IF_ERROR(
        FillEmbeddings(*embedding_lookup_, *signatures_.input_embeddings, ids));
    if (signatures_.input_per_layer_embeddings.has_value()) {
      ABSL_RETURN_IF_ERROR(
          FillEmbeddings(*per_layer_embedding_lookup_,
                         *signatures_.input_per_layer_embeddings, ids));
    }
  }

  for (const auto& [input_name, input_buffer] : state_buffers.input_buffers) {
    LITERT_ASSIGN_OR_RETURN(active_input_buffers_[input_name],
                            input_buffer.Duplicate());
  }
  if (signatures_.input_int32_param.has_value()) {
    // Only the rows of the tokens update the KV cache.
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_input_buffers_[*signatures_.input_int32_param], position,
        num_tokens));
  }

  for (const auto& [output_name, output_buffer] :
       state_buffers.output_buffers) {
    LITERT_ASSIGN_OR_RETURN(active_output_buffers_[output_name],
                            output_buffer.Duplicate());
  }
  for (auto& [output_name, output_buffer] : active_output_buffers_) {
    LITERT_RETURN_IF_ERROR(output_buffer.ClearEvent());
  }

  bool async = true;
  LITERT_RETURN_IF_ERROR(base_model_.RunAsync(
      signature_.Key(), active_input_buffers_, active_output_buffers_, async));
  num_pass_tokens_ = num_tokens;
  return absl::OkStatus();
}

absl::Status LlmLiteRtVerifier::FillTokens(absl::Span<const int> ids) {
  LITERT_ASSIGN_OR_RETURN(auto tokens_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              active_input_buffers_[signatures_.input_tokens],
                              TensorBuffer::LockMode::kWrite));
  auto* tokens_ptr = static_cast<int32_t*>(tokens_lock_and_addr.second);
  const int num_tokens = ids.size();
  for (int i = 0; i < width_; ++i) {
    tokens_ptr[i] = i < num_tokens ? ids[i] : pad_token_id_;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtVerifier::FillEmbeddings(EmbeddingLookupManager& lookup,
                                               absl::string_view name,
                                               absl::Span<const int> ids) {
  TensorBuffer& embeddings = active_input_buffers_[name];
  ABSL_RETURN_IF_ERROR(
      lookup.LookupPrefill(ids, &embeddings, /*token_offset=*/0));
  const int num_tokens = ids.size();
  if (num_tokens == width_) {
    return absl::OkStatus();
  }
  // The padding rows keep finite values: zeros.
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, embeddings.PackedSize());
  const size_t row_size = packed_size / width_;
  LITERT_ASSIGN_OR_RETURN(auto embeddings_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              embeddings, TensorBuffer::LockMode::kWrite));
  auto* embeddings_ptr = static_cast<char*>(embeddings_lock_and_addr.second);
  std::memset(embeddings_ptr + num_tokens * row_size, 0,
              (width_ - num_tokens) * row_size);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmLiteRtVerifier::MaskAndSample(
    absl::Span<const std::unique_ptr<LogitMask>> masks) {
  if (num_pass_tokens_ == 0) {
    return absl::FailedPreconditionError("No pass has been run.");
  }
  if (!masks.empty() && static_cast<int>(masks.size()) != width_) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected no masks or ", width_, " masks, got ", masks.size(), "."));
  }
  auto logits = active_output_buffers_.find(kLogitsName);
  RET_CHECK(logits != active_output_buffers_.end());
  if (!masks.empty()) {
    ABSL_RETURN_IF_ERROR(mask_runner_->ApplySequence(logits->second, masks));
  }
  ABSL_RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits->second, id_tensor_, /*scores_tensor=*/nullptr));
  LITERT_ASSIGN_OR_RETURN(auto id_vector,
                          CopyFromTensorBuffer<int32_t>(id_tensor_));
  RET_CHECK_EQ(static_cast<int>(id_vector.size()), width_);
  // The tokens of the padding rows are meaningless.
  id_vector.resize(num_pass_tokens_);
  return id_vector;
}

absl::StatusOr<std::shared_ptr<std::default_random_engine>>
LlmLiteRtVerifier::UpdateSamplerConfig(
    const proto::SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  if (rand_gen == nullptr) {
    // The sampler ignores `sampler_params.seed()` unless it is recreated, so
    // hand it a generator for the seed explicitly.
    rand_gen = own_rand_gen_.Get(sampler_params.seed());
  }
  ABSL_RETURN_IF_ERROR(UpdateOrRecreateSampler(
      sampler_, sampler_max_top_k_, sampler_backend_,
      /*batch_size=*/1, sampler_params, rand_gen, std::cref(env_),
      /*sequence_size=*/width_, vocab_size_, logits_data_type_));
  return rand_gen;
}

absl::StatusOr<TensorBuffer*> LlmLiteRtVerifier::GetOutputBuffer(
    absl::string_view name) {
  auto it = output_buffers_.find(name);
  if (it == output_buffers_.end()) {
    return absl::NotFoundError(absl::StrCat("Signature ", signature_.Key(),
                                            " has no output ", name, "."));
  }
  return &it->second;
}

}  // namespace litert::lm

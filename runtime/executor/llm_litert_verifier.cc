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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kInputPositionsName = "input_pos";
constexpr absl::string_view kLogitsName = "logits";

absl::StatusOr<int> GetVocabSizeFromLogitsTensor(TensorBuffer& logits_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  // logits tensor shape is [batch, seq, vocab].
  return logits_tensor_type.Layout().Dimensions()[2];
}

}  // namespace

absl::StatusOr<std::unique_ptr<LlmLiteRtVerifier>> LlmLiteRtVerifier::Create(
    Environment& env, CompiledModel& base_model,
    absl::string_view signature_key,
    const LlmExecutorSettings& executor_settings,
    EmbeddingLookupManager& embedding_lookup,
    std::optional<std::reference_wrapper<EmbeddingLookupManager>>
        per_layer_embedding_lookup,
    const proto::ExecutorMetadata* executor_metadata) {
  const ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  const Backend backend = executor_settings.GetBackend();
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

  LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                          signature.InputTensorType(kInputPositionsName));
  // Expected shape: [W].
  const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
  if (input_pos_dims.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ", kInputPositionsName, " input of signature ", signature_key,
        " must have rank 1, got rank ", input_pos_dims.size(), "."));
  }
  const int width = input_pos_dims[0];
  // Dynamic dimensions are negative.
  if (width < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ", kInputPositionsName, " input of signature ", signature_key,
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

  ABSL_ASSIGN_OR_RETURN(
      auto sampler,
      CreateSampler(backend, /*batch_size=*/1, GreedySamplerParams(), env,
                    /*sequence_size=*/width, vocab_size, logits_data_type));
  LITERT_ASSIGN_OR_RETURN(auto id_tensor,
                          CreateTensorBuffer<int32_t>({1, width}));
  ABSL_ASSIGN_OR_RETURN(ModelSignatures signatures,
                        GetModelSignaturesFromInputOutputNames(
                            signature.InputNames(), signature.OutputNames(),
                            /*strict=*/false));
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

  return absl::WrapUnique(new LlmLiteRtVerifier(
      env, base_model, std::move(signature), embedding_lookup,
      per_layer_embedding_lookup, width, vocab_size, std::move(signatures),
      GetAttentionMaskParams(executor_metadata), logits_data_type, backend,
      std::move(sampler), std::move(mask_runner), std::move(output_buffers),
      std::move(active_input_buffers), std::move(active_output_buffers),
      std::move(id_tensor)));
}

absl::Status LlmLiteRtVerifier::Run(
    int position, absl::Span<const int> ids,
    const LitertState::StateBuffers& state_buffers) {
  if (static_cast<int>(ids.size()) != width_) {
    return absl::InvalidArgumentError(
        absl::StrCat("A pass of signature ", signature_.Key(), " takes ",
                     width_, " tokens, got ", ids.size(), "."));
  }

  {
    LITERT_ASSIGN_OR_RETURN(auto input_pos_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                active_input_buffers_[kInputPositionsName],
                                TensorBuffer::LockMode::kWrite));
    auto* input_pos_ptr = static_cast<int32_t*>(input_pos_lock_and_addr.second);
    for (int i = 0; i < width_; ++i) {
      input_pos_ptr[i] = position + i;
    }
  }

  if (signatures_.input_attn_mask.has_value()) {
    auto& mask_buf = active_input_buffers_[*signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(mask_buf, /*is_f16=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/width_,
                                           attn_params_.global_type));
  }
  if (signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf = active_input_buffers_[*signatures_.input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(mask_buf, /*is_f16=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/width_, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kVerify));
  }

  if (signatures_.input_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_.LookupPrefill(
        ids, &active_input_buffers_[*signatures_.input_embeddings],
        /*token_offset=*/0));
  }
  if (per_layer_embedding_lookup_.has_value() &&
      signatures_.input_per_layer_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->get().LookupPrefill(
        ids, &active_input_buffers_[*signatures_.input_per_layer_embeddings],
        /*token_offset=*/0));
  }

  for (const auto& [input_name, input_buffer] : state_buffers.input_buffers) {
    LITERT_ASSIGN_OR_RETURN(active_input_buffers_[input_name],
                            input_buffer.Duplicate());
  }
  if (signatures_.input_int32_param.has_value()) {
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_input_buffers_[*signatures_.input_int32_param], position,
        width_));
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
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmLiteRtVerifier::MaskAndSample(
    absl::Span<const std::unique_ptr<LogitMask>> masks) {
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

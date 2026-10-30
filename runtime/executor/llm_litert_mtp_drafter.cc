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

#include "runtime/executor/llm_litert_mtp_drafter.h"

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

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_verifier.h"
#include "runtime/executor/nvidia_mtp_drafter_support.h"
#include "runtime/executor/state_interface.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

constexpr bool kEnableMtpDrafterLogs = false;

#define MTP_DRAFTER_LOG() \
  ABSL_LOG_IF(INFO, kEnableMtpDrafterLogs) << "MTP Drafter - "

constexpr absl::string_view kVerifySignatureRunner = "verify";

absl::StatusOr<std::unique_ptr<Sampler>> CreateGreedySampler(
    const Environment& env, Backend backend, int output_heads,
    int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  // On NVIDIA's TensorRT-RTX backend (Backend::NPU) the argmax runs on the
  // device next to the logits; elsewhere this is CreateSampler().
  return CreateMtpGreedySampler(backend, output_heads, GreedySamplerParams(),
                                env, sequence_size, vocab_size,
                                activation_data_type);
}

absl::Status ConcatenateEmbeddingsAndActivations(
    const std::vector<float>& embedding_vector,
    TensorBuffer& projected_activations, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(projected_activations,
                                     TensorBuffer::LockMode::kRead));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(projected_activations_lock_and_addr.second),
         chunk_size_bytes);
  return absl::OkStatus();
}

absl::Status ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
    const std::vector<float>& embedding_vector,
    TensorBuffer& verifier_projected_activations,
    int last_verified_token_id_idx, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto verifier_projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(verifier_projected_activations,
                                     TensorBuffer::LockMode::kRead));
  // Grab the last valid activation from verifier output buffer with shape
  // [batch, draft_steps + 1, hidden_size].
  // Offset is as such the last verified token id's index multiplied by the
  // hiden_size in bytes.
  size_t offset = last_verified_token_id_idx * chunk_size_bytes;
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(
             verifier_projected_activations_lock_and_addr.second) +
             offset,
         chunk_size_bytes);
  return absl::OkStatus();
}

}  // namespace

absl::Status UpdateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    litert::Options& compilation_options) {
  switch (executor_settings.GetBackend()) {
    case Backend::GPU: {
      LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                              compilation_options.GetGpuOptions());
      gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
      gpu_compilation_options.AddBufferStorageTensorPattern("kv_cache_");
      gpu_compilation_options.AddExternalTensorPattern("param_tensor");
      gpu_compilation_options.AddBufferStorageTensorPattern("param_tensor");
      break;
    }
    case Backend::CPU:
    case Backend::NPU: {
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  return absl::OkStatus();
}
LlmLiteRtMtpDrafter::~LlmLiteRtMtpDrafter() {
  ABSL_VLOG(1) << "Num drafted tokens: " << num_drafted_tokens_;
  ABSL_VLOG(1) << "Num verified tokens: " << num_verified_tokens_;
  if (num_drafted_tokens_ > 0) {
    ABSL_LOG(INFO) << "MTP Drafter - Success rate: "
                   << static_cast<double>(num_verified_tokens_) /
                          num_drafted_tokens_;
  }
}

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(
    Environment& env, ModelResources& resources,
    const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
    EmbeddingLookupManager& embedding_manager,
    std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
    const proto::ExecutorMetadata* executor_metadata) {
  ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  auto cache_suffix = std::string(ExecutorSettingsBase::kMtpDrafterCacheSuffix);
  ABSL_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, activation_data_type,
                               /*signatures=*/std::nullopt,
                               /*cache_suffix=*/cache_suffix));
  ABSL_RETURN_IF_ERROR(
      UpdateCompilationOptions(executor_settings, compilation_options));
  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, proto::LlmMetadata::TF_LITE_MTP_DRAFTER, compilation_options));
  ABSL_ASSIGN_OR_RETURN(
      auto model,
      resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_MTP_DRAFTER));
  ABSL_RETURN_IF_ERROR(AddNvidiaSharedValueCacheOptions(
      executor_settings.GetBackend(), *model, base_model, compilation_options));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, model->Get(), compilation_options));
  ABSL_ASSIGN_OR_RETURN(
      auto base_model_desc,
      resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_PREFILL_DECODE));

  return Create(env, std::move(compiled_model), executor_settings, base_model,
                *base_model_desc, embedding_manager, ple_manager,
                executor_metadata);
}

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(
    Environment& env, CompiledModel mtp_drafter_model,
    const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
    const Model& base_model_desc, EmbeddingLookupManager& embedding_manager,
    std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
    const proto::ExecutorMetadata* executor_metadata) {
  ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  const Backend backend = executor_settings.GetBackend();
  bool use_fp16_precision = backend == Backend::GPU &&
                            activation_data_type == ActivationDataType::FLOAT16;

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_output_buffers;
  std::vector<std::string> kv_cache_input_names;
  LITERT_ASSIGN_OR_RETURN(
      SimpleSignature drafter_signature,
      mtp_drafter_model.GetSignature(/*signature_index=*/0));
  {
    for (absl::string_view input_name : drafter_signature.InputNames()) {
      if (absl::StartsWith(input_name, "kv_cache_")) {
        kv_cache_input_names.emplace_back(input_name);
        continue;
      }

      LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                              mtp_drafter_model.CreateInputBuffer(
                                  drafter_signature.Key(), input_name));
      mtp_drafter_input_buffers[input_name] = std::move(input_buffer);
    }

    for (size_t i = 0; i < drafter_signature.OutputNames().size(); ++i) {
      absl::string_view output_name = drafter_signature.OutputNames()[i];
      if (output_name == "logits" && use_fp16_precision) {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            CreateFP16OutputBuffer(env, mtp_drafter_model,
                                   /*signature_index=*/0, output_name, i));
        mtp_drafter_output_buffers[output_name] = std::move(output_buffer);
      } else {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                mtp_drafter_model.CreateOutputBuffer(
                                    drafter_signature.Key(), output_name));
        mtp_drafter_output_buffers[output_name] = std::move(output_buffer);
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LlmLiteRtVerifier> verifier,
      LlmLiteRtVerifier::Create(
          env, base_model, kVerifySignatureRunner, executor_settings,
          &embedding_manager,
          ple_manager.has_value() ? &ple_manager->get() : nullptr,
          executor_metadata));
  // The verify pass covers the token before the drafted tokens.
  const int num_draft_steps = verifier->width() - 1;

  ActivationDataType drafter_logits_data_type;
  int drafter_vocab_size;
  {
    auto drafter_logits = mtp_drafter_output_buffers.find("logits");
    if (drafter_logits == mtp_drafter_output_buffers.end()) {
      return absl::InvalidArgumentError("The MTP drafter has no logits.");
    }
    LITERT_ASSIGN_OR_RETURN(auto drafter_logits_type,
                            drafter_logits->second.TensorType());
    if (drafter_logits_type.ElementType() == ElementType::Float16) {
      drafter_logits_data_type = ActivationDataType::FLOAT16;
    } else if (drafter_logits_type.ElementType() == ElementType::Float32) {
      drafter_logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported drafter logits type");
    }
    // The drafter logits shape is [batch, seq, vocab].
    const auto& drafter_logits_dims = drafter_logits_type.Layout().Dimensions();
    RET_CHECK_EQ(drafter_logits_dims.size(), 3);
    drafter_vocab_size = drafter_logits_dims[2];
  }
  // The drafted tokens are verified by the base model and masked by the same
  // constraint, so the drafter and the base model must share a vocabulary.
  if (drafter_vocab_size != verifier->vocab_size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The vocabulary size of the MTP drafter (", drafter_vocab_size,
        ") differs from the base model's (", verifier->vocab_size(), ")."));
  }
  if (num_draft_steps < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MTP drafting needs a verify signature of width 2 or more, got ",
        verifier->width(), "."));
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto drafter_projected_activations_type,
        drafter_signature.OutputTensorType("projected_activations"));
    if (drafter_projected_activations_type.ElementType() !=
        ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Unsupported drafter projected activations type: must be float32");
    }
  }

  {
    ABSL_ASSIGN_OR_RETURN(TensorBuffer * verifier_activations,
                          verifier->GetOutputBuffer("activations"));
    LITERT_ASSIGN_OR_RETURN(auto verifier_activations_type,
                            verifier_activations->TensorType());
    if (verifier_activations_type.ElementType() != ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Unsupported verifier activations type: must be float32");
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      auto drafter_sampler,
      CreateGreedySampler(env, backend,
                          /*output_heads=*/1,
                          /*sequence_size=*/1, drafter_vocab_size,
                          drafter_logits_data_type));

  LITERT_ASSIGN_OR_RETURN(auto drafter_id_tensor,
                          CreateTensorBuffer<int32_t>({1, 1}));

  ABSL_ASSIGN_OR_RETURN(
      auto drafter_model_signatures,
      GetModelSignaturesFromInputOutputNames(drafter_signature.InputNames(),
                                             drafter_signature.OutputNames(),
                                             /*strict=*/false));

  ABSL_ASSIGN_OR_RETURN(
      auto drafter_mask_runner,
      LiteRtLogitMaskRunner::Create(env, GetHwAcceleratorForBackend(backend)));

  return absl::WrapUnique(new LlmLiteRtMtpDrafter(
      std::move(mtp_drafter_model), std::move(drafter_signature), base_model,
      base_model_desc, embedding_manager, std::move(verifier),
      std::move(drafter_sampler), std::move(kv_cache_input_names),
      std::move(mtp_drafter_input_buffers),
      std::move(mtp_drafter_output_buffers), std::move(drafter_id_tensor),
      num_draft_steps, std::move(drafter_model_signatures),
      GetAttentionMaskParams(executor_metadata),
      std::move(drafter_mask_runner)));
}

absl::Status LlmLiteRtMtpDrafter::UpdateVerifierSamplerConfig(
    const proto::SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<std::default_random_engine> verifier_rand_gen,
      verifier_->UpdateSamplerConfig(sampler_params, std::move(rand_gen)));
  // This is called on every sampler re-sync (e.g. each session switch); the
  // lenience stream restarts only when the verifier's stream does.
  lenience_rng_.Update(sampler_params.seed(), verifier_rand_gen);
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::SetLenience(float lenience) {
  if (!(lenience >= 0.0f && lenience <= 1.0f)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Speculative lenience must be in [0, 1], got ", lenience));
  }
  lenience_ = lenience;
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::PrepareDrafterInputBuffers(
    int position, absl::flat_hash_map<absl::string_view, TensorBuffer>&
                      output_kv_cache_buffers) {
  for (const auto& kv_cache_input_name : kv_cache_input_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto kv_cache_buffer_dup,
        output_kv_cache_buffers.at(kv_cache_input_name).Duplicate());
    active_drafter_input_buffers_[kv_cache_input_name] =
        std::move(kv_cache_buffer_dup);
  }
  LITERT_RETURN_IF_ERROR(
      active_drafter_input_buffers_["input_pos"].Write<int32_t>(
          absl::MakeSpan(&position, 1)));
  if (drafter_signatures_.input_attn_mask.has_value()) {
    auto& mask_buf =
        active_drafter_input_buffers_[*drafter_signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/1,
                                           attn_params_.global_type));
  }
  if (drafter_signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf = active_drafter_input_buffers_[*drafter_signatures_
                                                        .input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/1, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kDecode));
  }
  if (drafter_signatures_.input_int32_param.has_value()) {
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_drafter_input_buffers_[*drafter_signatures_.input_int32_param],
        position,
        /*update_length=*/1));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::PrepareDrafterOutputBuffers() {
  for (auto& [output_name, output_buffer] : active_drafter_output_buffers_) {
    LITERT_RETURN_IF_ERROR(output_buffer.ClearEvent());
  }
  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtMtpDrafter::DraftingResult>
LlmLiteRtMtpDrafter::RunDraftingLoop(
    int token_id, std::optional<TensorBuffer>& activations,
    const Constraint* constraint,
    const Constraint::State* verified_constraint_state) {
  DraftingResult result;
  result.drafted_tokens.reserve(num_draft_steps_);
  result.draft_constraint_states.reserve(num_draft_steps_);
  if (constraint != nullptr) {
    // One extra slot for the final position's mask, added in RunVerification.
    result.draft_masks.reserve(num_draft_steps_ + 1);
  }
  int last_drafted_token_id = token_id;
  std::vector<float> embedding_vector;
  TensorBuffer* activations_ptr =
      activations.has_value() ? &activations.value() : nullptr;
  const Constraint::State* current_draft_state = verified_constraint_state;
  for (int i = 0; i < num_draft_steps_; ++i) {
    LITERT_RETURN_IF_ERROR(PrepareDrafterOutputBuffers());
    // Concat and lookup embeddings with previous activations.
    // Embedding lookup has shape [B = 1, T = 1, D = 1536]
    // Drafter output activation has shape [B = 1, T = 1, D = 1536]
    // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
    TensorBuffer* drafter_activations_buffer =
        &active_drafter_input_buffers_["activations"];
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                         embedding_vector));
    if (activations_ptr) {
      ABSL_RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(
          embedding_vector, *activations_ptr, *drafter_activations_buffer));
    } else {
      ABSL_ASSIGN_OR_RETURN(TensorBuffer * verifier_activations,
                            verifier_->GetOutputBuffer("activations"));
      ABSL_RETURN_IF_ERROR(
          ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
              embedding_vector, *verifier_activations,
              last_verified_token_id_idx_, *drafter_activations_buffer));
    }

    bool async = true;
    LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
        drafter_signature_.Key(), active_drafter_input_buffers_,
        active_drafter_output_buffers_, async));

    if (constraint != nullptr) {
      if (current_draft_state != nullptr) {
        ABSL_ASSIGN_OR_RETURN(auto mask,
                              constraint->ComputeMask(*current_draft_state));
        ABSL_RETURN_IF_ERROR(drafter_mask_runner_->Apply(
            active_drafter_output_buffers_["logits"], mask.get()));
        result.draft_masks.push_back(std::move(mask));
      } else {
        result.draft_masks.push_back(nullptr);
      }
    }

    ABSL_RETURN_IF_ERROR(drafter_sampler_->SampleToIdAndScoreBuffer(
        active_drafter_output_buffers_["logits"], drafter_id_tensor_,
        /*scores_tensor=*/nullptr));
    LITERT_ASSIGN_OR_RETURN(auto id_vector,
                            CopyFromTensorBuffer<int32_t>(drafter_id_tensor_));
    RET_CHECK_EQ(id_vector.size(), 1);
    int sampled_draft_id = id_vector[0];
    result.drafted_tokens.push_back(sampled_draft_id);

    if (constraint != nullptr && current_draft_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(
          auto next_state,
          AdvanceConstraintState(*constraint, *current_draft_state,
                                 sampled_draft_id));
      result.draft_constraint_states.push_back(std::move(next_state));
      current_draft_state = result.draft_constraint_states.back().get();
    }

    last_drafted_token_id = sampled_draft_id;
    activations_ptr = &active_drafter_output_buffers_["projected_activations"];
  }
  return result;
}

absl::StatusOr<std::vector<int>> LlmLiteRtMtpDrafter::RunVerification(
    int position, int token_id, DraftingResult& drafting_result,
    const LitertState::StateBuffers& state_buffers) {
  std::vector<int> drafted_tokens_with_input_token;
  drafted_tokens_with_input_token.reserve(num_draft_steps_ + 1);
  drafted_tokens_with_input_token.push_back(token_id);
  drafted_tokens_with_input_token.insert(drafted_tokens_with_input_token.end(),
                                         drafting_result.drafted_tokens.begin(),
                                         drafting_result.drafted_tokens.end());
  ABSL_RETURN_IF_ERROR(
      verifier_->Run(position, drafted_tokens_with_input_token, state_buffers));

  // Apply constraint masks to verifier logits for all positions in a single
  // pass. Masks for positions [0, num_draft_steps_ - 1] were already computed
  // during RunDraftingLoop; only the final position's mask needs to be
  // computed here.
  std::vector<std::unique_ptr<LogitMask>> masks;
  if (constraint_ != nullptr) {
    const auto& draft_constraint_states =
        drafting_result.draft_constraint_states;
    masks = std::move(drafting_result.draft_masks);
    const Constraint::State* final_state =
        draft_constraint_states.empty() ? constraint_state_.get()
                                        : draft_constraint_states.back().get();
    if (final_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask, constraint_->ComputeMask(*final_state));
      masks.push_back(std::move(mask));
    } else {
      masks.push_back(nullptr);
    }
  }
  return verifier_->MaskAndSample(masks);
}

absl::StatusOr<std::vector<std::vector<int>>> LlmLiteRtMtpDrafter::Draft(
    int position, int token_id, std::optional<TensorBuffer> activations,
    StateInterface& state, const Constraint* constraint) {
  auto* litert_state = dynamic_cast<LitertState*>(&state);
  RET_CHECK(litert_state != nullptr);
  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(base_model_, verifier_->signature_key()));

  ABSL_RETURN_IF_ERROR(
      PrepareDrafterInputBuffers(position - 1, state_buffers.output_buffers));

  // Initialize or reset constraint state when constraint changes or on first
  // step of decode turn.
  if (constraint == nullptr) {
    constraint_ = nullptr;
    constraint_state_.reset();
  } else if (constraint != constraint_ || activations.has_value() ||
             constraint_state_ == nullptr) {
    constraint_ = constraint;
    ABSL_ASSIGN_OR_RETURN(
        constraint_state_,
        AdvanceConstraintState(*constraint_, *constraint_->Start(), token_id));
  }

  const absl::Time draft_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(DraftingResult drafting_result,
                        RunDraftingLoop(token_id, activations, constraint_,
                                        constraint_state_.get()));
  const absl::Duration drafting_time = absl::Now() - draft_start;

  const absl::Time verify_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> verifier_id_vector,
      RunVerification(position, token_id, drafting_result, state_buffers));
  const absl::Duration verify_time = absl::Now() - verify_start;

  RET_CHECK_EQ(verifier_id_vector.size(),
               drafting_result.drafted_tokens.size() + 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const LenientAcceptResult accept_result = CountAcceptedPrefixWithLenience(
      verifier_id_vector, drafting_result.drafted_tokens, lenience_,
      [&](int /*i*/) -> double { return unit(lenience_rng_.engine()); });
  int num_correct_tokens = accept_result.accepted;
  int bonus_token = verifier_id_vector[num_correct_tokens];
  last_verified_token_id_idx_ = num_correct_tokens;

  // Update verified constraint state according to verified + bonus tokens.
  if (constraint_ != nullptr) {
    if (num_correct_tokens > 0 &&
        num_correct_tokens <=
            static_cast<int>(drafting_result.draft_constraint_states.size())) {
      constraint_state_ = std::move(
          drafting_result.draft_constraint_states[num_correct_tokens - 1]);
    }
    if (bonus_token >= 0 && constraint_state_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(constraint_state_,
                            AdvanceConstraintState(
                                *constraint_, *constraint_state_, bonus_token));
    }
  }

  MTP_DRAFTER_LOG() << "drafted_tokens: "
                    << absl::StrJoin(drafting_result.drafted_tokens, ", ");
  MTP_DRAFTER_LOG() << "bonus_token: " << bonus_token;
  MTP_DRAFTER_LOG() << "num_correct_tokens: " << num_correct_tokens;

  // The first token comes from the decode output and is always correct.
  std::vector<int> output_tokens = std::move(drafting_result.drafted_tokens);
  output_tokens.resize(num_correct_tokens);
  output_tokens.push_back(bonus_token);
  num_drafted_tokens_ += num_draft_steps_;
  num_verified_tokens_ += num_correct_tokens;
  last_round_stats_ = MtpRoundStats{
      .drafted_tokens = num_draft_steps_,
      .accepted_tokens = num_correct_tokens,
      .emitted_tokens = num_correct_tokens + 1,
      .lenient_accepted_tokens = accept_result.lenient_accepted,
      .drafting_time = drafting_time,
      .verify_time = verify_time,
  };

  MTP_DRAFTER_LOG() << "drafter output: " << absl::StrJoin(output_tokens, ", ");
  MTP_DRAFTER_LOG() << "--------------------------------------------------";

  return std::vector<std::vector<int>>{std::move(output_tokens)};
}

}  // namespace litert::lm

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
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
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
#include "runtime/executor/state_interface.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "tflite/types/half.h"  // from @litert

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
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return CreateSampler(backend, output_heads, std::move(sampler_params), env,
                       sequence_size, vocab_size, activation_data_type);
}

absl::StatusOr<TensorBuffer> CreateFP16OutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    absl::string_view output_name, size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  // Use runtime layout.
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(
      auto requirements,
      compiled_model.GetOutputBufferRequirements(signature_index, output_name));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType new_tensor_type(litert::ElementType::Float16,
                                   std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer_types, requirements.SupportedTypes());
  if (buffer_types.empty()) {
    return absl::InternalError("No supported buffer types found.");
  }
  auto buffer_type = buffer_types[0];
  LITERT_ASSIGN_OR_RETURN(
      auto buffer, TensorBuffer::CreateManaged(
                       env, buffer_type, std::move(new_tensor_type), size));
  return buffer;
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

absl::StatusOr<int> GetVocabSizeFromLogitsTensor(TensorBuffer& logits_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  // logits tensor shape is [batch, seq, vocab].
  return logits_tensor_type.Layout().Dimensions()[2];
}

template <typename T>
absl::Status ApplyMaskToLogitsTyped(TensorBuffer& logits_buffer,
                                    TensorBufferType buffer_type,
                                    const LogitMask& mask, int vocab_size) {
  if (buffer_type == TensorBufferType::kHostMemory) {
    LITERT_ASSIGN_OR_RETURN(auto span,
                            ReferTensorBufferAsSpan<T>(logits_buffer));
    RET_CHECK_GE(static_cast<int>(span.size()), vocab_size);
    return mask.Apply(span.subspan(0, vocab_size));
  } else {
    LITERT_ASSIGN_OR_RETURN(auto vec, CopyFromTensorBuffer<T>(logits_buffer));
    RET_CHECK_GE(static_cast<int>(vec.size()), vocab_size);
    ABSL_RETURN_IF_ERROR(
        mask.Apply(absl::MakeSpan(vec).subspan(0, vocab_size)));
    LITERT_RETURN_IF_ERROR(logits_buffer.Write(absl::MakeConstSpan(vec)));
    return absl::OkStatus();
  }
}

absl::Status ApplyMaskToLogits(TensorBuffer& logits_buffer,
                               const LogitMask* mask, int vocab_size) {
  if (mask == nullptr) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_buffer.BufferType());

  if (logits_type.ElementType() == ElementType::Float32) {
    return ApplyMaskToLogitsTyped<float>(logits_buffer, buffer_type, *mask,
                                         vocab_size);
  } else if (logits_type.ElementType() == ElementType::Float16) {
    return ApplyMaskToLogitsTyped<tflite::half>(logits_buffer, buffer_type,
                                                *mask, vocab_size);
  }
  return absl::InvalidArgumentError("Unsupported logits element type.");
}

template <typename T>
absl::Status ApplyMasksToLogitsSequenceTyped(
    TensorBuffer& logits_buffer, TensorBufferType buffer_type,
    absl::Span<const std::unique_ptr<LogitMask>> masks, int vocab_size) {
  if (buffer_type == TensorBufferType::kHostMemory) {
    LITERT_ASSIGN_OR_RETURN(auto span,
                            ReferTensorBufferAsSpan<T>(logits_buffer));
    RET_CHECK_GE(static_cast<int>(span.size()),
                 static_cast<int>(masks.size()) * vocab_size);
    for (size_t i = 0; i < masks.size(); ++i) {
      if (masks[i] != nullptr) {
        ABSL_RETURN_IF_ERROR(
            masks[i]->Apply(span.subspan(i * vocab_size, vocab_size)));
      }
    }
    return absl::OkStatus();
  } else {
    LITERT_ASSIGN_OR_RETURN(auto vec, CopyFromTensorBuffer<T>(logits_buffer));
    RET_CHECK_GE(static_cast<int>(vec.size()),
                 static_cast<int>(masks.size()) * vocab_size);
    for (size_t i = 0; i < masks.size(); ++i) {
      if (masks[i] != nullptr) {
        ABSL_RETURN_IF_ERROR(masks[i]->Apply(
            absl::MakeSpan(vec).subspan(i * vocab_size, vocab_size)));
      }
    }
    LITERT_RETURN_IF_ERROR(logits_buffer.Write(absl::MakeConstSpan(vec)));
    return absl::OkStatus();
  }
}

absl::Status ApplyMasksToLogitsSequence(
    TensorBuffer& logits_buffer,
    absl::Span<const std::unique_ptr<LogitMask>> masks, int vocab_size) {
  if (masks.empty()) {
    return absl::OkStatus();
  }
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, logits_buffer.BufferType());

  if (logits_type.ElementType() == ElementType::Float32) {
    return ApplyMasksToLogitsSequenceTyped<float>(logits_buffer, buffer_type,
                                                  masks, vocab_size);
  } else if (logits_type.ElementType() == ElementType::Float16) {
    return ApplyMasksToLogitsSequenceTyped<tflite::half>(
        logits_buffer, buffer_type, masks, vocab_size);
  }
  return absl::InvalidArgumentError("Unsupported logits element type.");
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
    case Backend::CPU: {
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
      resources, ModelType::kTfLiteMtpDrafter, compilation_options));
  ABSL_ASSIGN_OR_RETURN(auto model,
                        resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, model->Get(), compilation_options));
  ABSL_ASSIGN_OR_RETURN(
      auto base_model_desc,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

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

  LITERT_ASSIGN_OR_RETURN(SimpleSignature verify_signature,
                          base_model.FindSignature(kVerifySignatureRunner));

  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_output_buffers;
  int num_draft_steps;
  {
    for (absl::string_view input_name : verify_signature.InputNames()) {
      if (absl::StrContains(input_name, "kv")) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          base_model.CreateInputBuffer(verify_signature.Key(), input_name));
      verifier_input_buffers[input_name] = std::move(input_buffer);
    }
    LITERT_ASSIGN_OR_RETURN(
        size_t verify_signature_index,
        base_model.GetSignatureIndex(kVerifySignatureRunner));
    for (size_t i = 0; i < verify_signature.OutputNames().size(); ++i) {
      absl::string_view output_name = verify_signature.OutputNames()[i];
      if (absl::StrContains(output_name, "kv")) {
        continue;
      }
      if (output_name == "logits" && use_fp16_precision) {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            CreateFP16OutputBuffer(env, base_model, verify_signature_index,
                                   output_name, i));
        verifier_output_buffers[output_name] = std::move(output_buffer);
      } else {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            base_model.CreateOutputBuffer(verify_signature.Key(), output_name));
        verifier_output_buffers[output_name] = std::move(output_buffer);
      }
    }

    LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                            verify_signature.InputTensorType("input_pos"));
    // Expecred shape: [T = G + 1] where G is the number of draft steps
    const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
    num_draft_steps = input_pos_dims[0] - 1;
  }

  LITERT_ASSIGN_OR_RETURN(
      int vocab_size,
      GetVocabSizeFromLogitsTensor(verifier_output_buffers["logits"]));

  ActivationDataType drafter_logits_data_type;
  {
    LITERT_ASSIGN_OR_RETURN(auto drafter_logits_type,
                            mtp_drafter_output_buffers["logits"].TensorType());
    if (drafter_logits_type.ElementType() == ElementType::Float16) {
      drafter_logits_data_type = ActivationDataType::FLOAT16;
    } else if (drafter_logits_type.ElementType() == ElementType::Float32) {
      drafter_logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported drafter logits type");
    }
  }

  ActivationDataType verifier_logits_data_type;
  {
    LITERT_ASSIGN_OR_RETURN(auto verifier_logits_type,
                            verifier_output_buffers["logits"].TensorType());
    if (verifier_logits_type.ElementType() == ElementType::Float16) {
      verifier_logits_data_type = ActivationDataType::FLOAT16;
    } else if (verifier_logits_type.ElementType() == ElementType::Float32) {
      verifier_logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported verifier logits type");
    }
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
    LITERT_ASSIGN_OR_RETURN(auto verifier_activations_type,
                            verify_signature.OutputTensorType("activations"));
    if (verifier_activations_type.ElementType() != ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Unsupported verifier activations type: must be float32");
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto drafter_sampler,
                        CreateGreedySampler(env, backend,
                                            /*output_heads=*/1,
                                            /*sequence_size=*/1, vocab_size,
                                            drafter_logits_data_type));
  ABSL_ASSIGN_OR_RETURN(
      auto verifier_sampler,
      CreateGreedySampler(env, backend,
                          /*output_heads=*/1,
                          /*sequence_size=*/num_draft_steps + 1, vocab_size,
                          verifier_logits_data_type));

  LITERT_ASSIGN_OR_RETURN(auto drafter_id_tensor,
                          CreateTensorBuffer<int32_t>({1, 1}));
  LITERT_ASSIGN_OR_RETURN(
      auto verifier_id_tensor,
      CreateTensorBuffer<int32_t>({1, num_draft_steps + 1}));

  ABSL_ASSIGN_OR_RETURN(
      auto drafter_model_signatures,
      GetModelSignaturesFromInputOutputNames(drafter_signature.InputNames(),
                                             drafter_signature.OutputNames(),
                                             /*strict=*/false));
  ABSL_ASSIGN_OR_RETURN(
      auto verifier_model_signatures,
      GetModelSignaturesFromInputOutputNames(verify_signature.InputNames(),
                                             verify_signature.OutputNames(),
                                             /*strict=*/false));

  return absl::WrapUnique(new LlmLiteRtMtpDrafter(
      std::move(mtp_drafter_model), std::move(drafter_signature), base_model,
      std::move(verify_signature), base_model_desc, embedding_manager,
      ple_manager, std::move(drafter_sampler), std::move(verifier_sampler),
      std::move(kv_cache_input_names), std::move(mtp_drafter_input_buffers),
      std::move(mtp_drafter_output_buffers), std::move(verifier_input_buffers),
      std::move(verifier_output_buffers), std::move(drafter_id_tensor),
      std::move(verifier_id_tensor), num_draft_steps,
      std::move(drafter_model_signatures), std::move(verifier_model_signatures),
      vocab_size, GetAttentionMaskParams(executor_metadata)));
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
      ABSL_RETURN_IF_ERROR(
          ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
              embedding_vector, verifier_output_buffers_["activations"],
              last_verified_token_id_idx_, *drafter_activations_buffer));
    }

    bool async = true;
    LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
        drafter_signature_.Key(), active_drafter_input_buffers_,
        active_drafter_output_buffers_, async));

    if (constraint != nullptr && current_draft_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask,
                            constraint->ComputeMask(*current_draft_state));
      ABSL_RETURN_IF_ERROR(ApplyMaskToLogits(
          active_drafter_output_buffers_["logits"], mask.get(), vocab_size_));
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
          constraint->ComputeNext(*current_draft_state, sampled_draft_id));
      if (constraint->IsEnded(*next_state)) {
        next_state = constraint->Start();
      }
      result.draft_constraint_states.push_back(std::move(next_state));
      current_draft_state = result.draft_constraint_states.back().get();
    }

    last_drafted_token_id = sampled_draft_id;
    activations_ptr = &active_drafter_output_buffers_["projected_activations"];
  }
  return result;
}

absl::Status LlmLiteRtMtpDrafter::PrepareVerifierInputBuffers(
    int position, int token_id, const std::vector<int>& drafted_tokens,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        input_kv_cache_buffers) {
  {
    LITERT_ASSIGN_OR_RETURN(auto verifier_input_pos_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                active_verifier_input_buffers_["input_pos"],
                                TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(verifier_input_pos_lock_and_addr.second);
    for (int i = 0; i < num_draft_steps_ + 1; ++i) {
      *prefill_input_pos_ptr++ = position + i;
    }
  }

  std::vector<int> drafted_tokens_with_input_token;
  drafted_tokens_with_input_token.reserve(num_draft_steps_ + 1);
  drafted_tokens_with_input_token.push_back(token_id);
  drafted_tokens_with_input_token.insert(drafted_tokens_with_input_token.end(),
                                         drafted_tokens.begin(),
                                         drafted_tokens.end());

  if (verifier_signatures_.input_attn_mask.has_value()) {
    auto& mask_buf =
        active_verifier_input_buffers_[*verifier_signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/num_draft_steps_ + 1,
                                           attn_params_.global_type));
  }
  if (verifier_signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf =
        active_verifier_input_buffers_[*verifier_signatures_
                                            .input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/num_draft_steps_ + 1, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kVerify));
  }

  if (verifier_signatures_.input_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupPrefill(
        drafted_tokens_with_input_token,
        &active_verifier_input_buffers_[*verifier_signatures_.input_embeddings],
        /*offset=*/0));
  }
  if (ple_manager_.has_value() &&
      verifier_signatures_.input_per_layer_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(ple_manager_->get().LookupPrefill(
        drafted_tokens_with_input_token,
        &active_verifier_input_buffers_[*verifier_signatures_
                                             .input_per_layer_embeddings],
        /*offset=*/0));
  }

  for (const auto& [input_name, input_buffer] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    active_verifier_input_buffers_[input_name] = std::move(input_buffer_dup);
  }
  if (verifier_signatures_.input_int32_param.has_value()) {
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_verifier_input_buffers_[*verifier_signatures_.input_int32_param],
        position, num_draft_steps_ + 1));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::PrepareVerifierOutputBuffers(
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        output_kv_cache_buffers) {
  for (const auto& [output_name, output_buffer] : output_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    active_verifier_output_buffers_[output_name] = std::move(output_buffer_dup);
  }
  for (auto& [output_name, output_buffer] : active_verifier_output_buffers_) {
    LITERT_RETURN_IF_ERROR(output_buffer.ClearEvent());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmLiteRtMtpDrafter::RunVerification(
    const std::vector<std::unique_ptr<Constraint::State>>&
        draft_constraint_states) {
  bool async = true;
  LITERT_RETURN_IF_ERROR(base_model_.RunAsync(
      verify_signature_.Key(), active_verifier_input_buffers_,
      active_verifier_output_buffers_, async));

  // Apply constraint masks to verifier logits for all positions in a single
  // pass.
  if (constraint_ != nullptr) {
    std::vector<std::unique_ptr<LogitMask>> masks;
    masks.reserve(draft_constraint_states.size() + 1);
    if (constraint_state_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask,
                            constraint_->ComputeMask(*constraint_state_));
      masks.push_back(std::move(mask));
    } else {
      masks.push_back(nullptr);
    }
    for (const auto& draft_state : draft_constraint_states) {
      if (draft_state != nullptr) {
        ABSL_ASSIGN_OR_RETURN(auto mask,
                              constraint_->ComputeMask(*draft_state));
        masks.push_back(std::move(mask));
      } else {
        masks.push_back(nullptr);
      }
    }
    ABSL_RETURN_IF_ERROR(ApplyMasksToLogitsSequence(
        active_verifier_output_buffers_.at("logits"), masks, vocab_size_));
  }

  ABSL_RETURN_IF_ERROR(verifier_sampler_->SampleToIdAndScoreBuffer(
      active_verifier_output_buffers_.at("logits"), verifier_id_tensor_,
      /*scores_tensor=*/nullptr));

  LITERT_ASSIGN_OR_RETURN(auto id_vector,
                          CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
  RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1);
  return id_vector;
}

absl::StatusOr<std::vector<std::vector<int>>> LlmLiteRtMtpDrafter::Draft(
    int position, int token_id, std::optional<TensorBuffer> activations,
    StateInterface& state, const Constraint* constraint) {
  auto* litert_state = dynamic_cast<LitertState*>(&state);
  RET_CHECK(litert_state != nullptr);
  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(base_model_, verify_signature_.Key()));

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
    constraint_state_ = constraint_->Start();
    // Compute mask on the initial state so the underlying grammar engine (e.g.
    // llguidance) initializes parser rows and lexer state before committing the
    // first token.
    ABSL_RETURN_IF_ERROR(constraint_->ComputeMask(*constraint_state_).status());
    ABSL_ASSIGN_OR_RETURN(constraint_state_, constraint_->ComputeNext(
                                                 *constraint_state_, token_id));
    if (constraint_->IsEnded(*constraint_state_)) {
      constraint_state_ = constraint_->Start();
    }
  }

  ABSL_ASSIGN_OR_RETURN(DraftingResult drafting_result,
                        RunDraftingLoop(token_id, activations, constraint_,
                                        constraint_state_.get()));

  ABSL_RETURN_IF_ERROR(PrepareVerifierInputBuffers(
      position, token_id, drafting_result.drafted_tokens,
      state_buffers.input_buffers));
  ABSL_RETURN_IF_ERROR(
      PrepareVerifierOutputBuffers(state_buffers.output_buffers));

  ABSL_ASSIGN_OR_RETURN(
      std::vector<int> verifier_id_vector,
      RunVerification(drafting_result.draft_constraint_states));

  int num_correct_tokens = 0;
  while (num_correct_tokens < num_draft_steps_ &&
         verifier_id_vector[num_correct_tokens] ==
             drafting_result.drafted_tokens[num_correct_tokens]) {
    ++num_correct_tokens;
  }
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
      ABSL_ASSIGN_OR_RETURN(
          constraint_state_,
          constraint_->ComputeNext(*constraint_state_, bonus_token));
      if (constraint_->IsEnded(*constraint_state_)) {
        constraint_state_ = constraint_->Start();
      }
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

  MTP_DRAFTER_LOG() << "drafter output: " << absl::StrJoin(output_tokens, ", ");
  MTP_DRAFTER_LOG() << "--------------------------------------------------";

  return std::vector<std::vector<int>>{std::move(output_tokens)};
}

}  // namespace litert::lm

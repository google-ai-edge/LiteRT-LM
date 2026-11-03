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

#include "runtime/executor/embedding_litert_compiled_model_executor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/executor/executor_stats.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "absl/strings/match.h"  // from @com_google_absl
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kEmbeddingModuleName = "Embedding";

constexpr absl::string_view kEmbedderLookupLatency = "Embedder lookup";
constexpr absl::string_view kMaskPreparationLatency = "Mask preparation";
constexpr absl::string_view kPerLayerEmbedderLookupLatency =
    "Per-layer embedder lookup";
constexpr absl::string_view kTextEncoderInferenceLatency =
    "Text encoder inference";
constexpr absl::string_view kOutputCopyLatency = "Output copy";

constexpr absl::string_view kEmbeddingNumTokensMetric =
    "(e2e) Embedding num tokens";
constexpr absl::string_view kEmbeddingNumPaddedTokensMetric =
    "(e2e) Embedding num padded tokens";
constexpr absl::string_view kEmbeddingE2eTokensPerSecondMetric =
    "(e2e) Embedding tokens per second";
constexpr absl::string_view kEmbeddingE2ePaddedTokensPerSecondMetric =
    "(e2e) Embedding tokens per second (padded)";
constexpr absl::string_view kEmbeddingEncoderStackTokensPerSecondMetric =
    "(EncoderStackOnly) Embedding tokens per second";
constexpr absl::string_view kEmbeddingEncoderStackPaddedTokensPerSecondMetric =
    "(EncoderStackOnly) Embedding tokens per second (padded)";
constexpr absl::string_view kTextEncoderMetricsPrefix = "Text encoder ";

constexpr absl::string_view kEncoderSignatureRunner = "encoder";
constexpr absl::string_view kPerLayerEmbeddingsName = "per_layer";
constexpr absl::string_view kInputMaskName = "input_mask";
constexpr absl::string_view kEmbeddingsName = "embeddings";

class CompiledModelWrapper : public litert::CompiledModel {
 public:
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        const LiteRtModel litert_model,
                                        Options& compilation_options) {
    return litert::CompiledModel::Create(env, litert_model,
                                         compilation_options);
  }
};

absl::StatusOr<std::map<int, size_t>> GetTextEncoderSignatureMap(
    const litert::Model& model,
    const std::vector<std::string>& selected_signatures = {}) {
  std::map<int, size_t> encoder_signatures;

  for (int i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    absl::string_view key = signature.Key();
    if (absl::StartsWith(key, kEncoderSignatureRunner)) {
      if (!selected_signatures.empty() &&
          !absl::c_linear_search(selected_signatures, key)) {
        continue;
      }
      if (signature.InputNames().empty()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Text encoder signature has no input tensors: ", key));
      }

      // Find the embeddings input tensor type to get sequence length.
      std::optional<RankedTensorType> embeddings_tensor_type;
      for (auto name : signature.InputNames()) {
        if (absl::StrContains(name, kEmbeddingsName)) {
          LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                                  signature.InputTensorType(name));
          break;
        }
      }

      // Fallback to the first input tensor if kEmbeddingsName was not found.
      if (!embeddings_tensor_type.has_value()) {
        LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                                signature.InputTensorType(0));
      }

      int sequence_length = 0;
      if (embeddings_tensor_type->Layout().Rank() == 3) {
        // [batch_size, seq_len, embed_dim]
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[1];
      } else if (embeddings_tensor_type->Layout().Rank() == 2) {
        // [seq_len, embed_dim]
        sequence_length = embeddings_tensor_type->Layout().Dimensions()[0];
      } else {
        return absl::FailedPreconditionError(
            absl::StrCat("Embedding input tensor has unsupported dimension: ",
                         embeddings_tensor_type->Layout().Rank()));
      }

      encoder_signatures[sequence_length] = i;
    }
  }

  if (encoder_signatures.empty()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "No signature found with prefix ", kEncoderSignatureRunner));
  }

  return encoder_signatures;
}

}  // namespace

// static
absl::StatusOr<CompiledTextEncoderInfo>
EmbeddingLiteRtCompiledModelExecutor::CompileTextEncoder(
    const EmbeddingExecutorSettings& executor_settings, Environment& env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder_model,
      resources.GetTFLiteModel(proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER));
  RET_CHECK_NE(text_encoder_model, nullptr) << "TEXT_ENCODER model is null.";

  litert::Options options;
  switch (executor_settings.GetBackend()) {
    case Backend::CPU: {
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                              options.GetOptions<::litert::CpuOptions>());
      LITERT_RETURN_IF_ERROR(
          SetCpuOptions(cpu_options, executor_settings.GetNumThreads()));
      auto weight_cache_file = executor_settings.GetWeightCacheFile(
          absl::StrCat(EmbeddingExecutorSettings::kEncoderName,
                       ExecutorSettingsBase::kXnnpackCacheSuffix),
          /*check_and_clean=*/true);
      ABSL_RETURN_IF_ERROR(SetCpuCacheOptions(
          weight_cache_file,
          /*logging_prefix=*/EmbeddingExecutorSettings::kEncoderName,
          cpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                              options.GetOptions<::litert::GpuOptions>());
      ABSL_ASSIGN_OR_RETURN(
          const auto cache_files,
          GetGpuModelCacheData(executor_settings,
                               EmbeddingExecutorSettings::kEncoderName));
      LITERT_RETURN_IF_ERROR(
          SetCommonGpuOptions(executor_settings, gpu_options));
      // The text encoder is compiled for the GPU only, so it is always fully
      // delegated to a single delegate. This lets TFLite skip preparing the
      // CPU kernels before delegation, which otherwise allocates a CPU arena
      // for every STABLEHLO_COMPOSITE decomposition subgraph that is never
      // released.
      gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
      ABSL_RETURN_IF_ERROR(SetGpuCacheOptions(
          cache_files.weight_cache_file, cache_files.program_cache_file,
          cache_files.cache_key,
          /*logging_prefix=*/EmbeddingExecutorSettings::kEncoderName,
          /*cache_compiled_shaders_only=*/false, gpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              options.GetOptions<qualcomm::QualcommOptions>());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpFileReadMemoryBudgetMB(16);
      ABSL_RETURN_IF_ERROR(
          SetNpuOptions(executor_settings.GetNpuOptions(), options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                      litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ",
                       GetBackendString(executor_settings.GetBackend())));
  }
  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER, options));

  if (!executor_settings.GetSelectedSignatures().empty()) {
    std::vector<absl::string_view> selected_signatures;
    selected_signatures.reserve(
        executor_settings.GetSelectedSignatures().size());
    for (const auto& sig : executor_settings.GetSelectedSignatures()) {
      selected_signatures.push_back(sig);
    }
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options, options.GetRuntimeOptions());
    LITERT_RETURN_IF_ERROR(
        runtime_options.SetSelectedSignatures(selected_signatures));
  }

#ifdef __EMSCRIPTEN__
  extern void SetCurrentlyCompilingModel(absl::string_view model_type_str)
      __attribute__((weak));
  if (SetCurrentlyCompilingModel) {
    SetCurrentlyCompilingModel(TfLiteModelTypeToWireString(
        proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER));
  }
#endif
  auto compiled_model_status_or =
      CompiledModelWrapper::Create(env, text_encoder_model->Get(), options);
#ifdef __EMSCRIPTEN__
  if (SetCurrentlyCompilingModel) {
    SetCurrentlyCompilingModel("");
  }
#endif
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          std::move(compiled_model_status_or));
  auto compiled_model_ptr =
      std::make_unique<litert::CompiledModel>(std::move(compiled_model));

  ABSL_ASSIGN_OR_RETURN(
      auto encoder_signatures,
      GetTextEncoderSignatureMap(*text_encoder_model,
                                 executor_settings.GetSelectedSignatures()));
  size_t default_signature_index = encoder_signatures.begin()->second;

  LITERT_ASSIGN_OR_RETURN(
      auto input_tensor_type,
      text_encoder_model->GetInputTensorType(default_signature_index, 0));
  const auto& input_dims = input_tensor_type.Layout().Dimensions();
  std::vector<int> expected_input_dimension(input_dims.begin(),
                                            input_dims.end());

  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_type,
      text_encoder_model->GetOutputTensorType(default_signature_index, 0));
  const auto& output_dims = output_tensor_type.Layout().Dimensions();
  RET_CHECK(!output_dims.empty()) << "Output dimensions cannot be empty.";
  int embedding_dimension = output_dims.back();

  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      input_buffers_cache;
  absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
      output_buffers_cache;
  for (const auto& [seq_len, signature_index] : encoder_signatures) {
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffers,
        compiled_model_ptr->CreateInputBuffers(signature_index));
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffers,
        compiled_model_ptr->CreateOutputBuffers(signature_index));

    input_buffers_cache[signature_index] = std::move(input_buffers);
    output_buffers_cache[signature_index] = std::move(output_buffers);
  }

  CompiledTextEncoderInfo info;
  info.compiled_model = std::move(compiled_model_ptr);
  info.expected_input_dimension = std::move(expected_input_dimension);
  info.embedding_dimension = embedding_dimension;
  info.encoder_signatures = std::move(encoder_signatures);
  info.input_buffers_cache = std::move(input_buffers_cache);
  info.output_buffers_cache = std::move(output_buffers_cache);
  return info;
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources) {
  RET_CHECK_NE(resources, nullptr) << "ModelResources is null.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      env, *resources, embedding_lookup, per_layer_embedding_lookup));
  if (embedding_lookup == nullptr) {
    ABSL_LOG(INFO) << "kTfLiteEmbedder model not found in resources. The "
                      "executor only supports pre-computed input embeddings.";
  }

  ABSL_ASSIGN_OR_RETURN(auto compiled_text_encoder_info,
                        CompileTextEncoder(executor_settings, env, *resources));

  return Create(std::move(executor_settings), env, std::move(resources),
                std::move(embedding_lookup),
                std::move(per_layer_embedding_lookup),
                std::move(compiled_text_encoder_info));
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
    CompiledTextEncoderInfo compiled_text_encoder_info) {
  RET_CHECK_NE(compiled_text_encoder_info.compiled_model, nullptr)
      << "The compiled text encoder is null.";
  return absl::WrapUnique(new EmbeddingLiteRtCompiledModelExecutor(
      std::move(executor_settings), env, std::move(resources),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      std::move(compiled_text_encoder_info.compiled_model),
      std::move(compiled_text_encoder_info.expected_input_dimension),
      compiled_text_encoder_info.embedding_dimension,
      std::move(compiled_text_encoder_info.encoder_signatures),
      std::move(compiled_text_encoder_info.input_buffers_cache),
      std::move(compiled_text_encoder_info.output_buffers_cache)));
}

// static
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(
      auto resources,
      BuildLiteRtCompiledModelResources(executor_settings.GetModelAssets()));
  return Create(std::move(executor_settings), env, std::move(resources));
}

absl::StatusOr<EmbeddingLiteRtCompiledModelExecutor::EncoderRun>
EmbeddingLiteRtCompiledModelExecutor::PrepareEncoderRun(
    size_t sequence_length) {
  // Find the smallest signature larger than or equal to the input size.
  auto it = encoder_signatures_.lower_bound(sequence_length);
  if (it == encoder_signatures_.end()) {
    return absl::InternalError(absl::StrCat(
        "No suitable signature found for sequence length ", sequence_length));
  }
  EncoderRun run;
  run.signature_index = it->second;
  run.padded_sequence_length = it->first;

  // Lazily create input buffers.
  if (!input_buffers_cache_.contains(run.signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffers,
        compiled_model_->CreateInputBuffers(run.signature_index));
    input_buffers_cache_[run.signature_index] = std::move(input_buffers);
  }

  // Lazily create output buffers.
  if (!output_buffers_cache_.contains(run.signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffers,
        compiled_model_->CreateOutputBuffers(run.signature_index));
    output_buffers_cache_[run.signature_index] = std::move(output_buffers);
  }
  run.input_buffers = &input_buffers_cache_[run.signature_index];
  run.output_buffers = &output_buffers_cache_[run.signature_index];

  LITERT_ASSIGN_OR_RETURN(
      auto input_names,
      compiled_model_->GetSignatureInputNames(run.signature_index));
  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (absl::StrContains(name, kPerLayerEmbeddingsName)) {
      run.per_layer_embeddings_buffer_index = i;
    } else if (absl::StrContains(name, kInputMaskName)) {
      run.input_mask_buffer_index = i;
    } else if (absl::StrContains(name, kEmbeddingsName)) {
      run.embeddings_buffer_index = i;
    }
  }
  return run;
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::FinishEncoderRun(const EncoderRun& run,
                                                       size_t sequence_length) {
  std::vector<litert::TensorBuffer>& input_buffers = *run.input_buffers;
  std::vector<litert::TensorBuffer>& output_buffers = *run.output_buffers;

  // Compute mask for text encoder. The mask masks out tokens outside of the
  // input sequence length, which can be smaller than the static size of the
  // TFLite graph.
  if (run.input_mask_buffer_index.has_value()) {
    ScopedLatency scoped(latency_stats_, kMaskPreparationLatency);
    auto& mask_buffer = input_buffers[*run.input_mask_buffer_index];
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            mask_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* mask_ptr = static_cast<float*>(mask_lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(auto mask_size_bytes, mask_buffer.Size());
    size_t mask_elements = mask_size_bytes / sizeof(float);

    const size_t active_elements = std::min(sequence_length, mask_elements);
    std::fill_n(mask_ptr, active_elements, 1.0f);
    if (active_elements < mask_elements) {
      std::fill(mask_ptr + active_elements, mask_ptr + mask_elements, 0.0f);
    }
  }

  if (IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  {
    ScopedLatency scoped(latency_stats_, kTextEncoderInferenceLatency);
    LITERT_RETURN_IF_ERROR(compiled_model_->Run(run.signature_index,
                                                input_buffers, output_buffers));
  }

  std::vector<float> output_vector;
  {
    ScopedLatency scoped(latency_stats_, kOutputCopyLatency);
    LITERT_ASSIGN_OR_RETURN(output_vector,
                            CopyFromTensorBuffer<float>(output_buffers[0]));
  }

  AccumulateStat(latency_stats_, kEmbeddingNumTokensMetric,
                 static_cast<int64_t>(sequence_length));
  AccumulateStat(latency_stats_, kEmbeddingNumPaddedTokensMetric,
                 static_cast<int64_t>(run.padded_sequence_length));

  return output_vector;
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::RunEncoderForTokens(
    absl::Span<const int32_t> tokens) {
  ScopedLatency scoped_total_latency(latency_stats_);
  RET_CHECK_NE(embedding_lookup_, nullptr);

  ABSL_ASSIGN_OR_RETURN(EncoderRun run, PrepareEncoderRun(tokens.size()));
  std::vector<litert::TensorBuffer>& input_buffers = *run.input_buffers;

  {
    ScopedLatency scoped(latency_stats_, kEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[run.embeddings_buffer_index],
        /*token_offset=*/0));
  }

  if (run.per_layer_embeddings_buffer_index.has_value() &&
      per_layer_embedding_lookup_ != nullptr) {
    ScopedLatency scoped(latency_stats_, kPerLayerEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[*run.per_layer_embeddings_buffer_index],
        /*token_offset=*/0));
  }

  return FinishEncoderRun(run, tokens.size());
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings) {
  ScopedLatency scoped_total_latency(latency_stats_);

  if (expected_input_dimension_.empty() ||
      expected_input_dimension_.back() <= 0) {
    return absl::FailedPreconditionError(
        "The embedding dimension of the text encoder input is unknown.");
  }
  const size_t embed_dim = expected_input_dimension_.back();
  if (input_embeddings.empty() || input_embeddings.size() % embed_dim != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The size of the input embeddings (", input_embeddings.size(),
        ") must be a positive multiple of the embedding dimension (", embed_dim,
        ")."));
  }
  const size_t sequence_length = input_embeddings.size() / embed_dim;
  if (encoder_signatures_.empty()) {
    return absl::FailedPreconditionError("No encoder signatures available.");
  }
  const size_t max_seq_len = std::prev(encoder_signatures_.end())->first;
  if (sequence_length > max_seq_len) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input sequence length (", sequence_length,
        ") exceeds maximum supported signature length (", max_seq_len, ")."));
  }

  ABSL_ASSIGN_OR_RETURN(EncoderRun run, PrepareEncoderRun(sequence_length));
  if (run.per_layer_embeddings_buffer_index.has_value()) {
    return absl::UnimplementedError(
        "The text encoder requires per-layer embeddings, which are computed "
        "from token IDs. Use ComputeEmbedding instead.");
  }

  {
    ScopedLatency scoped(latency_stats_, kEmbedderLookupLatency);
    litert::TensorBuffer& embeddings_buffer =
        (*run.input_buffers)[run.embeddings_buffer_index];
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, embeddings_buffer.TensorType());
    if (tensor_type.ElementType() != litert::ElementType::Float32) {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported text encoder input element type: ",
                       static_cast<int>(tensor_type.ElementType()),
                       ". Only float32 input embeddings are supported."));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            embeddings_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* embeddings_ptr = static_cast<float*>(lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size_bytes, embeddings_buffer.Size());
    const size_t buffer_elements = buffer_size_bytes / sizeof(float);
    RET_CHECK_LE(input_embeddings.size(), buffer_elements)
        << "The text encoder input buffer is too small.";
    absl::c_copy(input_embeddings, embeddings_ptr);
    std::fill(embeddings_ptr + input_embeddings.size(),
              embeddings_ptr + buffer_elements, 0.0f);
  }

  return FinishEncoderRun(run, sequence_length);
}

bool EmbeddingLiteRtCompiledModelExecutor::IsCancelled() const {
  if (!cancelled_ && should_cancel_ != nullptr && (*should_cancel_)()) {
    cancelled_ = true;
  }
  return cancelled_;
}

// static
bool EmbeddingLiteRtCompiledModelExecutor::CheckCancelled(void* data) {
  return static_cast<const EmbeddingLiteRtCompiledModelExecutor*>(data)
      ->IsCancelled();
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings,
    absl::FunctionRef<bool()> should_cancel) {
  RET_CHECK_NE(compiled_model_, nullptr) << "The text encoder is not compiled.";
  should_cancel_ = &should_cancel;
  cancelled_ = false;
  // Register the per-op cancellation check for the duration of this call only,
  // so that calls without a cancellation callback do not pay for it.
  compiled_model_->SetCancellationFunction(this, &CheckCancelled);
  absl::Cleanup end_cancellable_call = [this] {
    compiled_model_->SetCancellationFunction(/*data=*/nullptr,
                                             /*check_cancelled_func=*/nullptr);
    should_cancel_ = nullptr;
    cancelled_ = false;
  };

  if (IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  absl::StatusOr<std::vector<float>> output =
      ComputeEmbeddingFromInputEmbeddings(input_embeddings);
  // An interrupted LiteRT run reports a generic error. Use the latched state
  // rather than calling `should_cancel` again.
  if (!output.ok() && cancelled_ && !absl::IsCancelled(output.status())) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  return output;
}

absl::StatusOr<EmbeddingOutput>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbedding(
    const ExecutorInputs& inputs, const ComputeEmbeddingOptions& options) {
  if (embedding_lookup_ == nullptr) {
    return absl::FailedPreconditionError(
        "Computing embeddings from token IDs requires the embedder model "
        "(kTfLiteEmbedder), which the model does not contain. Use "
        "ComputeEmbeddingFromInputEmbeddings instead.");
  }
  ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  if (per_layer_embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(
        per_layer_embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }
  auto cleanup =
      absl::MakeCleanup([this]() {
        auto status = embedding_lookup_->CleanupMultiModalEmbeddings();
        if (!status.ok()) {
          ABSL_LOG(WARNING)
              << "Failed to cleanup multimodal embeddings: " << status;
        }
        if (per_layer_embedding_lookup_ != nullptr) {
          status = per_layer_embedding_lookup_->CleanupMultiModalEmbeddings();
          if (!status.ok()) {
            ABSL_LOG(WARNING)
                << "Failed to cleanup per layer multimodal embeddings: "
                << status;
          }
        }
      });

  ABSL_ASSIGN_OR_RETURN(auto text_data_ptr, inputs.GetTextDataPtr());
  RET_CHECK_NE(text_data_ptr, nullptr) << "TextData cannot be null.";

  LITERT_ASSIGN_OR_RETURN(
      auto token_ids_vec,
      CopyFromTensorBuffer<int32_t>(text_data_ptr->GetTokenIds()));

  int input_length = static_cast<int>(token_ids_vec.size());
  if (encoder_signatures_.empty()) {
    return absl::FailedPreconditionError("No encoder signatures available.");
  }

  size_t max_seq_len = std::prev(encoder_signatures_.end())->first;

  if (token_ids_vec.size() > max_seq_len) {
    switch (options.input_overflow_strategy) {
      case InputOverflowStrategy::kError: {
        return absl::InvalidArgumentError(
            absl::StrCat("Input sequence length (", token_ids_vec.size(),
                         ") exceeds maximum supported signature length (",
                         max_seq_len, ")."));
      }
      case InputOverflowStrategy::kTruncate: {
        if (!options.eos_token_ids.empty() &&
            options.eos_token_ids.size() < max_seq_len) {
          const size_t num_eos = options.eos_token_ids.size();
          token_ids_vec.resize(max_seq_len);
          std::copy(options.eos_token_ids.begin(), options.eos_token_ids.end(),
                    token_ids_vec.end() - num_eos);
        } else {
          token_ids_vec.resize(max_seq_len);
        }
        LITERT_ASSIGN_OR_RETURN(
            auto output_vector,
            RunEncoderForTokens(absl::MakeSpan(token_ids_vec)));
        return EmbeddingOutput{
            .embedding = std::move(output_vector),
            .input_length = input_length,
            .truncated_length = static_cast<int>(max_seq_len),
            .num_chunks = 1,
        };
      }
      case InputOverflowStrategy::kChunkAndAverage: {
        int num_chunks = static_cast<int>(
            (token_ids_vec.size() + max_seq_len - 1) / max_seq_len);
        std::vector<float> accumulated_embedding;
        for (int i = 0; i < num_chunks; ++i) {
          size_t start = i * max_seq_len;
          size_t end = std::min(start + max_seq_len, token_ids_vec.size());
          absl::Span<const int32_t> chunk_tokens =
              absl::MakeSpan(token_ids_vec.data() + start, end - start);
          LITERT_ASSIGN_OR_RETURN(auto chunk_embedding,
                                  RunEncoderForTokens(chunk_tokens));
          if (i == 0) {
            accumulated_embedding = std::move(chunk_embedding);
          } else {
            if (accumulated_embedding.size() != chunk_embedding.size()) {
              return absl::InternalError(
                  "Mismatch in embedding output dimensions between chunks.");
            }
            for (size_t j = 0; j < accumulated_embedding.size(); ++j) {
              accumulated_embedding[j] += chunk_embedding[j];
            }
          }
        }
        float scale = 1.0f / static_cast<float>(num_chunks);
        for (float& val : accumulated_embedding) {
          val *= scale;
        }
        return EmbeddingOutput{
            .embedding = std::move(accumulated_embedding),
            .input_length = input_length,
            .truncated_length = std::nullopt,
            .num_chunks = num_chunks,
        };
      }
    }
  }

  // Input length <= max_seq_len
  LITERT_ASSIGN_OR_RETURN(auto output_vector,
                          RunEncoderForTokens(absl::MakeSpan(token_ids_vec)));
  return EmbeddingOutput{
      .embedding = std::move(output_vector),
      .input_length = input_length,
      .truncated_length = std::nullopt,
      .num_chunks = 1,
  };
}

absl::StatusOr<std::vector<EmbeddingOutput>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingBatch(
    const std::vector<ExecutorInputs>& batch_inputs,
    const ComputeEmbeddingOptions& options) {
  std::vector<EmbeddingOutput> batch_outputs;
  batch_outputs.reserve(batch_inputs.size());
  // Batch inference isn't supported yet. Loop over the batch for now.
  for (const auto& inputs : batch_inputs) {
    ABSL_ASSIGN_OR_RETURN(auto output, ComputeEmbedding(inputs, options));
    batch_outputs.push_back(std::move(output));
  }
  return batch_outputs;
}

absl::string_view EmbeddingLiteRtCompiledModelExecutor::ExecutorBackendName()
    const {
  return backend_name_;
}

absl::StatusOr<std::vector<int>>
EmbeddingLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

absl::StatusOr<int>
EmbeddingLiteRtCompiledModelExecutor::GetEmbeddingDimension() const {
  return embedding_dimension_;
}

litert::Environment* EmbeddingLiteRtCompiledModelExecutor::GetEnvironment()
    const {
  return &env_;
}

absl::Status EmbeddingLiteRtCompiledModelExecutor::StartProfiling() {
  latency_stats_ = ExecutorStats();
  if (compiled_model_ != nullptr) {
    StartCompiledModelMetricsCollection(*compiled_model_);
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorStats>
EmbeddingLiteRtCompiledModelExecutor::StopProfiling() {
  if (!latency_stats_.has_value()) {
    return absl::FailedPreconditionError("Profiling has not been started.");
  }
  ExecutorStats stats = *std::move(latency_stats_);
  latency_stats_ = std::nullopt;
  stats.module_name = kEmbeddingModuleName;

  if (compiled_model_ != nullptr) {
    CollectCompiledModelMetrics(*compiled_model_, stats,
                                kTextEncoderMetricsPrefix);
  }

  auto num_tokens_metric = stats.GetMetric(kEmbeddingNumTokensMetric);
  int64_t num_tokens = 0;
  if (num_tokens_metric.has_value() &&
      std::holds_alternative<int64_t>(*num_tokens_metric)) {
    num_tokens = std::get<int64_t>(*num_tokens_metric);
  }

  auto num_padded_tokens_metric =
      stats.GetMetric(kEmbeddingNumPaddedTokensMetric);
  int64_t num_padded_tokens = 0;
  if (num_padded_tokens_metric.has_value() &&
      std::holds_alternative<int64_t>(*num_padded_tokens_metric)) {
    num_padded_tokens = std::get<int64_t>(*num_padded_tokens_metric);
  }

  absl::Duration total_latency = stats.GetTotalLatency();
  if (total_latency > absl::ZeroDuration()) {
    double total_seconds = absl::ToDoubleSeconds(total_latency);
    if (total_seconds > 0.0) {
      if (num_tokens > 0) {
        stats.Accumulate(kEmbeddingE2eTokensPerSecondMetric,
                         static_cast<double>(num_tokens) / total_seconds);
      }
      if (num_padded_tokens > 0) {
        stats.Accumulate(
            kEmbeddingE2ePaddedTokensPerSecondMetric,
            static_cast<double>(num_padded_tokens) / total_seconds);
      }
    }
  }

  auto encoder_latency = stats.GetLatency(kTextEncoderInferenceLatency);
  if (encoder_latency.has_value() && *encoder_latency > absl::ZeroDuration()) {
    double encoder_seconds = absl::ToDoubleSeconds(*encoder_latency);
    if (encoder_seconds > 0.0) {
      if (num_tokens > 0) {
        stats.Accumulate(kEmbeddingEncoderStackTokensPerSecondMetric,
                         static_cast<double>(num_tokens) / encoder_seconds);
      }
      if (num_padded_tokens > 0) {
        stats.Accumulate(
            kEmbeddingEncoderStackPaddedTokensPerSecondMetric,
            static_cast<double>(num_padded_tokens) / encoder_seconds);
      }
    }
  }

  return stats;
}

EmbeddingLiteRtCompiledModelExecutor::EmbeddingLiteRtCompiledModelExecutor(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
    std::unique_ptr<litert::CompiledModel> compiled_model,
    std::vector<int> expected_input_dimension, int embedding_dimension,
    std::map<int, size_t> encoder_signatures,
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        input_buffers_cache,
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        output_buffers_cache)
    : executor_settings_(std::move(executor_settings)),
      env_(env),
      resources_(std::move(resources)),
      embedding_lookup_(std::move(embedding_lookup)),
      per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
      compiled_model_(std::move(compiled_model)),
      expected_input_dimension_(std::move(expected_input_dimension)),
      embedding_dimension_(embedding_dimension),
      backend_name_(GetBackendString(executor_settings_.GetBackend())),
      encoder_signatures_(std::move(encoder_signatures)),
      input_buffers_cache_(std::move(input_buffers_cache)),
      output_buffers_cache_(std::move(output_buffers_cache)) {}

}  // namespace litert::lm

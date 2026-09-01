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
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_google_tensor_options.h"  // from @litert
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
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

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
absl::StatusOr<std::unique_ptr<EmbeddingLiteRtCompiledModelExecutor>>
EmbeddingLiteRtCompiledModelExecutor::Create(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::unique_ptr<ModelResources> resources) {
  RET_CHECK_NE(resources, nullptr) << "ModelResources is null.";

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      env, *resources, embedding_lookup, per_layer_embedding_lookup));
  if (embedding_lookup == nullptr) {
    return absl::NotFoundError(
        "kTfLiteEmbedder model not found in resources for embedding.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder_model,
      resources->GetTFLiteModel(ModelType::kTfLiteTextEncoder));
  RET_CHECK_NE(text_encoder_model, nullptr) << "TEXT_ENCODER model is null.";

  litert::Options options;
  switch (executor_settings.GetBackend()) {
    case Backend::CPU: {
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
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
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
      ABSL_ASSIGN_OR_RETURN(
          const auto cache_files,
          GetGpuModelCacheData(executor_settings,
                               EmbeddingExecutorSettings::kEncoderName));
      LITERT_RETURN_IF_ERROR(
          SetCommonGpuOptions(executor_settings, gpu_options));
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
                              options.GetQualcommOptions());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpPerformanceMode(
          qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
      LITERT_ASSIGN_OR_RETURN(auto& google_tensor_options,
                              options.GetGoogleTensorOptions());
      google_tensor_options.SetPerformanceMode(
          google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
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
      *resources, ModelType::kTfLiteTextEncoder, options));

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

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModelWrapper::Create(env, text_encoder_model->Get(), options));
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

  return absl::WrapUnique(new EmbeddingLiteRtCompiledModelExecutor(
      std::move(executor_settings), env, std::move(resources),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      std::move(compiled_model_ptr), std::move(expected_input_dimension),
      embedding_dimension, std::move(encoder_signatures)));
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

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::RunEncoderForTokens(
    absl::Span<const int32_t> tokens) {
  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder_model,
      resources_->GetTFLiteModel(ModelType::kTfLiteTextEncoder));

  // Find the smallest signature larger than or equal to the input size.
  auto it = encoder_signatures_.lower_bound(tokens.size());
  if (it == encoder_signatures_.end()) {
    return absl::InternalError(absl::StrCat(
        "No suitable signature found for sequence length ", tokens.size()));
  }
  size_t signature_index = it->second;

  // Lazily create input buffers.
  if (!input_buffers_cache_.contains(signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffers,
        compiled_model_->CreateInputBuffers(signature_index));
    input_buffers_cache_[signature_index] = std::move(input_buffers);
  }
  auto& input_buffers = input_buffers_cache_[signature_index];

  // Lazily create output buffers.
  if (!output_buffers_cache_.contains(signature_index)) {
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffers,
        compiled_model_->CreateOutputBuffers(signature_index));
    output_buffers_cache_[signature_index] = std::move(output_buffers);
  }
  auto& output_buffers = output_buffers_cache_[signature_index];
  LITERT_ASSIGN_OR_RETURN(auto sig,
                          text_encoder_model->GetSignature(signature_index));

  auto input_names = sig.InputNames();

  size_t embeddings_buffer_index = 0;
  std::optional<size_t> input_mask_buffer_index;
  std::optional<size_t> per_layer_embeddings_buffer_index;

  for (size_t i = 0; i < input_names.size(); ++i) {
    absl::string_view name = input_names[i];
    if (absl::StrContains(name, kPerLayerEmbeddingsName)) {
      per_layer_embeddings_buffer_index = i;
    } else if (absl::StrContains(name, kInputMaskName)) {
      input_mask_buffer_index = i;
    } else if (absl::StrContains(name, kEmbeddingsName)) {
      embeddings_buffer_index = i;
    }
  }

  ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
      tokens, &input_buffers[embeddings_buffer_index], /*token_offset=*/0));

  // Compute mask for text encoder. The mask masks out tokens outside of the
  // input sequence length, which can be smaller than the static size of the
  // TFLite graph.
  if (input_mask_buffer_index.has_value()) {
    auto& mask_buffer = input_buffers[*input_mask_buffer_index];
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            mask_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* mask_ptr = static_cast<float*>(mask_lock_and_addr.second);
    LITERT_ASSIGN_OR_RETURN(auto mask_size_bytes, mask_buffer.Size());
    size_t mask_elements = mask_size_bytes / sizeof(float);

    const size_t active_elements = std::min(tokens.size(), mask_elements);
    std::fill_n(mask_ptr, active_elements, 1.0f);
    if (active_elements < mask_elements) {
      std::fill(mask_ptr + active_elements, mask_ptr + mask_elements, 0.0f);
    }
  }

  if (per_layer_embeddings_buffer_index.has_value() &&
      per_layer_embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[*per_layer_embeddings_buffer_index],
        /*token_offset=*/0));
  }

  LITERT_RETURN_IF_ERROR(
      compiled_model_->Run(signature_index, input_buffers, output_buffers));

  LITERT_ASSIGN_OR_RETURN(auto output_vector,
                          CopyFromTensorBuffer<float>(output_buffers[0]));
  return output_vector;
}

absl::StatusOr<EmbeddingOutput>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbedding(
    const ExecutorInputs& inputs, const ComputeEmbeddingOptions& options) {
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

EmbeddingLiteRtCompiledModelExecutor::EmbeddingLiteRtCompiledModelExecutor(
    EmbeddingExecutorSettings executor_settings, Environment& env,
    std::unique_ptr<ModelResources> resources,
    std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
    std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
    std::unique_ptr<litert::CompiledModel> compiled_model,
    std::vector<int> expected_input_dimension, int embedding_dimension,
    std::map<int, size_t> encoder_signatures)
    : executor_settings_(std::move(executor_settings)),
      env_(env),
      resources_(std::move(resources)),
      embedding_lookup_(std::move(embedding_lookup)),
      per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
      compiled_model_(std::move(compiled_model)),
      expected_input_dimension_(std::move(expected_input_dimension)),
      embedding_dimension_(embedding_dimension),
      backend_name_(GetBackendString(executor_settings_.GetBackend())),
      encoder_signatures_(std::move(encoder_signatures)) {}

}  // namespace litert::lm

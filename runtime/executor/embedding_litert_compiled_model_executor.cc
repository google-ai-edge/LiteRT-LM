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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/btree_map.h"  // from @com_google_absl
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
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/embedding_lookup/embedding_lookup_text.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/cancellation_latch.h"
#include "runtime/executor/embedding/embedding_executor_base.h"
#include "runtime/executor/embedding/embedding_executor_settings.h"
#include "runtime/executor/embedding_litert_compiled_model_executor_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/run_options.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/streamed_weights_uploader.h"

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

class CompiledModelWrapper : public litert::CompiledModel {
 public:
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        const LiteRtModel litert_model,
                                        Options& compilation_options) {
    return litert::CompiledModel::Create(env, litert_model,
                                         compilation_options);
  }
};

}  // namespace

// static
absl::StatusOr<CompiledTextEncoderInfo>
EmbeddingLiteRtCompiledModelExecutor::CompileTextEncoder(
    const EmbeddingExecutorSettings& executor_settings, Environment& env,
    ModelResources& resources, std::shared_ptr<DataStream> weights_stream) {
  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder_model,
      resources.GetTFLiteModel(proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER));
  RET_CHECK_NE(text_encoder_model, nullptr) << "TEXT_ENCODER model is null.";

  ABSL_ASSIGN_OR_RETURN(
      auto signature_maps,
      GetTextEncoderSignatureMaps(*text_encoder_model,
                                  executor_settings.GetSelectedSignatures()));

  // Compiles the text encoder with the given signatures (all if empty).
  auto compile = [&](const std::vector<std::string>& selected_signature_names)
      -> absl::StatusOr<litert::CompiledModel> {
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
        LITERT_ASSIGN_OR_RETURN(
            auto& qualcomm_options,
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

    if (!selected_signature_names.empty()) {
      std::vector<absl::string_view> selected_signatures(
          selected_signature_names.begin(), selected_signature_names.end());
      LITERT_ASSIGN_OR_RETURN(auto& runtime_options,
                              options.GetRuntimeOptions());
      LITERT_RETURN_IF_ERROR(
          runtime_options.SetSelectedSignatures(selected_signatures));
    }

    // The uploader must outlive compilation. Destroying it afterwards drops
    // the weights stream, which discards whatever compilation didn't consume.
    // A stream can only be read once, so only the first compilation streams
    // the weights.
    std::unique_ptr<StreamedWeightsUploader> weights_uploader;
    if (weights_stream != nullptr) {
      weights_uploader =
          std::make_unique<StreamedWeightsUploader>(std::move(weights_stream));
      weights_uploader->SetOnOptions(options);
    }
    auto compiled_model_status_or =
        CompiledModelWrapper::Create(env, text_encoder_model->Get(), options);
    weights_uploader.reset();
    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                            std::move(compiled_model_status_or));
    return compiled_model;
  };

  // The compiled model and the buffers of every loaded encoder signature.
  struct CompiledSignatures {
    std::unique_ptr<litert::CompiledModel> compiled_model;
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        input_buffers_cache;
    absl::flat_hash_map<size_t, std::vector<litert::TensorBuffer>>
        output_buffers_cache;
  };
  // Compiles the text encoder and creates the buffers of all the signatures in
  // `maps`, which allocates their tensors.
  auto compile_and_allocate =
      [&](const std::vector<std::string>& selected_signature_names,
          const TextEncoderSignatureMaps& maps)
      -> absl::StatusOr<CompiledSignatures> {
    ABSL_ASSIGN_OR_RETURN(auto compiled_model,
                          compile(selected_signature_names));
    CompiledSignatures compiled;
    compiled.compiled_model =
        std::make_unique<litert::CompiledModel>(std::move(compiled_model));
    std::vector<size_t> signature_indices;
    size_t num_signatures = maps.single.size();
    for (const auto& [batch_size, signatures] : maps.batched) {
      num_signatures += signatures.size();
    }
    signature_indices.reserve(num_signatures);
    for (const auto& [seq_len, signature_index] : maps.single) {
      signature_indices.push_back(signature_index);
    }
    for (const auto& [batch_size, signatures] : maps.batched) {
      for (const auto& [seq_len, signature_index] : signatures) {
        signature_indices.push_back(signature_index);
      }
    }
    for (size_t signature_index : signature_indices) {
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffers,
          compiled.compiled_model->CreateInputBuffers(signature_index));
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffers,
          compiled.compiled_model->CreateOutputBuffers(signature_index));
      compiled.input_buffers_cache[signature_index] = std::move(input_buffers);
      compiled.output_buffers_cache[signature_index] =
          std::move(output_buffers);
    }
    return compiled;
  };

  // The streamed weights are consumed by the first compilation, so a model
  // compiled from streamed weights cannot be compiled again.
  const bool has_streamed_weights = weights_stream != nullptr;
  absl::StatusOr<CompiledSignatures> compiled_signatures = compile_and_allocate(
      executor_settings.GetSelectedSignatures(), signature_maps);
  if (!compiled_signatures.ok() && !signature_maps.batched.empty() &&
      !has_streamed_weights) {
    // A backend may not support the batched signatures (e.g. a GPU delegate
    // that cannot fully delegate them). Batching is an optimization, so fall
    // back to the batch size 1 signatures rather than failing. Note that this
    // retries on any compile or buffer allocation failure when batched
    // signatures are present.
    ABSL_LOG(WARNING) << "Failed to load the batched text encoder signatures, "
                         "retrying without them; batches will be computed one "
                         "sequence at a time. Error: "
                      << compiled_signatures.status();
    std::vector<std::string> single_signature_names;
    for (const auto& [seq_len, signature_index] : signature_maps.single) {
      LITERT_ASSIGN_OR_RETURN(
          auto signature, text_encoder_model->GetSignature(signature_index));
      single_signature_names.emplace_back(signature.Key());
    }
    signature_maps.batched.clear();
    compiled_signatures =
        compile_and_allocate(single_signature_names, signature_maps);
  }
  ABSL_RETURN_IF_ERROR(compiled_signatures.status());

  absl::btree_map<int, size_t>& encoder_signatures = signature_maps.single;
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

  CompiledTextEncoderInfo info;
  info.compiled_model = std::move(compiled_signatures->compiled_model);
  info.expected_input_dimension = std::move(expected_input_dimension);
  info.embedding_dimension = embedding_dimension;
  info.encoder_signatures = std::move(encoder_signatures);
  info.batched_encoder_signatures = std::move(signature_maps.batched);
  info.input_buffers_cache =
      std::move(compiled_signatures->input_buffers_cache);
  info.output_buffers_cache =
      std::move(compiled_signatures->output_buffers_cache);
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
  CompiledTextEncoderInfo& info = compiled_text_encoder_info;

  // Gets (or creates) the input and output buffers of a signature.
  struct SignatureBuffers {
    std::vector<litert::TensorBuffer>* inputs;
    std::vector<litert::TensorBuffer>* outputs;
  };
  auto get_buffers =
      [&](size_t signature_index) -> absl::StatusOr<SignatureBuffers> {
    auto [input_it, input_inserted] =
        info.input_buffers_cache.try_emplace(signature_index);
    if (input_inserted) {
      auto cleanup =
          absl::MakeCleanup([&] { info.input_buffers_cache.erase(input_it); });
      LITERT_ASSIGN_OR_RETURN(
          input_it->second,
          info.compiled_model->CreateInputBuffers(signature_index));
      std::move(cleanup).Cancel();
    }
    auto [output_it, output_inserted] =
        info.output_buffers_cache.try_emplace(signature_index);
    if (output_inserted) {
      auto cleanup = absl::MakeCleanup(
          [&] { info.output_buffers_cache.erase(output_it); });
      LITERT_ASSIGN_OR_RETURN(
          output_it->second,
          info.compiled_model->CreateOutputBuffers(signature_index));
      std::move(cleanup).Cancel();
    }
    return SignatureBuffers{.inputs = &input_it->second,
                            .outputs = &output_it->second};
  };

  // Drops the batched signatures that RunBatchedEncoderForTokens cannot run,
  // so that their inputs run on the remaining signatures instead of failing
  // at run time.
  if (!info.batched_encoder_signatures.empty() &&
      !info.encoder_signatures.empty()) {
    size_t embedding_floats_per_token = 0;
    if (embedding_lookup != nullptr &&
        embedding_lookup->GetTextEmbeddingLookup() != nullptr) {
      embedding_floats_per_token =
          embedding_lookup->GetTextEmbeddingLookup()->GetFloatsPerToken();
    } else if (!info.expected_input_dimension.empty() &&
               info.expected_input_dimension.back() > 0) {
      embedding_floats_per_token = info.expected_input_dimension.back();
    }
    std::optional<size_t> per_layer_floats_per_token;
    if (per_layer_embedding_lookup != nullptr &&
        per_layer_embedding_lookup->GetTextEmbeddingLookup() != nullptr) {
      per_layer_floats_per_token =
          per_layer_embedding_lookup->GetTextEmbeddingLookup()
              ->GetFloatsPerToken();
    }

    for (auto& [batch_size, signatures] : info.batched_encoder_signatures) {
      for (auto it = signatures.begin(); it != signatures.end();) {
        const auto [sequence_length, signature_index] = *it;
        // The batch size 1 signature that the same inputs would otherwise
        // run on.
        auto single_it = info.encoder_signatures.lower_bound(sequence_length);
        if (single_it == info.encoder_signatures.end()) --single_it;
        absl::Status status = [&]() -> absl::Status {
          ABSL_ASSIGN_OR_RETURN(auto single_buffers,
                                get_buffers(single_it->second));
          const auto& single_outputs = *single_buffers.outputs;
          RET_CHECK(!single_outputs.empty());
          ABSL_ASSIGN_OR_RETURN(
              size_t single_output_elements,
              GetFloat32Elements(single_outputs[0], "output"));
          ABSL_ASSIGN_OR_RETURN(auto batched_buffers,
                                get_buffers(signature_index));
          LITERT_ASSIGN_OR_RETURN(
              auto input_names,
              info.compiled_model->GetSignatureInputNames(signature_index));
          return ValidateBatchedSignature(
              GetEncoderInputIndices(input_names), *batched_buffers.inputs,
              *batched_buffers.outputs, batch_size, sequence_length,
              embedding_floats_per_token, per_layer_floats_per_token,
              single_output_elements);
        }();
        if (status.ok()) {
          ++it;
          continue;
        }
        ABSL_LOG(WARNING) << "Not using the batched text encoder signature "
                          << signature_index << " (" << batch_size << "x"
                          << sequence_length << "): " << status;
        info.input_buffers_cache.erase(signature_index);
        info.output_buffers_cache.erase(signature_index);
        it = signatures.erase(it);
      }
    }
  }
  absl::erase_if(info.batched_encoder_signatures,
                 [](const auto& entry) { return entry.second.empty(); });

  return absl::WrapUnique(new EmbeddingLiteRtCompiledModelExecutor(
      std::move(executor_settings), env, std::move(resources),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      std::move(compiled_text_encoder_info.compiled_model),
      std::move(compiled_text_encoder_info.expected_input_dimension),
      compiled_text_encoder_info.embedding_dimension,
      std::move(compiled_text_encoder_info.encoder_signatures),
      std::move(compiled_text_encoder_info.batched_encoder_signatures),
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
  return PrepareEncoderRunForSignature(it->second, it->first);
}

absl::StatusOr<EmbeddingLiteRtCompiledModelExecutor::EncoderRun>
EmbeddingLiteRtCompiledModelExecutor::PrepareEncoderRunForSignature(
    size_t signature_index, size_t padded_sequence_length) {
  EncoderRun run;
  run.signature_index = signature_index;
  run.padded_sequence_length = padded_sequence_length;

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
  const EncoderInputIndices input_indices = GetEncoderInputIndices(input_names);
  run.embeddings_buffer_index = input_indices.embeddings;
  run.input_mask_buffer_index = input_indices.input_mask;
  run.per_layer_embeddings_buffer_index = input_indices.per_layer_embeddings;
  return run;
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::FinishEncoderRun(
    const EncoderRun& run, size_t sequence_length, CancellationLatch* latch) {
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
    LITERT_ASSIGN_OR_RETURN(auto mask_size_bytes, mask_buffer.PackedSize());
    size_t mask_elements = mask_size_bytes / sizeof(float);

    const size_t active_elements = std::min(sequence_length, mask_elements);
    std::fill_n(mask_ptr, active_elements, 1.0f);
    if (active_elements < mask_elements) {
      std::fill(mask_ptr + active_elements, mask_ptr + mask_elements, 0.0f);
    }
  }

  if (latch != nullptr && latch->IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  {
    ABSL_ASSIGN_OR_RETURN(std::optional<litert::Options> run_options,
                          CreateRunOptions(run_options_factory_));
    ScopedLatency scoped(latency_stats_, kTextEncoderInferenceLatency);
    auto result = compiled_model_->Run(run.signature_index, input_buffers,
                                       output_buffers,
                                       GetRunOptionsPtr(run_options));
    // An interrupted run reports a generic error.
    if (!result.HasValue() && latch != nullptr && latch->IsCancelled()) {
      return absl::CancelledError("The text encoder run was cancelled.");
    }
    LITERT_RETURN_IF_ERROR(result);
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

  if (run.per_layer_embeddings_buffer_index.has_value()) {
    if (per_layer_embedding_lookup_ == nullptr) {
      return absl::FailedPreconditionError(
          "The text encoder requires per-layer embeddings, but the model has "
          "no per-layer embedder.");
    }
    ScopedLatency scoped(latency_stats_, kPerLayerEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
        tokens, &input_buffers[*run.per_layer_embeddings_buffer_index],
        /*token_offset=*/0));
  }

  return FinishEncoderRun(run, tokens.size(), /*latch=*/nullptr);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings) {
  return EncodeInputEmbeddings(input_embeddings, /*latch=*/nullptr);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::EncodeInputEmbeddings(
    absl::Span<const float> input_embeddings, CancellationLatch* latch) {
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
    ABSL_ASSIGN_OR_RETURN(
        const size_t buffer_elements,
        GetFloat32Elements(embeddings_buffer, "input embeddings"));
    RET_CHECK_LE(input_embeddings.size(), buffer_elements)
        << "The text encoder input buffer is too small.";
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            embeddings_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* embeddings_ptr = static_cast<float*>(lock_and_addr.second);
    absl::c_copy(input_embeddings, embeddings_ptr);
    std::fill(embeddings_ptr + input_embeddings.size(),
              embeddings_ptr + buffer_elements, 0.0f);
  }

  return FinishEncoderRun(run, sequence_length, latch);
}

absl::StatusOr<std::vector<float>>
EmbeddingLiteRtCompiledModelExecutor::ComputeEmbeddingFromInputEmbeddings(
    absl::Span<const float> input_embeddings,
    absl::FunctionRef<bool()> should_cancel) {
  RET_CHECK_NE(compiled_model_, nullptr) << "The text encoder is not compiled.";
  CancellationLatch latch(should_cancel, {compiled_model_.get()});
  if (latch.IsCancelled()) {
    return absl::CancelledError("The text encoder run was cancelled.");
  }
  return EncodeInputEmbeddings(input_embeddings, &latch);
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
        ABSL_ASSIGN_OR_RETURN(
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
          ABSL_ASSIGN_OR_RETURN(auto chunk_embedding,
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
  ABSL_ASSIGN_OR_RETURN(auto output_vector,
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
  const size_t num_inputs = batch_inputs.size();
  if (encoder_signatures_.empty()) {
    return absl::FailedPreconditionError("No encoder signatures available.");
  }
  const size_t max_seq_len = std::prev(encoder_signatures_.end())->first;
  const bool can_batch = !batched_encoder_signatures_.empty() &&
                         embedding_lookup_ != nullptr && num_inputs > 1;
  const bool fail_on_overflow =
      options.input_overflow_strategy == InputOverflowStrategy::kError;
  const size_t max_batched_seq_len =
      can_batch ? static_cast<size_t>(GetMaxBatchedSequenceLength(
                      encoder_signatures_, batched_encoder_signatures_))
                : 0;

  // The token IDs of the inputs that can be batched (referenced in place if
  // possible, copied otherwise) and their lengths (0 for the other inputs).
  std::vector<absl::Span<const int32_t>> token_ids(num_inputs);
  std::vector<std::vector<int32_t>> token_id_copies(num_inputs);
  std::vector<size_t> batch_lengths(num_inputs, 0);
  if (can_batch || fail_on_overflow) {
    for (size_t i = 0; i < num_inputs; ++i) {
      const ExecutorInputs& inputs = batch_inputs[i];
      ABSL_ASSIGN_OR_RETURN(auto text_data_ptr, inputs.GetTextDataPtr());
      RET_CHECK_NE(text_data_ptr, nullptr) << "TextData cannot be null.";
      const litert::TensorBuffer& token_buffer = text_data_ptr->GetTokenIds();
      LITERT_ASSIGN_OR_RETURN(auto token_type, token_buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(const size_t num_tokens,
                              token_type.Layout().NumElements());
      // Fail before running any text encoder, as ComputeEmbedding would.
      if (fail_on_overflow && num_tokens > max_seq_len) {
        return absl::InvalidArgumentError(
            absl::StrCat("Input sequence length (", num_tokens,
                         ") exceeds maximum supported signature length (",
                         max_seq_len, ")."));
      }
      // Multimodal inputs need per-input embedding updates, so they run one
      // at a time below.
      if (!can_batch || num_tokens == 0 || num_tokens > max_batched_seq_len ||
          inputs.GetVisionDataPtr().ok() || inputs.GetAudioDataPtr().ok()) {
        continue;
      }
      if (auto span = ReferTensorBufferAsSpan<int32_t>(token_buffer);
          span.HasValue()) {
        token_ids[i] = *span;
      } else {
        LITERT_ASSIGN_OR_RETURN(token_id_copies[i],
                                CopyFromTensorBuffer<int32_t>(token_buffer));
        token_ids[i] = token_id_copies[i];
      }
      RET_CHECK_EQ(token_ids[i].size(), num_tokens);
      // Negative tokens are multimodal placeholders, which the batched path
      // does not resolve.
      if (absl::c_any_of(token_ids[i], [](int32_t t) { return t < 0; })) {
        token_ids[i] = {};
        token_id_copies[i] = {};
        continue;
      }
      batch_lengths[i] = num_tokens;
    }
  }

  std::vector<std::optional<EmbeddingOutput>> outputs(num_inputs);
  for (const BatchedTextEncoderGroup& group : GroupInputsForBatchedSignatures(
           batch_lengths, encoder_signatures_, batched_encoder_signatures_)) {
    std::vector<absl::Span<const int32_t>> batch_tokens;
    batch_tokens.reserve(group.inputs.size());
    for (size_t index : group.inputs) {
      batch_tokens.push_back(token_ids[index]);
    }
    ABSL_ASSIGN_OR_RETURN(
        auto embeddings,
        RunBatchedEncoderForTokens(group.signature_index,
                                   group.padded_sequence_length, batch_tokens));
    RET_CHECK_EQ(embeddings.size(), group.inputs.size());
    for (size_t j = 0; j < group.inputs.size(); ++j) {
      const size_t index = group.inputs[j];
      outputs[index] = EmbeddingOutput{
          .embedding = std::move(embeddings[j]),
          .input_length = static_cast<int>(batch_lengths[index]),
          .truncated_length = std::nullopt,
          .num_chunks = 1,
      };
    }
  }

  // Compute the remaining inputs one at a time.
  std::vector<EmbeddingOutput> batch_outputs;
  batch_outputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (!outputs[i].has_value()) {
      ABSL_ASSIGN_OR_RETURN(outputs[i],
                            ComputeEmbedding(batch_inputs[i], options));
    }
    batch_outputs.push_back(*std::move(outputs[i]));
  }
  return batch_outputs;
}

absl::StatusOr<std::vector<std::vector<float>>>
EmbeddingLiteRtCompiledModelExecutor::RunBatchedEncoderForTokens(
    size_t signature_index, size_t padded_sequence_length,
    absl::Span<const absl::Span<const int32_t>> batch_tokens) {
  ScopedLatency scoped_total_latency(latency_stats_);
  RET_CHECK_NE(embedding_lookup_, nullptr);
  const size_t batch_size = batch_tokens.size();
  RET_CHECK_GT(batch_size, 0);

  ABSL_ASSIGN_OR_RETURN(
      EncoderRun run,
      PrepareEncoderRunForSignature(signature_index, padded_sequence_length));
  std::vector<litert::TensorBuffer>& input_buffers = *run.input_buffers;
  std::vector<litert::TensorBuffer>& output_buffers = *run.output_buffers;
  // The signature was validated when the executor was created (see
  // ValidateBatchedSignature); these checks only guard the buffer accesses.
  RET_CHECK(!output_buffers.empty())
      << "The batched text encoder has no output buffers.";
  RET_CHECK(!run.per_layer_embeddings_buffer_index.has_value() ||
            per_layer_embedding_lookup_ != nullptr)
      << "The batched text encoder requires per-layer embeddings, but the "
         "model has no per-layer embedder.";

  // Writes the embeddings of every sequence into its row of `buffer`, padding
  // each row to `padded_sequence_length` with the default embedding.
  auto fill_embeddings = [&](EmbeddingLookupManager& lookup,
                             litert::TensorBuffer& buffer) -> absl::Status {
    EmbeddingLookupText* text_lookup = lookup.GetTextEmbeddingLookup();
    RET_CHECK_NE(text_lookup, nullptr);
    const size_t floats_per_token = text_lookup->GetFloatsPerToken();
    const std::vector<float>& default_embedding =
        text_lookup->GetDefaultEmbeddingVector();
    RET_CHECK_EQ(default_embedding.size(), floats_per_token);
    const size_t floats_per_row = padded_sequence_length * floats_per_token;

    ABSL_ASSIGN_OR_RETURN(size_t buffer_elements,
                          GetFloat32Elements(buffer, "embeddings"));
    if (buffer_elements != batch_size * floats_per_row) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unexpected batched text encoder input size: ", buffer_elements,
          " floats, expected ", batch_size, " x ", padded_sequence_length,
          " x ", floats_per_token, "."));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            buffer, litert::TensorBuffer::LockMode::kWrite));
    float* row_ptr = static_cast<float*>(lock_and_addr.second);
    std::vector<float> token_embedding(floats_per_token);
    for (absl::Span<const int32_t> tokens : batch_tokens) {
      RET_CHECK_LE(tokens.size(), padded_sequence_length);
      float* token_ptr = row_ptr;
      for (int32_t token : tokens) {
        ABSL_RETURN_IF_ERROR(lookup.LookupPrefill(token, token_embedding));
        absl::c_copy(token_embedding, token_ptr);
        token_ptr += floats_per_token;
      }
      for (size_t t = tokens.size(); t < padded_sequence_length; ++t) {
        absl::c_copy(default_embedding, token_ptr);
        token_ptr += floats_per_token;
      }
      row_ptr += floats_per_row;
    }
    return absl::OkStatus();
  };

  {
    ScopedLatency scoped(latency_stats_, kEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(fill_embeddings(
        *embedding_lookup_, input_buffers[run.embeddings_buffer_index]));
  }
  if (run.per_layer_embeddings_buffer_index.has_value()) {
    ScopedLatency scoped(latency_stats_, kPerLayerEmbedderLookupLatency);
    ABSL_RETURN_IF_ERROR(
        fill_embeddings(*per_layer_embedding_lookup_,
                        input_buffers[*run.per_layer_embeddings_buffer_index]));
  }

  // The mask must have exactly `padded_sequence_length` elements per row (e.g.
  // `[B, T]` or `[B, T, 1]`); unmask each row's own tokens.
  if (run.input_mask_buffer_index.has_value()) {
    ScopedLatency scoped(latency_stats_, kMaskPreparationLatency);
    auto& mask_buffer = input_buffers[*run.input_mask_buffer_index];
    ABSL_ASSIGN_OR_RETURN(const size_t mask_elements,
                          GetFloat32Elements(mask_buffer, "input mask"));
    if (mask_elements != batch_size * padded_sequence_length) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unexpected batched text encoder input mask size: ", mask_elements,
          " floats, expected ", batch_size, " x ", padded_sequence_length,
          "."));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto mask_lock_and_addr,
        litert::TensorBufferScopedLock::Create(
            mask_buffer, litert::TensorBuffer::LockMode::kWrite));
    float* mask_ptr = static_cast<float*>(mask_lock_and_addr.second);
    for (size_t b = 0; b < batch_size; ++b) {
      float* row = mask_ptr + b * padded_sequence_length;
      const size_t active = batch_tokens[b].size();
      std::fill_n(row, active, 1.0f);
      std::fill(row + active, row + padded_sequence_length, 0.0f);
    }
  }

  {
    ABSL_ASSIGN_OR_RETURN(std::optional<litert::Options> run_options,
                          CreateRunOptions(run_options_factory_));
    ScopedLatency scoped(latency_stats_, kTextEncoderInferenceLatency);
    LITERT_RETURN_IF_ERROR(compiled_model_->Run(run.signature_index,
                                                input_buffers, output_buffers,
                                                GetRunOptionsPtr(run_options)));
  }

  std::vector<std::vector<float>> embeddings;
  {
    ScopedLatency scoped(latency_stats_, kOutputCopyLatency);
    LITERT_ASSIGN_OR_RETURN(auto output_vector,
                            CopyFromTensorBuffer<float>(output_buffers[0]));
    if (output_vector.size() % batch_size != 0) {
      return absl::InternalError(absl::StrCat(
          "The batched text encoder output size (", output_vector.size(),
          ") is not a multiple of the batch size (", batch_size, ")."));
    }
    const size_t row_size = output_vector.size() / batch_size;
    embeddings.reserve(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
      embeddings.emplace_back(output_vector.begin() + b * row_size,
                              output_vector.begin() + (b + 1) * row_size);
    }
  }

  int64_t num_tokens = 0;
  for (absl::Span<const int32_t> tokens : batch_tokens) {
    num_tokens += tokens.size();
  }
  AccumulateStat(latency_stats_, kEmbeddingNumTokensMetric, num_tokens);
  AccumulateStat(latency_stats_, kEmbeddingNumPaddedTokensMetric,
                 static_cast<int64_t>(batch_size * padded_sequence_length));
  return embeddings;
}

absl::string_view EmbeddingLiteRtCompiledModelExecutor::ExecutorBackendName()
    const {
  return backend_name_;
}

absl::StatusOr<std::vector<int>>
EmbeddingLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

std::vector<TextEncoderVariant>
EmbeddingLiteRtCompiledModelExecutor::GetSupportedVariants() const {
  // The input embeddings are [batch, seq_len, embed_dim] or [seq_len,
  // embed_dim]. A dynamic (non-positive) batch dimension is run with a batch
  // size of 1.
  const int batch_size = (expected_input_dimension_.size() == 3 &&
                          expected_input_dimension_[0] > 0)
                             ? expected_input_dimension_[0]
                             : 1;
  std::vector<TextEncoderVariant> variants;
  variants.reserve(encoder_signatures_.size());
  // encoder_signatures_ is ordered by ascending sequence length.
  for (const auto& [sequence_length, signature_index] : encoder_signatures_) {
    variants.push_back(
        {.batch_size = batch_size, .sequence_length = sequence_length});
  }
  return variants;
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
    absl::btree_map<int, size_t> encoder_signatures,
    absl::btree_map<int, absl::btree_map<int, size_t>>
        batched_encoder_signatures,
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
      batched_encoder_signatures_(std::move(batched_encoder_signatures)),
      input_buffers_cache_(std::move(input_buffers_cache)),
      output_buffers_cache_(std::move(output_buffers_cache)) {}

}  // namespace litert::lm

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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_executor_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/executor/vision_executor_utils.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/data_stream.h"
#ifdef __EMSCRIPTEN__
#include "runtime/util/file_data_stream.h"
#endif
#include "runtime/util/litert_lm_streaming_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/logging.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {

#ifdef __EMSCRIPTEN__
void SetCurrentlyCompilingModel(ModelType model_type);
#endif

namespace {
// Global map of stored weights streams, mapped by ModelType.
#ifndef __EMSCRIPTEN__
absl::Mutex& GetStoredWeightsStreamsMutex() {
  static auto* const m = new absl::Mutex();
  return *m;
}
#endif

std::unordered_map<ModelType, std::shared_ptr<DataStream>>&
GetStoredWeightsStreams() {
  static auto* const m =
      new std::unordered_map<ModelType, std::shared_ptr<DataStream>>();
  return *m;
}
absl::StatusOr<std::unique_ptr<CompiledModel>> CompileModel(
    const LlmExecutorSettings& executor_settings, Environment& lrt_env,
    ModelResources& resources, ModelType model_type) {
  ABSL_ASSIGN_OR_RETURN(auto model_buffer_view,
                        resources.GetTFLiteModelBuffer(model_type));
  litert::BufferRef<uint8_t> model_buffer(
      reinterpret_cast<const uint8_t*>(model_buffer_view.data()),
      model_buffer_view.size());

  ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);

  std::optional<ModelSignatures> signatures;
  std::optional<std::string> cache_suffix;

  if (model_type == ModelType::kTfLitePrefillDecode) {
    ABSL_ASSIGN_OR_RETURN(auto litert_model,
                          resources.GetTFLiteModel(model_type));
    if (!litert_model || !*litert_model) {
      return absl::InternalError("Failed to build LiteRt model");
    }
    LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                            litert_model->FindSignature("decode"));
    ABSL_ASSIGN_OR_RETURN(
        ModelSignatures sigs,
        GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                               decode_signature.OutputNames()));
    signatures = sigs;
  } else if (model_type == ModelType::kTfLiteMtpDrafter) {
    cache_suffix = std::string(ExecutorSettingsBase::kMtpDrafterCacheSuffix);
  }

  ABSL_ASSIGN_OR_RETURN(auto compilation_options,
                        CreateCompilationOptions(
                            executor_settings, activation_data_type,
                            signatures ? &*signatures : nullptr, cache_suffix));
  ABSL_RETURN_IF_ERROR(
      UpdateCompilationOptions(executor_settings, compilation_options));

  std::unique_ptr<CompiledModel> compiled_model;
  {
#ifdef __EMSCRIPTEN__
    SetCurrentlyCompilingModel(model_type);
#endif
    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model_tmp,
        CompiledModel::Create(lrt_env, model_buffer, compilation_options));
#ifdef __EMSCRIPTEN__
    SetCurrentlyCompilingModel(ModelType::kUnknown);
#endif
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  return compiled_model;
}

ModelType g_currently_compiling_model = ModelType::kUnknown;
std::atomic<int> g_temp_file_counter(0);
}  // namespace

void SetCurrentlyCompilingModel(ModelType model_type) {
  g_currently_compiling_model = model_type;
}

ModelType GetCurrentlyCompilingModel() { return g_currently_compiling_model; }

void StoreWeightsStream(ModelType model_type,
                        std::shared_ptr<DataStream> stream) {
#ifndef __EMSCRIPTEN__
  absl::MutexLock lock(&GetStoredWeightsStreamsMutex());
#endif
  GetStoredWeightsStreams()[model_type] = std::move(stream);
}

absl::Status ReadStoredWeights(int model_type_int, uint64_t offset,
                               uint64_t size, void* buffer) {
  ModelType model_type = static_cast<ModelType>(model_type_int);
  std::shared_ptr<DataStream> stream;
  {
#ifndef __EMSCRIPTEN__
    absl::MutexLock lock(&GetStoredWeightsStreamsMutex());
#endif
    auto& streams = GetStoredWeightsStreams();
    auto it = streams.find(model_type);
    if (it == streams.end() || it->second == nullptr) {
      return absl::NotFoundError(absl::StrCat(
          "Stored weights stream not found for model type: ", model_type_int));
    }
    stream = it->second;
  }
  return stream->ReadAndDiscard(buffer, offset, size);
}

absl::Status ClearStoredWeightsStreams() {
#ifndef __EMSCRIPTEN__
  absl::MutexLock lock(&GetStoredWeightsStreamsMutex());
#endif
  auto& streams = GetStoredWeightsStreams();
  for (auto& [model_type, stream] : streams) {
    if (stream) {
      (void)stream->Discard(0, UINT64_MAX);
    }
  }
  streams.clear();
  return absl::OkStatus();
}

class StreamingWeightsModelResources : public ModelResources {
 public:
  explicit StreamingWeightsModelResources(
      const proto::LlmMetadata& llm_metadata)
      : llm_metadata_(llm_metadata) {}

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override {
    auto it = models_.find(model_type);
    if (it != models_.end() && it->second != nullptr) {
      return it->second.get();
    }

    auto buf_it = model_buffers_.find(model_type);
    if (buf_it == model_buffers_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Model buffer not found for type: ", static_cast<int>(model_type)));
    }

    auto expected_model =
        litert::Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
            reinterpret_cast<uint8_t*>(buf_it->second.data()),
            buf_it->second.size()));
    if (!expected_model.HasValue()) {
      return absl::InternalError(
          absl::StrCat("Failed to create model from buffer for type: ",
                       static_cast<int>(model_type)));
    }
    models_[model_type] =
        std::make_unique<litert::Model>(std::move(*expected_model));
    return models_[model_type].get();
  }

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override {
    auto it = model_buffers_.find(model_type);
    if (it == model_buffers_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Model buffer not found for type: ", static_cast<int>(model_type)));
    }
    return absl::string_view(it->second.data(), it->second.size());
  }

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    if (per_layer_weights_file_.has_value()) {
      return std::ref(*per_layer_weights_file_);
    }
    return absl::UnimplementedError("GetScopedFile not implemented.");
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    if (model_type == ModelType::kTfLitePerLayerEmbedder &&
        per_layer_weights_section_offset_.has_value()) {
      return *per_layer_weights_section_offset_;
    }
    return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
  }

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    return std::nullopt;
  }

  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    return std::nullopt;
  }

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override {
    return absl::UnimplementedError("Not implemented.");
  }

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override {
    return &llm_metadata_;
  }

  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetTFLiteModelSectionFileRegion not implemented.");
  }

  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata()
      override {
    return absl::UnimplementedError("GetExecutorMetadata not implemented.");
  }

  void SetLlmMetadata(const proto::LlmMetadata& llm_metadata) {
    llm_metadata_ = llm_metadata;
  }

  void SetModelBuffer(ModelType model_type, std::vector<char> buffer) {
    model_buffers_[model_type] = std::move(buffer);
  }

  absl::Status SetPerLayerWeightsFromStream(DataStream& stream, size_t size) {
    std::string path =
        absl::StrCat("/tmp/per_layer_embedder_weights_", getpid(), "_",
                     g_temp_file_counter.fetch_add(1), ".bin");
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      return absl::InternalError(
          "Failed to open temporary weights file for writing");
    }

    constexpr size_t kChunkSize = 1024 * 1024;  // 1MB chunk
    std::vector<char> chunk(kChunkSize);
    size_t remaining = size;
    size_t offset = 0;
    while (remaining > 0) {
      size_t to_read = std::min(kChunkSize, remaining);
      ABSL_RETURN_IF_ERROR(
          stream.ReadAndDiscard(chunk.data(), offset, to_read));
      out.write(chunk.data(), to_read);
      remaining -= to_read;
      offset += to_read;
    }
    out.close();

    auto scoped_file_or = ScopedFile::Open(path);
    if (!scoped_file_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to open temporary weights file: ",
                       scoped_file_or.status().message()));
    }
    unlink(path.c_str());
    per_layer_weights_file_ = std::move(*scoped_file_or);
    per_layer_weights_section_offset_ = {0, size};
    ABSL_LOG(INFO) << "Saved per-layer embedder weights to virtual file: "
                   << path << " (size: " << size << ") using chunked streaming";
    return absl::OkStatus();
  }

 private:
  proto::LlmMetadata llm_metadata_;
  std::unordered_map<ModelType, std::vector<char>> model_buffers_;
  std::unordered_map<ModelType, std::unique_ptr<litert::Model>> models_;
  std::optional<ScopedFile> per_layer_weights_file_;
  std::optional<std::pair<size_t, size_t>> per_layer_weights_section_offset_;
};

class EngineAdvancedImpl : public Engine {
 public:
  ~EngineAdvancedImpl() override {
    auto status = WaitUntilDone(Engine::kDefaultTimeout);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for engine to finish: " << status;
    }

    if (living_sessions_ > 0) {
      ABSL_LOG(ERROR) << "EngineAdvancedImpl destructed with "
                      << living_sessions_ << " living sessions!";
    }

    execution_manager_.reset();
    owned_env_.reset();
    tokenizer_.reset();
    litert_model_resources_.reset();
  }

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  static absl::StatusOr<std::unique_ptr<Engine>> CreateStreamingWeights(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  EngineAdvancedImpl(EngineSettings engine_settings,
                     std::unique_ptr<ModelResources> litert_model_resources,
                     std::unique_ptr<OwnedEnvironment> owned_env,
                     std::unique_ptr<Tokenizer> tokenizer,
                     std::unique_ptr<ExecutionManager> execution_manager,
                     std::optional<BenchmarkInfo> benchmark_info)
      : engine_settings_(std::move(engine_settings)),
        litert_model_resources_(std::move(litert_model_resources)),
        owned_env_(std::move(owned_env)),
        tokenizer_(std::move(tokenizer)),
        execution_manager_(std::move(execution_manager)),
        benchmark_info_(std::move(benchmark_info)) {}

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override {
    std::optional<BenchmarkInfo> session_benchmark_info;
    if (benchmark_info_.has_value()) {
      // Each session will have its own benchmark info, which will be populated
      // with the session-specific information.
      session_benchmark_info = benchmark_info_;
      ABSL_RETURN_IF_ERROR(session_benchmark_info->TimeInitPhaseStart(
          BenchmarkInfo::InitPhase::kSession));
    }

    SessionConfig config = session_config;
    // NOTE: Consider moving this logic to be part of the SessionConfig
    // class.
    ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));

    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "Model resources are not initialized.");
    }

    ABSL_ASSIGN_OR_RETURN(
        auto session,
        SessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                std::move(session_benchmark_info),
                                &living_sessions_));

    if (benchmark_info_.has_value()) {
      auto session_benchmark_info_or = session->GetMutableBenchmarkInfo();
      if (session_benchmark_info_or.ok()) {
        ABSL_RETURN_IF_ERROR(
            session_benchmark_info_or.value()->TimeInitPhaseEnd(
                BenchmarkInfo::InitPhase::kSession));
      }
    }
    return session;
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return execution_manager_->WaitUntilAllDone(timeout);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

  const Tokenizer& GetTokenizer() const override { return *tokenizer_; }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return GetAudioExecutorPropertiesFromModelResources(
        *litert_model_resources_);
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return GetVisionExecutorPropertiesFromModelResources(
        *litert_model_resources_);
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;

  // Model resources, which must outlive `executor_`.
  std::unique_ptr<ModelResources> litert_model_resources_;

  // Owned environment, which must outlive `executor_`.
  std::unique_ptr<OwnedEnvironment> owned_env_;

  // Tokenizer shared by all sessions.
  std::unique_ptr<Tokenizer> tokenizer_;

  // Execution manager for the engine. All additional pointers to this object
  // must be weak pointers. The ultimate ownership of this object is in the
  // EngineAdvancedImpl.
  std::shared_ptr<ExecutionManager> execution_manager_;

  // Counter for living sessions.
  std::atomic<int> living_sessions_{0};

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> EngineAdvancedImpl::Create(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
#ifdef __EMSCRIPTEN__
  const Backend backend =
      engine_settings.GetMainExecutorSettings().GetBackend();
  // Path-to-stream upgrade logic for GPU on Web
  if (backend == Backend::GPU) {
    const auto& model_assets =
        engine_settings.GetMainExecutorSettings().GetModelAssets();
    if (!model_assets.HasDataStream()) {
      auto has_external_weights_or = ModelHasExternalWeights(model_assets);
      if (has_external_weights_or.ok() && *has_external_weights_or) {
        ABSL_LOG(INFO) << "External weights detected on GPU/WASM. "
                          "Upgrading to streaming path...";
        auto path_or = model_assets.GetPath();
        if (path_or.ok()) {
          std::string path = std::string(*path_or);
          auto file_stream_or = FileDataStream::Create(path);
          if (file_stream_or.ok()) {
            auto streaming_model_assets_or =
                ModelAssets::Create(std::move(*file_stream_or));
            if (streaming_model_assets_or.ok()) {
              engine_settings.GetMutableMainExecutorSettings().SetModelAssets(
                  std::move(*streaming_model_assets_or));
            } else {
              ABSL_LOG(ERROR) << "Failed to create streaming ModelAssets: "
                              << streaming_model_assets_or.status();
            }
          } else {
            ABSL_LOG(ERROR) << "Failed to create FileDataStream: "
                            << file_stream_or.status();
          }
        } else {
          ABSL_LOG(ERROR) << "Failed to get model path for streaming upgrade: "
                          << path_or.status();
        }
      }
    }
  }
#endif

  if (engine_settings.GetMainExecutorSettings()
          .GetModelAssets()
          .HasDataStream()) {
    return CreateStreamingWeights(engine_settings, input_prompt_as_hint);
  }

  std::optional<BenchmarkInfo> benchmark_info =
      engine_settings.IsBenchmarkEnabled()
          ? std::make_optional<BenchmarkInfo>(
                engine_settings.GetBenchmarkParams().value())
          : std::nullopt;

  const auto& advanced_settings =
      engine_settings.GetMainExecutorSettings().GetAdvancedSettings();
  // Magic-number replacement mutates the model flatbuffer in place.
  const bool enable_file_backed_model_loading =
      engine_settings.GetMainExecutorSettings().GetBackend() == Backend::NPU &&
      advanced_settings && !advanced_settings->configure_magic_numbers;

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kModelAssets));
  }
  const auto& model_assets =
      engine_settings.GetMutableMainExecutorSettings().GetModelAssets();
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        BuildLiteRtCompiledModelResources(
                            model_assets, enable_file_backed_model_loading));
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kModelAssets));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }

  ABSL_ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }
  bool hasLlmModelType = llm_metadata->has_llm_model_type();
  absl::Duration tokenizer_duration = absl::ZeroDuration();
  // This lambda is used to create the tokenizer asynchronously if the model
  // type is available, such that the tokenizer can be created in parallel with
  // the executor.
  auto create_tokenizer =
      [&tokenizer_duration,
       &model_resources]() -> absl::StatusOr<std::unique_ptr<Tokenizer>> {
    absl::Time start_time = absl::Now();
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                          model_resources->GetTokenizer());
    tokenizer_duration = absl::Now() - start_time;
    return std::move(tokenizer);
  };

  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();

  std::future<absl::StatusOr<std::unique_ptr<Tokenizer>>> tokenizer_future;
  std::unique_ptr<Tokenizer> tokenizer;
  if (!hasLlmModelType) {
    ABSL_VLOG(1)
        << "Legacy model files don't have LlmModelType, loading tokenizer now";
    ABSL_ASSIGN_OR_RETURN(tokenizer, create_tokenizer());
    // Update and load the parameters from the model file and convert the
    // tokens to ids.
    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
  } else {
    // If the model type is available, wait for the tokenizer to be created
    // after the model is loaded.
    ABSL_VLOG(1) << "New model files have LlmModelType, loading tokenizer "
                    "asynchronously";

    if (engine_settings.GetParallelFileSectionLoading()) {
      // Launch the tokenizer creation in a separate thread in parallel with the
      // model loading.
      tokenizer_future = std::async(std::launch::async, create_tokenizer);
    } else {
      // Launch the tokenizer creation in the same thread.
      tokenizer_future = std::async(std::launch::deferred, create_tokenizer);
    }

    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        nullptr, llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }

  std::unique_ptr<OwnedEnvironment> owned_env;
  {
    ABSL_ASSIGN_OR_RETURN(
        auto temp_owned_env,
        CreateEnvironment(engine_settings, model_resources.get()));
    owned_env = std::make_unique<OwnedEnvironment>(std::move(temp_owned_env));
  }

  std::unique_ptr<LlmExecutor> executor;

  switch (main_executor_settings.GetBackend()) {
    default: {
      ABSL_ASSIGN_OR_RETURN(executor, CreateLlmLiteRtCompiledModelExecutor(
                                          main_executor_settings,
                                          owned_env->env, *model_resources));
    }
  };

  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_ptr;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    vision_executor_settings_ptr = std::make_unique<VisionExecutorSettings>(
        std::move(engine_settings.GetVisionExecutorSettings().value()));
    if (vision_executor_settings_ptr->GetAdapterBackend() != Backend::CPU) {
      ABSL_LOG(WARNING) << "Vision adapter backend is not CPU, which may cause "
                           "precision loss.";
    }
  }

  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_ptr;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    audio_executor_settings_ptr = std::make_unique<AudioExecutorSettings>(
        std::move(engine_settings.GetAudioExecutorSettings().value()));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kExecutor));
  }

  if (hasLlmModelType) {
    // Now load the tokenizer and update the engine settings.
    ABSL_ASSIGN_OR_RETURN(tokenizer, tokenizer_future.get());
    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
    // As we load the tokenizer asynchronously, we need to update the executor
    // settings after the tokenizer is loaded.
    ABSL_RETURN_IF_ERROR(executor->UpdateExecutorSettings(
        engine_settings.GetMainExecutorSettings()));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kTokenizer, tokenizer_duration));
  }
  std::unique_ptr<ExecutionManager> execution_manager;
  if (!engine_settings.GetSingleThreadedExecution()) {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        ThreadedExecutionManager::Create(
            tokenizer.get(), model_resources.get(), std::move(executor),
            std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), model_resources.get(), std::move(executor),
            std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  auto llm_impl = std::make_unique<EngineAdvancedImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(owned_env), std::move(tokenizer), std::move(execution_manager),
      std::move(benchmark_info));

  return llm_impl;
};

absl::StatusOr<std::unique_ptr<Engine>>
EngineAdvancedImpl::CreateStreamingWeights(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  ABSL_LOG(INFO) << "Constructing EngineAdvancedImpl from a weight stream...";

  std::optional<BenchmarkInfo> benchmark_info;
  if (engine_settings.IsBenchmarkEnabled()) {
    benchmark_info = std::make_optional<BenchmarkInfo>(
        engine_settings.GetBenchmarkParams().value());
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<litert::lm::DataStream> data_stream,
                        engine_settings.GetMainExecutorSettings()
                            .GetModelAssets()
                            .GetDataStream());
  ABSL_LOG(INFO) << "Got data stream. Loading header...";

  LitertLmStreamingLoader loader(data_stream);
  ABSL_RETURN_IF_ERROR(loader.LoadHeader());
  ABSL_LOG(INFO) << "Header loaded. Processing sections...";

  proto::LlmMetadata llm_metadata;
  bool set_llm_metadata = false;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<LlmExecutor> executor;
  std::unique_ptr<OwnedEnvironment> owned_env;
  std::unique_ptr<CompiledModel> compiled_main_model;
  std::unique_ptr<CompiledModel> compiled_drafter_model;

  auto streaming_model_resources =
      std::make_unique<StreamingWeightsModelResources>(llm_metadata);

  auto maybe_set_engine_settings = [&tokenizer, &set_llm_metadata,
                                    &engine_settings, &llm_metadata,
                                    &executor]() -> absl::Status {
    if (tokenizer != nullptr && set_llm_metadata) {
      ABSL_LOG(INFO) << "Setting engine settings...";
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          tokenizer.get(), &llm_metadata));

      if (executor != nullptr) {
        ABSL_RETURN_IF_ERROR(executor->UpdateExecutorSettings(
            engine_settings.GetMainExecutorSettings()));
      }
      ABSL_LOG(INFO) << "Engine settings set.";
    }
    return absl::OkStatus();
  };

  for (;;) {
    ABSL_ASSIGN_OR_RETURN(auto section, loader.GetNextSection());
    if (!section.has_value()) {
      ABSL_LOG(INFO) << "No more sections to process.";
      break;
    }

    const schema::SectionObject* section_metadata = section->section;
    switch (section_metadata->data_type()) {
      case schema::AnySectionDataType_NONE:
        ABSL_LOG(WARNING) << "Skipping section with no data type.";
        break;
      case schema::AnySectionDataType_GenericBinaryData:
        ABSL_LOG(WARNING)
            << "Skipping section with data type: GenericBinaryData";
        break;
      case schema::AnySectionDataType_Deprecated:
        ABSL_LOG(WARNING) << "Skipping section with data type: Deprecated";
        break;
      case schema::AnySectionDataType_LlmMetadataProto: {
        ABSL_LOG(INFO) << "Processing section data type: LlmMetadata";
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        if (!llm_metadata.ParseFromString(
                absl::string_view(buffer.data(), buffer.size()))) {
          return absl::InternalError("Failed to parse LlmMetadata");
        }

        set_llm_metadata = true;
        ABSL_LOG(INFO) << "LlmMetadataProto processed.";

        streaming_model_resources->SetLlmMetadata(llm_metadata);

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
      }
      case schema::AnySectionDataType_SP_Tokenizer: {
        ABSL_LOG(INFO) << "Processing section data type: SP_Tokenizer";
        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
              BenchmarkInfo::InitPhase::kTokenizer));
        }
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        ABSL_ASSIGN_OR_RETURN(
            tokenizer, SentencePieceTokenizer::CreateFromBuffer(
                           absl::string_view(buffer.data(), buffer.size())));
        ABSL_LOG(INFO) << "SentencePieceTokenizer created.";

        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
              BenchmarkInfo::InitPhase::kTokenizer));
        }

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
      }
      case schema::AnySectionDataType_TFLiteModel: {
        ABSL_LOG(INFO) << "Processing section data type: TFLiteModel";

        std::optional<ModelType> model_type = section->buffer_key.model_type;
        if (!model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteModel section.");
        }
        if (*model_type == ModelType::kUnknown) {
          return absl::UnimplementedError("kUnknown is not implemented");
        }

        ABSL_LOG(INFO) << "Caching model from stream for type: "
                       << static_cast<int>(*model_type);
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        streaming_model_resources->SetModelBuffer(*model_type,
                                                  std::move(buffer));
        if (*model_type == ModelType::kTfLitePrefillDecode &&
            owned_env == nullptr) {
          if (!set_llm_metadata) {
            return absl::InternalError(
                "LlmMetadata must be parsed before TFLiteModel.");
          }
          ABSL_ASSIGN_OR_RETURN(
              auto temp_owned_env,
              CreateEnvironment(engine_settings,
                                streaming_model_resources.get()));
          owned_env =
              std::make_unique<OwnedEnvironment>(std::move(temp_owned_env));
        }
        break;
      }
      case schema::AnySectionDataType_TFLiteWeights: {
        std::optional<ModelType> model_type = section->buffer_key.model_type;
        if (!model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteWeights section.");
        }
        if (*model_type == ModelType::kTfLitePerLayerEmbedder) {
          if (engine_settings.GetMainExecutorSettings().GetBackend() ==
              Backend::GPU) {
            ABSL_LOG(INFO) << "Storing TFLiteWeights section stream for "
                              "per-layer embedder (GPU)...";
            StoreWeightsStream(*model_type, std::move(section->data_stream));
          } else {
            ABSL_LOG(INFO)
                << "Caching per-layer embedder weights from stream (CPU)...";
            size_t size = section_metadata->end_offset() -
                          section_metadata->begin_offset();
            ABSL_RETURN_IF_ERROR(
                streaming_model_resources->SetPerLayerWeightsFromStream(
                    *section->data_stream, size));
          }
        } else {
          ABSL_LOG(INFO)
              << "Storing TFLiteWeights section stream for model type: "
              << static_cast<int>(*model_type);
          StoreWeightsStream(*model_type, std::move(section->data_stream));
          if (owned_env == nullptr) {
            return absl::InternalError(
                "Environment must be created before compilation.");
          }
          if (*model_type == ModelType::kTfLitePrefillDecode) {
            ABSL_LOG(INFO) << "Compiling main model from stream...";
            ABSL_ASSIGN_OR_RETURN(
                compiled_main_model,
                CompileModel(engine_settings.GetMainExecutorSettings(),
                             owned_env->env, *streaming_model_resources,
                             ModelType::kTfLitePrefillDecode));
            ABSL_LOG(INFO) << "Main model compiled.";
          } else if (*model_type == ModelType::kTfLiteMtpDrafter) {
            ABSL_LOG(INFO) << "Compiling drafter model from stream...";
            ABSL_ASSIGN_OR_RETURN(
                compiled_drafter_model,
                CompileModel(engine_settings.GetMainExecutorSettings(),
                             owned_env->env, *streaming_model_resources,
                             ModelType::kTfLiteMtpDrafter));
            ABSL_LOG(INFO) << "Drafter model compiled.";
          }
        }
        break;
      }
      case schema::AnySectionDataType_HF_Tokenizer_Zlib: {
        return absl::UnimplementedError(
            "Streaming HF_Tokenizer_Zlib section is not supported yet.");
      }
      case schema::AnySectionDataType_EmbeddingMetadataProto:
      case schema::AnySectionDataType_ExecutorMetadataProto:
        ABSL_LOG(WARNING) << "Skipping section with data type: "
                          << section_metadata->data_type();
        break;
    }
  }

  if (owned_env == nullptr) {
    return absl::InternalError(
        "Environment was not initialized during streaming.");
  }

  if (compiled_main_model == nullptr) {
    ABSL_LOG(INFO)
        << "Main model was not compiled during streaming. Compiling now...";
    ABSL_ASSIGN_OR_RETURN(
        compiled_main_model,
        CompileModel(engine_settings.GetMainExecutorSettings(), owned_env->env,
                     *streaming_model_resources,
                     ModelType::kTfLitePrefillDecode));
    ABSL_LOG(INFO) << "Main model compiled.";
  }

  const auto& advanced_settings =
      engine_settings.GetMainExecutorSettings().GetAdvancedSettings();
  if (advanced_settings.has_value() &&
      advanced_settings->enable_speculative_decoding &&
      compiled_drafter_model == nullptr) {
    ABSL_LOG(INFO)
        << "Drafter model was not compiled during streaming. Compiling now...";
    ABSL_ASSIGN_OR_RETURN(
        compiled_drafter_model,
        CompileModel(engine_settings.GetMainExecutorSettings(), owned_env->env,
                     *streaming_model_resources, ModelType::kTfLiteMtpDrafter));
    ABSL_LOG(INFO) << "Drafter model compiled.";
  }

  ABSL_ASSIGN_OR_RETURN(
      executor,
      CreateLlmLiteRtCompiledModelExecutor(
          engine_settings.GetMainExecutorSettings(), owned_env->env,
          std::move(compiled_main_model), streaming_model_resources.get(),
          /*embedding_lookup=*/nullptr,
          /*per_layer_embedding_lookup=*/nullptr,
          std::move(compiled_drafter_model)));

  if (tokenizer == nullptr) {
    return absl::InternalError("Failed to build tokenizer for streaming.");
  }

  std::unique_ptr<ExecutionManager> execution_manager;
  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_ptr;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    vision_executor_settings_ptr = std::make_unique<VisionExecutorSettings>(
        std::move(engine_settings.GetVisionExecutorSettings().value()));
  }

  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_ptr;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    audio_executor_settings_ptr = std::make_unique<AudioExecutorSettings>(
        std::move(engine_settings.GetAudioExecutorSettings().value()));
  }

  if (!engine_settings.GetSingleThreadedExecution()) {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        ThreadedExecutionManager::Create(
            tokenizer.get(), streaming_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), streaming_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  ABSL_RETURN_IF_ERROR(ClearStoredWeightsStreams());

  auto llm_impl = std::make_unique<EngineAdvancedImpl>(
      std::move(engine_settings), std::move(streaming_model_resources),
      std::move(owned_env), std::move(tokenizer), std::move(execution_manager),
      std::move(benchmark_info));
  return llm_impl;
}

LITERT_LM_REGISTER_ENGINE(
    EngineFactory::EngineType::kAdvancedLiteRTCompiledModel,
    [](EngineSettings settings, absl::string_view input_prompt_as_hint) {
      return EngineAdvancedImpl::Create(std::move(settings),
                                        input_prompt_as_hint);
    });

}  // namespace litert::lm

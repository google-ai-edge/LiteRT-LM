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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_streaming.h"
#include "runtime/core/audio_session_advanced.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/audio/audio_executor_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"
#include "runtime/executor/model_signature_utils.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/executor/vision/vision_executor_utils.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#ifdef __EMSCRIPTEN__
#include "runtime/util/file_data_stream.h"
#endif
#include "runtime/util/litert_lm_streaming_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/streamed_weights_manager.h"
#include "schema/core/litertlm_header_schema_generated.h"
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
#include "support/tokenizer/sentencepiece_tokenizer.h"
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
#include "support/tokenizer/tokenizer.h"

#if defined(LITERT_LM_DEBUGGER_ENABLED)
#include "runtime/util/runtime_debugger.h"
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

namespace litert::lm {
namespace {

std::optional<int> GetVisionTokensPerImageFromMetadata(
    const proto::LlmMetadata* llm_metadata) {
  if (llm_metadata == nullptr || !llm_metadata->has_llm_model_type()) {
    return std::nullopt;
  }
  const auto& model_type = llm_metadata->llm_model_type();
  int max_num_patches = 0;
  int pooling_kernel_size = 1;
  if (model_type.has_gemma4()) {
    max_num_patches = model_type.gemma4().max_num_patches();
    pooling_kernel_size = model_type.gemma4().pooling_kernel_size() > 0
                              ? model_type.gemma4().pooling_kernel_size()
                              : 3;
  } else if (model_type.has_generic_model()) {
    max_num_patches = model_type.generic_model().max_num_patches();
    if (model_type.generic_model().pooling_kernel_size() > 0) {
      pooling_kernel_size = model_type.generic_model().pooling_kernel_size();
    }
  } else if (model_type.has_lfm2()) {
    max_num_patches = model_type.lfm2().max_num_patches();
    pooling_kernel_size = model_type.lfm2().pooling_kernel_size() > 0
                              ? model_type.lfm2().pooling_kernel_size()
                              : 2;
  }
  if (max_num_patches <= 0) {
    return std::nullopt;
  }
  const int patch_num_shrink_factor = pooling_kernel_size * pooling_kernel_size;
  return max_num_patches / patch_num_shrink_factor;
}

}  // namespace

namespace {

// Compiles a LiteRt model from the given model type. Used for prefill/decode
// models and MTP drafter models.
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

  ABSL_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(
          executor_settings, activation_data_type,
          signatures ? std::make_optional(&*signatures) : std::nullopt,
          cache_suffix));

  // Set the MTP drafter compilation options. These would normally be set by
  // LlmLiteRtMtpDrafter::Create in regular non-streaming mode, but in streaming
  // mode, we compile the model ourselves, so we need to set the compilation
  // options here.
  if (model_type == ModelType::kTfLiteMtpDrafter) {
    ABSL_RETURN_IF_ERROR(
        UpdateCompilationOptions(executor_settings, compilation_options));
  }

  std::unique_ptr<CompiledModel> compiled_model;
  {
#ifdef __EMSCRIPTEN__
    SetCurrentlyCompilingModel(model_type);
    // Reset on failure too: the weight upload callback resolves its source
    // purely from this global, so a stale value would misdirect the next
    // submodel's upload.
    absl::Cleanup reset_compiling_model = [] {
      SetCurrentlyCompilingModel(ModelType::kUnknown);
    };
#endif
    LITERT_ASSIGN_OR_RETURN(
        auto compiled_model_tmp,
        CompiledModel::Create(lrt_env, model_buffer, compilation_options));
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  return compiled_model;
}

#ifdef __EMSCRIPTEN__
// Creates model assets that stream the model file at the path of
// `model_assets`.
absl::StatusOr<ModelAssets> CreateFileStreamModelAssets(
    const ModelAssets& model_assets) {
  ABSL_ASSIGN_OR_RETURN(auto path, model_assets.GetPath());
  ABSL_ASSIGN_OR_RETURN(auto file_stream,
                        FileDataStream::Create(std::string(path)));
  return ModelAssets::Create(std::move(file_stream));
}

// On the Web, GPU models with external weights are loaded through the streaming
// path, which uploads the weights directly into GPU buffers. If the main model
// is such a model and is given by path, replaces its model assets with a
// file-backed data stream so that the engine is created through the streaming
// path. Errors are logged and the original model assets are kept, so engine
// creation falls back to the regular path.
void MaybeUpgradeToStreamingModelAssets(EngineSettings& engine_settings) {
  auto& main_executor_settings =
      engine_settings.GetMutableMainExecutorSettings();
  const ModelAssets& model_assets = main_executor_settings.GetModelAssets();
  if (main_executor_settings.GetBackend() != Backend::GPU ||
      model_assets.HasDataStream()) {
    return;
  }
  absl::StatusOr<bool> has_external_weights =
      ModelHasExternalWeights(model_assets);
  if (!has_external_weights.ok() || !*has_external_weights) {
    return;
  }
  ABSL_LOG(INFO) << "External weights detected on GPU/WASM. "
                    "Upgrading to the streaming path.";
  absl::StatusOr<ModelAssets> streaming_model_assets =
      CreateFileStreamModelAssets(model_assets);
  if (!streaming_model_assets.ok()) {
    ABSL_LOG(ERROR) << "Failed to create streaming ModelAssets: "
                    << streaming_model_assets.status();
    return;
  }
  main_executor_settings.SetModelAssets(*std::move(streaming_model_assets));
}
#endif  // __EMSCRIPTEN__

}  // namespace

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
    // TODO(b/418794726): Move this logics to be part of the SessionConfig
    // class.
    ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));

    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "Model resources are not initialized.");
    }

    std::unique_ptr<SessionAdvanced> session;
    if (config.EnableAudioSessionAdvanced()) {
      ABSL_ASSIGN_OR_RETURN(
          session, AudioSessionAdvanced::Create(
                       execution_manager_, tokenizer_.get(), config,
                       std::move(session_benchmark_info), &living_sessions_,
                       /*engine=*/this));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          session, SessionAdvanced::Create(
                       execution_manager_, tokenizer_.get(), config,
                       std::move(session_benchmark_info), &living_sessions_,
                       /*engine=*/this));
    }

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

  absl::StatusOr<const support::Tokenizer*> GetTokenizer(
      ModelType model_type) const override {
    if (model_type == ModelType::kTfLitePrefillDecode) {
      if (!tokenizer_) {
        return absl::NotFoundError("Primary tokenizer not initialized.");
      }
      return tokenizer_.get();
    }
    if (!litert_model_resources_) {
      return absl::FailedPreconditionError(
          "Model resources are not initialized.");
    }
    return litert_model_resources_->GetOrCreateTokenizer(model_type);
  }

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

  absl::Status UpdateGpuEnableMetalResidencySet(
      bool enable_metal_residency_set) override {
    return execution_manager_->UpdateGpuEnableMetalResidencySet(
        enable_metal_residency_set);
  }

  absl::StatusOr<const ::litert::Environment*> GetEnvironment() const override {
    if (owned_env_ == nullptr) {
      return absl::NotFoundError("LiteRT environment is not available.");
    }
    return &owned_env_->env;
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
  MaybeUpgradeToStreamingModelAssets(engine_settings);
#endif  // __EMSCRIPTEN__

  // Model assets backed by a data stream (on any platform) are loaded section
  // by section through the streaming path.
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
  const bool is_npu =
      engine_settings.GetMainExecutorSettings().GetBackend() == Backend::NPU;
  // Magic-number replacement mutates the model flatbuffer in place.
  const bool enable_file_backed_model_loading =
      is_npu && advanced_settings &&
      !advanced_settings->configure_magic_numbers;

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
                            model_assets, enable_file_backed_model_loading,
                            /*enable_file_backed_for_aot_npu=*/is_npu));
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kModelAssets));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }

  ABSL_ASSIGN_OR_RETURN(const auto* llm_metadata,
                        model_resources->GetLlmMetadata());
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }

  // Select vision encoder and adapter signatures if max_vision_tokens_per_image
  // is set by user, or fallback to metadata from proto if it exists.
  std::optional<int> vision_token_limit;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    if (engine_settings.GetMaxVisionTokensPerImage().has_value()) {
      const int max_vision_tokens_per_image =
          *engine_settings.GetMaxVisionTokensPerImage();
      if (max_vision_tokens_per_image <= 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("max_vision_tokens_per_image must be positive, got: ",
                         max_vision_tokens_per_image));
      }
      if (GetVisionTokensPerImageFromMetadata(llm_metadata).has_value()) {
        vision_token_limit = max_vision_tokens_per_image;
      }
    } else {
      vision_token_limit = GetVisionTokensPerImageFromMetadata(llm_metadata);
      if (vision_token_limit.has_value() && *vision_token_limit > 0) {
        engine_settings.SetMaxVisionTokensPerImage(*vision_token_limit);
      }
    }
  }

  if (vision_token_limit.has_value() && *vision_token_limit > 0) {
    ABSL_ASSIGN_OR_RETURN(
        auto vision_sig_info,
        SelectVisionEncoderSignatures(*model_resources, *vision_token_limit));
    engine_settings.GetMutableVisionExecutorSettings()
        ->SetEncoderSelectedSignatures(vision_sig_info.signature_names);

    ABSL_ASSIGN_OR_RETURN(
        auto adapter_sig_info,
        SelectVisionAdapterSignatures(*model_resources, *vision_token_limit));
    if (adapter_sig_info.has_value()) {
      engine_settings.GetMutableVisionExecutorSettings()
          ->SetAdapterSelectedSignatures(adapter_sig_info->signature_names);
    }
  }
  bool hasLlmModelType =
      llm_metadata != nullptr && llm_metadata->has_llm_model_type();
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

  // Engine-scoped Debugger handle enabled via LITERT_LM_DEBUGGER_ENABLED=1.
  // Defaults to nullptr for zero-overhead in standard production Release
  // builds.
  std::shared_ptr<RuntimeDebugger> runtime_debugger = nullptr;
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  runtime_debugger =
      RuntimeDebugger::Create(main_executor_settings.GetCacheDir());
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

  switch (main_executor_settings.GetBackend()) {
    default: {
      ABSL_ASSIGN_OR_RETURN(executor, CreateLlmLiteRtCompiledModelExecutor(
                                          main_executor_settings,
                                          owned_env->env, *model_resources));
#if defined(LITERT_LM_DEBUGGER_ENABLED)
      if (main_executor_settings.GetBackend() == Backend::CPU ||
          main_executor_settings.GetBackend() == Backend::GPU) {
        if (auto* litert_executor =
                dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(
                    executor.get())) {
          if (runtime_debugger != nullptr) {
            litert_executor->UpdatePreGraphRunCallback(
                runtime_debugger->CreatePreGraphRunCallback());
            litert_executor->UpdatePostGraphRunCallback(
                runtime_debugger->CreatePostGraphRunCallback());
          }
        }
      }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
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
            std::move(audio_executor_settings_ptr), &owned_env->env,
            /*audio_executor=*/nullptr, runtime_debugger));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), model_resources.get(), std::move(executor),
            std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env,
            /*audio_executor=*/nullptr, runtime_debugger));
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
  // Locals are destroyed in reverse declaration order. The environment must
  // outlive everything created from it (e.g. the executor and compiled models,
  // whose GPU buffers are released through the environment), including on
  // early error returns, so it is declared first.
  std::unique_ptr<OwnedEnvironment> owned_env;
  auto streaming_model_resources = std::make_unique<ModelResourcesStreaming>();
  for (ModelType model_type :
       {ModelType::kTfLitePrefillDecode, ModelType::kTfLiteVisionEncoder,
        ModelType::kTfLiteAudioEncoderHw}) {
    if (auto bc = loader.GetTFLiteModelBackendConstraint(model_type);
        bc.has_value()) {
      streaming_model_resources->SetTFLiteModelBackendConstraint(
          model_type, std::move(*bc));
    }
    if (auto pat = loader.GetTFLiteModelPreferActivationType(model_type);
        pat.has_value()) {
      streaming_model_resources->SetTFLiteModelPreferActivationType(
          model_type, std::move(*pat));
    }
  }
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<LlmExecutor> executor;
  std::unique_ptr<CompiledModel> compiled_main_model;
  std::unique_ptr<CompiledModel> compiled_drafter_model;

  // Drop the registrations made below on every path out of this function.
  // Otherwise a failed load leaves the streams' cached bytes pinned until the
  // next engine creation clears them.
  absl::Cleanup clear_stored_weights = [] {
    ClearStoredWeightsStreams().IgnoreError();
  };

  // Whether `engine_settings` has been updated and validated with the model's
  // metadata. Models must not be compiled before this is true, since
  // compilation depends on the updated settings (e.g. max_num_tokens).
  bool engine_settings_updated = false;
  auto maybe_set_engine_settings =
      [&tokenizer, &set_llm_metadata, &engine_settings, &llm_metadata,
       &engine_settings_updated, &streaming_model_resources,
       input_prompt_as_hint]() -> absl::Status {
    if (!set_llm_metadata) {
      return absl::OkStatus();
    }
    if (tokenizer != nullptr) {
      ABSL_LOG(INFO) << "Setting engine settings...";
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          tokenizer.get(), &llm_metadata, input_prompt_as_hint,
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLitePrefillDecode),
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLiteVisionEncoder),
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLiteAudioEncoderHw),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLitePrefillDecode),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLiteVisionEncoder),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLiteAudioEncoderHw)));
      engine_settings_updated = true;
      ABSL_LOG(INFO) << "Engine settings set.";
    } else if (llm_metadata.has_llm_model_type() && !engine_settings_updated) {
      // Like the non-streaming path, model files with LlmModelType can update
      // the settings without the tokenizer. They are updated again with the
      // tokenizer once it is loaded, before the executor is created.
      ABSL_LOG(INFO) << "Setting engine settings without tokenizer...";
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          /*tokenizer=*/nullptr, &llm_metadata, input_prompt_as_hint,
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLitePrefillDecode),
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLiteVisionEncoder),
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              ModelType::kTfLiteAudioEncoderHw),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLitePrefillDecode),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLiteVisionEncoder),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              ModelType::kTfLiteAudioEncoderHw)));
      engine_settings_updated = true;
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
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
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
            tokenizer,
            ::litert::support::SentencePieceTokenizer::CreateFromBuffer(
                absl::string_view(buffer.data(), buffer.size())));
        ABSL_LOG(INFO) << "SentencePieceTokenizer created.";

        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
              BenchmarkInfo::InitPhase::kTokenizer));
        }

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
#else
        return absl::UnimplementedError(
            "SentencePiece tokenizer support is not enabled.");
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
      }
      case schema::AnySectionDataType_TFLiteModel: {
        ABSL_LOG(INFO) << "Processing section data type: TFLiteModel";

        if (!section->buffer_key.model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteModel section.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const ModelType model_type,
            StringToModelType(*section->buffer_key.model_type));
        if (model_type == ModelType::kUnknown) {
          return absl::UnimplementedError("kUnknown is not implemented");
        }
        if (section->backend_constraint.has_value()) {
          streaming_model_resources->SetTFLiteModelBackendConstraint(
              model_type, *section->backend_constraint);
        }
        if (section->prefer_activation_type.has_value()) {
          streaming_model_resources->SetTFLiteModelPreferActivationType(
              model_type, *section->prefer_activation_type);
        }

        ABSL_LOG(INFO) << "Caching model from stream for type: "
                       << static_cast<int>(model_type);
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        streaming_model_resources->SetModelBuffer(model_type,
                                                  std::move(buffer));
        if (model_type == ModelType::kTfLitePrefillDecode &&
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
        if (!section->buffer_key.model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteWeights section.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const ModelType model_type,
            StringToModelType(*section->buffer_key.model_type));
        if (model_type == ModelType::kTfLitePerLayerEmbedder ||
            model_type == ModelType::kTfLiteEmbedder) {
          ABSL_LOG(INFO)
              << "Caching embedder weights from stream (CPU) for model type: "
              << static_cast<int>(model_type);
          size_t size =
              section_metadata->end_offset() - section_metadata->begin_offset();
          ABSL_RETURN_IF_ERROR(streaming_model_resources->SetWeightsFromStream(
              model_type, *section->data_stream, size));
        } else {
          if (model_type == ModelType::kTfLiteMtpDrafter) {
            const auto& advanced_settings =
                engine_settings.GetMainExecutorSettings().GetAdvancedSettings();
            if (!advanced_settings.has_value() ||
                !advanced_settings->enable_speculative_decoding) {
              ABSL_LOG(INFO)
                  << "Speculative decoding is not enabled. Skipping weights "
                     "for MTP drafter.";
              break;
            }
          }
          ABSL_LOG(INFO)
              << "Storing TFLiteWeights section stream for model type: "
              << static_cast<int>(model_type);
          StoreWeightsStream(model_type, std::move(section->data_stream));
          if (owned_env == nullptr) {
            return absl::InternalError(
                "Environment must be created before compilation.");
          }
          if ((model_type == ModelType::kTfLitePrefillDecode ||
               model_type == ModelType::kTfLiteMtpDrafter) &&
              !engine_settings_updated) {
            // The weights stream is consumed by compilation, so the model must
            // be compiled now, and compilation depends on the updated settings.
            return absl::FailedPreconditionError(
                "Engine settings must be updated from LlmMetadata (and, for "
                "model files without LlmModelType, the tokenizer) before "
                "models with streamed weights can be compiled. Make sure "
                "these sections precede the TFLiteWeights sections.");
          }
          if (model_type == ModelType::kTfLitePrefillDecode) {
            ABSL_LOG(INFO) << "Compiling main model from stream...";
            ABSL_ASSIGN_OR_RETURN(
                compiled_main_model,
                CompileModel(engine_settings.GetMainExecutorSettings(),
                             owned_env->env, *streaming_model_resources,
                             ModelType::kTfLitePrefillDecode));
            ABSL_LOG(INFO) << "Main model compiled.";
          } else if (model_type == ModelType::kTfLiteMtpDrafter) {
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
      case schema::AnySectionDataType_ExecutorMetadataProto: {
        ABSL_LOG(INFO) << "Processing section data type: ExecutorMetadataProto";
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        proto::ExecutorMetadata executor_metadata;
        if (!executor_metadata.ParseFromString(
                absl::string_view(buffer.data(), buffer.size()))) {
          return absl::InternalError("Failed to parse ExecutorMetadata");
        }
        streaming_model_resources->SetExecutorMetadata(
            std::move(executor_metadata));
        ABSL_LOG(INFO) << "ExecutorMetadataProto processed.";
        break;
      }
      case schema::AnySectionDataType_EmbeddingMetadataProto:
      case schema::AnySectionDataType_TtsMetadataProto:
      case schema::AnySectionDataType_AsrMetadataProto:
      case schema::AnySectionDataType_Text2ImageMetadataProto:
        ABSL_LOG(WARNING) << "Skipping section with data type: "
                          << section_metadata->data_type();
        break;
    }
  }

  if (owned_env == nullptr) {
    return absl::InternalError(
        "Environment was not initialized during streaming.");
  }

  // The settings were updated with the tokenizer when it was loaded, so the
  // models compiled and the executor created below use the final settings.
  if (tokenizer == nullptr) {
    return absl::InternalError("Failed to build tokenizer for streaming.");
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

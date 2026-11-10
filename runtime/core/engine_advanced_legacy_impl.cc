// Copyright 2025 The ODML Authors.
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

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_data_provider.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_streaming.h"
#include "runtime/core/audio_session_advanced.h"
#include "runtime/core/engine_legacy_streaming_utils.h"
#include "runtime/core/engine_vision_utils.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/litert_lm_streaming_loader.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#if LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
#include "runtime/components/model_resources_litert_lm_google.h"
#include "runtime/util/file_format_util.h"
#endif  // LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
#include "support/tokenizer/huggingface_tokenizer.h"
#endif  // ENABLE_HUGGINGFACE_TOKENIZER

#if !defined(LITERTLM_CPU_ONLY)
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/executor/gpu_artisan/llm_gpu_artisan_executor.h"
#include "runtime/util/read_data_fn_util.h"
#endif  // !defined(LITERTLM_CPU_ONLY)

namespace litert::lm {

namespace {

namespace oi = ::odml::infra;

#if LITERT_LM_EXCLUDE_LEGACY_EXECUTORS
struct LegacyEngineModelResources {
  std::unique_ptr<ModelResources> litert_lm_model_resources;
};

absl::StatusOr<std::unique_ptr<LegacyEngineModelResources>>
BuildLegacyEngineModelResources(std::shared_ptr<ScopedFile> scoped_model_file) {
  ABSL_ASSIGN_OR_RETURN(FileFormat file_format,
                        GetFileFormat(/*model_path=*/"", scoped_model_file));
  if (file_format != FileFormat::LITERT_LM) {
    return absl::InvalidArgumentError(
        "Only .litertlm models are supported when legacy executors are "
        "excluded.");
  }
  ABSL_ASSIGN_OR_RETURN(auto loader, LitertLmLoader::Create(scoped_model_file));
  ABSL_ASSIGN_OR_RETURN(
      auto litert_lm_resources,
      ModelResourcesLitertLmGoogle::Create(std::move(loader)));
  auto model_resources = std::make_unique<LegacyEngineModelResources>();
  model_resources->litert_lm_model_resources = std::move(litert_lm_resources);
  return model_resources;
}
#endif  // LITERT_LM_EXCLUDE_LEGACY_EXECUTORS

class EngineAdvancedLegacyImpl : public Engine {
 public:
  ~EngineAdvancedLegacyImpl() override;

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  static absl::StatusOr<std::unique_ptr<Engine>> CreateStreamingWeights(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override;

  absl::Status WaitUntilDone(absl::Duration timeout) override;

  const EngineSettings& GetEngineSettings() const override;

  const Tokenizer& GetTokenizer() const override { return *tokenizer_; }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return execution_manager_->GetAudioExecutorProperties();
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return execution_manager_->GetVisionExecutorProperties();
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
  explicit EngineAdvancedLegacyImpl(
      EngineSettings engine_settings,
      std::unique_ptr<LegacyEngineModelResources> model_resources,
      std::unique_ptr<OwnedEnvironment> owned_env,
      std::unique_ptr<ExecutionManager> execution_manager,
      std::unique_ptr<Tokenizer> task_tokenizer,
      std::optional<BenchmarkInfo> benchmark_info);

  EngineSettings engine_settings_;
  std::unique_ptr<LegacyEngineModelResources> model_resources_;
  std::unique_ptr<OwnedEnvironment> owned_env_;
  std::shared_ptr<ExecutionManager> execution_manager_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::optional<BenchmarkInfo> benchmark_info_;

  // Counter for living sessions.
  std::atomic<int> living_sessions_{0};
};

absl::StatusOr<std::unique_ptr<LlmExecutor>> BuildExecutor(
    const LegacyEngineModelResources& model_resources,
    const EngineSettings& engine_settings) {
  // Create executor that creates and owns the interpreter and kv cache.
  std::unique_ptr<LlmExecutor> executor;
  ABSL_VLOG(1) << "Executor settings: "
               << engine_settings.GetMainExecutorSettings();

  if (engine_settings.GetMainExecutorSettings().GetBackend() == Backend::CPU) {
    return absl::InvalidArgumentError(
        "Legacy CPU executor is excluded in this build.");
#if !defined(LITERTLM_CPU_ONLY)
  } else if (engine_settings.GetMainExecutorSettings().GetBackend() ==
             Backend::GPU_ARTISAN) {
    if (model_resources.litert_lm_model_resources == nullptr) {
      return absl::InternalError(
          "Failed to build GPU_ARTISAN executor: "
          "model_resources.litert_lm_model_resources is null. ");
    }
    ABSL_ASSIGN_OR_RETURN(executor,
                          oi::LlmGpuArtisanExecutor::Create(
                              engine_settings.GetMainExecutorSettings(),
                              *model_resources.litert_lm_model_resources));
#endif  // !defined(LITERTLM_CPU_ONLY)
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend: ",
                     engine_settings.GetMainExecutorSettings().GetBackend()));
  }

  return std::move(executor);
}

RuntimeConfig MakeDefaultRuntimeConfig() {
  RuntimeConfig runtime_config;
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::GREEDY);
  sampler_params.set_k(1);
  sampler_params.set_temperature(0.0f);
  runtime_config.sampler_params = sampler_params;
  runtime_config.tokens_per_decode = 1;
  runtime_config.output_heads = 1;
  return runtime_config;
}

}  // namespace

EngineAdvancedLegacyImpl::~EngineAdvancedLegacyImpl() {
  auto status = WaitUntilDone(Engine::kDefaultTimeout);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to wait for engine to finish: " << status;
  }

  if (living_sessions_ > 0) {
    ABSL_LOG(ERROR) << "EngineAdvancedLegacyImpl destructed with "
                    << living_sessions_ << " living sessions!";
  }

  execution_manager_.reset();
  tokenizer_.reset();
  owned_env_.reset();
  model_resources_.reset();
}

EngineAdvancedLegacyImpl::EngineAdvancedLegacyImpl(
    EngineSettings engine_settings,
    std::unique_ptr<LegacyEngineModelResources> model_resources,
    std::unique_ptr<OwnedEnvironment> owned_env,
    std::unique_ptr<ExecutionManager> execution_manager,
    std::unique_ptr<Tokenizer> tokenizer,
    std::optional<BenchmarkInfo> benchmark_info)
    : engine_settings_(std::move(engine_settings)),
      model_resources_(std::move(model_resources)),
      owned_env_(std::move(owned_env)),
      execution_manager_(std::move(execution_manager)),
      tokenizer_(std::move(tokenizer)),
      benchmark_info_(std::move(benchmark_info)) {}

// Method to create the Session.
absl::StatusOr<std::unique_ptr<Engine::Session>>
EngineAdvancedLegacyImpl::CreateSession(const SessionConfig& session_config) {
  auto config = session_config;
  ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));
  if (config.EnableAudioSessionAdvanced()) {
    return AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(),
                                        config, benchmark_info_,
                                        &living_sessions_, /*engine=*/this);
  }
  return SessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                 benchmark_info_, &living_sessions_,
                                 /*engine=*/this);
}

absl::Status EngineAdvancedLegacyImpl::WaitUntilDone(absl::Duration timeout) {
  return execution_manager_->WaitUntilAllDone(timeout);
}

const EngineSettings& EngineAdvancedLegacyImpl::GetEngineSettings() const {
  return engine_settings_;
}

absl::StatusOr<std::unique_ptr<Engine>>
EngineAdvancedLegacyImpl::CreateStreamingWeights(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  ABSL_VLOG(1) << "Constructing legacy EngineImpl from a weight stream...";

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
  ABSL_VLOG(1) << "Got data stream. Loading header...";

  LitertLmStreamingLoader loader(data_stream);
  ABSL_RETURN_IF_ERROR(loader.LoadHeader());
  ABSL_VLOG(1) << "Header loaded. Processing sections...";

  proto::LlmMetadata llm_metadata;
  bool set_llm_metadata = false;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<LlmExecutor> executor;
#if !defined(LITERTLM_CPU_ONLY)
  // Set if `executor` is an artisan executor, which needs the end-of-vision
  // embedding for vision inputs. The end-of-vision model may be streamed after
  // the executor is created, so the embedding is set after all sections are
  // processed.
  oi::LlmGpuArtisanExecutor* artisan_executor = nullptr;
  int artisan_embedding_dimension = 0;
#endif  // !defined(LITERTLM_CPU_ONLY)

  // Holds the vision models read from the stream. It's owned by the engine
  // (through `ExecutorModelResources` below), since the vision executor
  // references it.
  auto streaming_model_resources = std::make_unique<ModelResourcesStreaming>();
  for (const proto::LlmMetadata::TfLiteModelType model_type :
       {proto::LlmMetadata::TF_LITE_VISION_ENCODER,
        proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW}) {
    if (auto backend_constraint =
            loader.GetTFLiteModelBackendConstraint(model_type);
        backend_constraint.has_value()) {
      streaming_model_resources->SetTFLiteModelBackendConstraint(
          model_type, std::move(*backend_constraint));
    }
    if (auto prefer_activation_type =
            loader.GetTFLiteModelPreferActivationType(model_type);
        prefer_activation_type.has_value()) {
      streaming_model_resources->SetTFLiteModelPreferActivationType(
          model_type, std::move(*prefer_activation_type));
    }
  }

  auto maybe_set_engine_settings =
      [&tokenizer, &set_llm_metadata, &engine_settings, &llm_metadata,
       &executor, &streaming_model_resources]() -> absl::Status {
    if (tokenizer != nullptr && set_llm_metadata) {
      ABSL_VLOG(1) << "Setting engine settings...";
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          tokenizer.get(), &llm_metadata, /*input_prompt_as_hint=*/"",
          /*text_backend_constraint=*/std::nullopt,
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          streaming_model_resources->GetTFLiteModelBackendConstraint(
              proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW),
          /*text_prefer_activation_type=*/std::nullopt,
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          streaming_model_resources->GetTFLiteModelPreferActivationType(
              proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)));

      if (executor != nullptr) {
        ABSL_RETURN_IF_ERROR(executor->UpdateExecutorSettings(
            engine_settings.GetMainExecutorSettings()));
      }
      ABSL_VLOG(1) << "Engine settings set.";
    }
    return absl::OkStatus();
  };

  for (;;) {
    ABSL_ASSIGN_OR_RETURN(auto section, loader.GetNextSection());
    if (!section.has_value()) {
      ABSL_VLOG(1) << "No more sections to process.";
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
        ABSL_VLOG(1) << "Processing section data type: LlmMetadata";
        // LlmMetadataProto is small enough that we can read it into a buffer,
        // turn it into a string view, and pass it to
        // ExtractOrConvertLlmMetadata.
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        // TODO: b/477709280 - Can we use ExtractOrConvertLlmMetadata instead?
        if (!llm_metadata.ParseFromString(
                absl::string_view(buffer.data(), buffer.size()))) {  // NOLINT
          return absl::InternalError("Failed to parse LlmMetadata");
        }

        set_llm_metadata = true;
        ABSL_VLOG(1) << "LlmMetadataProto processed.";

        streaming_model_resources->SetLlmMetadata(llm_metadata);

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
      }
      case schema::AnySectionDataType_SP_Tokenizer: {
        ABSL_VLOG(1) << "Processing section data type: SP_Tokenizer";
        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
              BenchmarkInfo::InitPhase::kTokenizer));
        }
        // SP_Tokenizer is small enough that we can read it into a buffer and
        // turn it into a Tokenizer.
        std::vector<char> buffer(section_metadata->end_offset() -
                                 section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            buffer.data(), 0, buffer.size()));
        ABSL_ASSIGN_OR_RETURN(
            tokenizer, SentencePieceTokenizer::CreateFromBuffer(
                           absl::string_view(buffer.data(), buffer.size())));
        ABSL_VLOG(1) << "SentencePieceTokenizer created.";

        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
              BenchmarkInfo::InitPhase::kTokenizer));
        }

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
      }
      case schema::AnySectionDataType_TFLiteModel: {
        ABSL_VLOG(1) << "Processing section data type: TFLiteModel";

        if (!section->buffer_key.model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteModel section.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const proto::LlmMetadata::TfLiteModelType model_type,
            ParseLlmTfLiteModelType(*section->buffer_key.model_type));
        switch (model_type) {
          case proto::LlmMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED:
            return absl::InvalidArgumentError(
                "Unknown model type in TFLiteModel section.");
          case proto::LlmMetadata::TF_LITE_PREFILL_DECODE:
            return absl::UnimplementedError(
                "Streaming kTfLitePrefillDecode models is not supported yet.");
          case proto::LlmMetadata::TF_LITE_EMBEDDER:
            return absl::UnimplementedError(
                "Streaming kTfLiteEmbedder models is not supported yet.");
          case proto::LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER:
            return absl::UnimplementedError(
                "Streaming kTfLitePerLayerEmbedder models is not supported "
                "yet.");
          case proto::LlmMetadata::TF_LITE_AUX:
            return absl::UnimplementedError(
                "Streaming kTfLiteAux models is not supported yet.");
          case proto::LlmMetadata::TF_LITE_AUDIO_FRONTEND:
            return absl::UnimplementedError(
                "Streaming kTfLiteAudioFrontend models is not supported yet.");
          case proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW:
          case proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER:
          case proto::LlmMetadata::TF_LITE_END_OF_AUDIO:
          case proto::LlmMetadata::TF_LITE_VISION_ENCODER:
          case proto::LlmMetadata::TF_LITE_VISION_ADAPTER:
          case proto::LlmMetadata::TF_LITE_END_OF_VISION: {
            const size_t size = section_metadata->end_offset() -
                                section_metadata->begin_offset();
            if (!ShouldCacheStreamedModel(model_type, engine_settings)) {
              ABSL_VLOG(1) << "Modality is not enabled. Skipping model type: "
                           << static_cast<int>(model_type);
              ABSL_RETURN_IF_ERROR(section->data_stream->Discard(0, size));
              break;
            }
            if (section->backend_constraint.has_value()) {
              streaming_model_resources->SetTFLiteModelBackendConstraint(
                  model_type, *section->backend_constraint);
            }
            if (section->prefer_activation_type.has_value()) {
              streaming_model_resources->SetTFLiteModelPreferActivationType(
                  model_type, *section->prefer_activation_type);
            }
            // The vision and audio executors (and the artisan executor's
            // end-of-vision and end-of-audio embeddings) are created from the
            // cached models after all sections are processed.
            ABSL_LOG(INFO) << "Caching model (" << size
                           << " bytes) from stream for type: "
                           << static_cast<int>(model_type);
            std::vector<char> buffer(size);
            ABSL_RETURN_IF_ERROR(
                section->data_stream->ReadAndDiscard(buffer.data(), 0, size));
            streaming_model_resources->SetModelBuffer(model_type,
                                                      std::move(buffer));
            break;
          }
          case proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER: {
#if !defined(LITERTLM_CPU_ONLY)
            ABSL_VLOG(1) << "Loading ArtisanTextDecoder model.";

            auto read_data_fn =
                MakeReadDataFn(std::move(section->data_stream),
                               section_metadata->end_offset() -
                                   section_metadata->begin_offset());
            ABSL_ASSIGN_OR_RETURN(
                auto model_data,
                oi::llm_utils::ModelData::Create(std::move(read_data_fn)));
            artisan_embedding_dimension = model_data->GetLlmParameters()
                                              .transformer_parameters()
                                              .embedding_dim();
            // The media embedding cache is allocated when the executor is
            // created, so reserve room for embeddings from a LiteRT audio
            // encoder now, even though its section may be streamed later.
            const bool has_tflite_audio_encoder = loader.HasSection(
                BufferKey(schema::AnySectionDataType_TFLiteModel,
                          proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW));
            const int max_num_audio_tokens = GetNumExternalAudioTokensToReserve(
                has_tflite_audio_encoder,
                /*artisan_has_audio_encoder=*/
                model_data->GetLlmParameters().has_audio_encoder_parameters(),
                engine_settings,
                oi::LlmGpuArtisanExecutor::kDefaultMaxNumExternalAudioTokens);
            ABSL_ASSIGN_OR_RETURN(
                auto created_artisan_executor,
                oi::LlmGpuArtisanExecutor::Create(
                    engine_settings.GetMainExecutorSettings(), model_data,
                    /*end_of_vision_embedding=*/{},
                    /*end_of_audio_embedding=*/{}, max_num_audio_tokens));
            artisan_executor = created_artisan_executor.get();
            executor = std::move(created_artisan_executor);
            break;
#else
            return absl::UnimplementedError(
                "ArtisanTextDecoder models are not supported on CPU yet.");
#endif
          }
          case proto::LlmMetadata::TF_LITE_MTP_DRAFTER:
            return absl::UnimplementedError(
                "Streaming kTfLiteMtpDrafter models is not supported yet.");
          case proto::LlmMetadata::TF_LITE_MTP_AUX:
            return absl::UnimplementedError(
                "Streaming kTfLiteMtpAux models is not supported yet.");
          default:
            return absl::InvalidArgumentError(
                "Unsupported model type for streaming.");
        }
        break;
      }
      case schema::AnySectionDataType_TFLiteWeights: {
        if (!section->buffer_key.model_type.has_value()) {
          return absl::InvalidArgumentError(
              "Model type is not set for TFLiteWeights section.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const proto::LlmMetadata::TfLiteModelType model_type,
            ParseLlmTfLiteModelType(*section->buffer_key.model_type));
        // Streaming external weights is only supported for vision and audio
        // encoders and adapters.
        if (model_type != proto::LlmMetadata::TF_LITE_VISION_ENCODER &&
            model_type != proto::LlmMetadata::TF_LITE_VISION_ADAPTER &&
            model_type != proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW &&
            model_type != proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER) {
          return absl::UnimplementedError(absl::StrCat(
              "Streaming TFLiteWeights section is not supported yet for model "
              "type: ",
              static_cast<int>(model_type)));
        }
        const size_t size =
            section_metadata->end_offset() - section_metadata->begin_offset();
        if (!ShouldCacheStreamedModel(model_type, engine_settings)) {
          ABSL_VLOG(1) << "Modality is not enabled. Skipping weights for model "
                          "type: "
                       << static_cast<int>(model_type);
          ABSL_RETURN_IF_ERROR(section->data_stream->Discard(0, size));
          break;
        }
        // Vision and audio weights are stored in host memory until the vision
        // and audio executors are created after all sections are processed.
        ABSL_LOG(INFO) << "Reading weights (" << size
                       << " bytes) for model type "
                       << static_cast<int>(model_type)
                       << " into host memory...";
        ABSL_RETURN_IF_ERROR(streaming_model_resources->SetWeightsFromStream(
            model_type, *section->data_stream, size));
        break;
      }
      case schema::AnySectionDataType_HF_Tokenizer_Zlib: {
#ifdef ENABLE_HUGGINGFACE_TOKENIZER
        ABSL_VLOG(1) << "Processing section data type: HF_Tokenizer_Zlib";
        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
              BenchmarkInfo::InitPhase::kTokenizer));
        }
        std::vector<uint8_t> compressed_buffer(
            section_metadata->end_offset() - section_metadata->begin_offset());
        ABSL_RETURN_IF_ERROR(section->data_stream->ReadAndDiscard(
            reinterpret_cast<char*>(compressed_buffer.data()), 0,
            compressed_buffer.size()));

        std::vector<uint8_t> decompressed_buffer;
        ABSL_RETURN_IF_ERROR(schema::DecompressData(compressed_buffer.data(),
                                                    compressed_buffer.size(),
                                                    &decompressed_buffer));

        std::string json_data(
            reinterpret_cast<const char*>(decompressed_buffer.data()),
            decompressed_buffer.size());
        ABSL_ASSIGN_OR_RETURN(
            tokenizer,
            ::litert::support::HuggingFaceTokenizer::CreateFromJson(json_data));
        ABSL_VLOG(1) << "HuggingFaceTokenizer created.";

        if (engine_settings.IsBenchmarkEnabled()) {
          ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
              BenchmarkInfo::InitPhase::kTokenizer));
        }

        ABSL_RETURN_IF_ERROR(maybe_set_engine_settings());

        break;
#else
        ABSL_RETURN_IF_ERROR(section->data_stream->Discard(
            0,
            section_metadata->end_offset() - section_metadata->begin_offset()));
        return absl::UnimplementedError(
            "HuggingFace tokenizer found, but LiteRT LM was built with "
            "--define=DISABLE_HUGGINGFACE_TOKENIZER=1.");
#endif  // ENABLE_HUGGINGFACE_TOKENIZER
      }
      case schema::AnySectionDataType_ExecutorMetadataProto:
        return absl::UnimplementedError(
            "Streaming LlmExecutorMetadata section is not supported yet.");
      case schema::AnySectionDataType_EmbeddingMetadataProto:
        return absl::UnimplementedError(
            "Streaming EmbeddingMetadataProto section is not supported yet.");
      case schema::AnySectionDataType_Text2SpeechMetadataProto:
        return absl::UnimplementedError(
            "Streaming Text2SpeechMetadataProto section is not supported yet.");
      case schema::AnySectionDataType_Speech2TextMetadataProto:
        return absl::UnimplementedError(
            "Streaming Speech2TextMetadataProto section is not supported yet.");
      case schema::AnySectionDataType_Text2ImageMetadataProto:
        return absl::UnimplementedError(
            "Streaming Text2ImageMetadataProto section is not supported yet.");
      case schema::AnySectionDataType_Any2AnyMetadataProto:
        return absl::UnimplementedError(
            "Streaming Any2AnyMetadataProto section is not supported yet.");
    }
  }

  if (executor == nullptr) {
    return absl::InternalError("Failed to build executor for streaming.");
  }
  ABSL_RETURN_IF_ERROR(
      executor->UpdateRuntimeConfig(MakeDefaultRuntimeConfig()));
  std::unique_ptr<OwnedEnvironment> owned_env;
  {
    ABSL_ASSIGN_OR_RETURN(auto temp_owned_env,
                          CreateEnvironment(engine_settings, nullptr));
    owned_env = std::make_unique<OwnedEnvironment>(std::move(temp_owned_env));
  }

  if (tokenizer == nullptr) {
    return absl::InternalError("Failed to build tokenizer for streaming.");
  }

  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_ptr;
  std::unique_ptr<VisionExecutor> vision_executor;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    if (streaming_model_resources
            ->GetTFLiteModel(proto::LlmMetadata::TF_LITE_VISION_ENCODER)
            .ok()) {
      ABSL_RETURN_IF_ERROR(MaybeSelectVisionSignatures(
          engine_settings, *streaming_model_resources,
          set_llm_metadata ? &llm_metadata : nullptr));
      vision_executor_settings_ptr = std::make_unique<VisionExecutorSettings>(
          std::move(engine_settings.GetVisionExecutorSettings().value()));
      if (vision_executor_settings_ptr->GetAdapterBackend() != Backend::CPU) {
        ABSL_LOG(WARNING) << "Vision adapter backend is not CPU, which may "
                             "cause precision loss.";
      }

      // The vision executor must be created now, while the vision encoder
      // weights are still in host memory. It can not be lazily created later
      // since the model can not be reloaded from the (already consumed) stream.
      //
      // The encoder and adapter keep a reference to the settings they are
      // created with, so use the heap-allocated settings that the
      // ResourceManager takes ownership of below, which outlive the executor.
      ABSL_LOG(INFO) << "Creating vision executor from stream...";
      ABSL_ASSIGN_OR_RETURN(vision_executor,
                            VisionLiteRtCompiledModelExecutor::Create(
                                *vision_executor_settings_ptr, owned_env->env,
                                *streaming_model_resources));
      ABSL_LOG(INFO) << "Vision executor created.";

      // We only need to keep the encoder weights in host memory when running
      // on CPU. Don't release the vision adapter weights, as it always runs on
      // CPU.
      if (vision_executor_settings_ptr->GetBackend() != Backend::CPU) {
        streaming_model_resources->ReleaseWeights(
            proto::LlmMetadata::TF_LITE_VISION_ENCODER);
        ABSL_LOG(INFO) << "Released host memory for vision encoder weights.";
      }

#if !defined(LITERTLM_CPU_ONLY)
      if (artisan_executor != nullptr &&
          engine_settings.GetMainExecutorSettings().GetMaxNumImages() == 0) {
        ABSL_LOG(WARNING)
            << "Vision is enabled, but max_num_images is 0, so artisan models "
               "can't take image inputs. Set max_num_images in the main "
               "executor settings.";
      }

      // Like when creating the artisan executor from (non-streamed) model
      // resources, set the end-of-vision embedding from the end-of-vision
      // model, if any.
      if (auto end_of_vision_model = streaming_model_resources->GetTFLiteModel(
              proto::LlmMetadata::TF_LITE_END_OF_VISION);
          artisan_executor != nullptr && end_of_vision_model.ok()) {
        ABSL_ASSIGN_OR_RETURN(
            std::vector<float> end_of_vision_embedding,
            oi::LlmGpuArtisanExecutor::ComputeEndOfVisionEmbedding(
                *end_of_vision_model, artisan_embedding_dimension));
        artisan_executor->SetEndOfVisionEmbedding(
            std::move(end_of_vision_embedding));
        ABSL_LOG(INFO) << "Set end-of-vision embedding.";
      }
#endif  // !defined(LITERTLM_CPU_ONLY)
    } else {
      ABSL_LOG(WARNING) << "Vision executor settings are set, but the model "
                           "has no vision encoder. Vision is disabled.";
      // Clear the settings so the engine doesn't report vision support.
      // Otherwise sessions that enable the vision modality (which the Web SDK
      // does by default when vision settings are present) fail to load the
      // vision executor.
      engine_settings.GetMutableVisionExecutorSettings().reset();
    }
  }

  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_ptr;
  std::unique_ptr<AudioExecutor> audio_executor;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    if (streaming_model_resources
            ->GetTFLiteModel(proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)
            .ok()) {
      audio_executor_settings_ptr = std::make_unique<AudioExecutorSettings>(
          std::move(engine_settings.GetAudioExecutorSettings().value()));

      // Create the audio executor now, while the audio encoder weights are
      // still in host memory. It can not be lazily created later since the
      // model can not be reloaded from the (already consumed) stream.
      ABSL_LOG(INFO) << "Creating audio executor from stream...";
      ABSL_ASSIGN_OR_RETURN(audio_executor,
                            AudioLiteRtCompiledModelExecutor::Create(
                                *audio_executor_settings_ptr, owned_env->env,
                                *streaming_model_resources));
      ABSL_LOG(INFO) << "Audio executor created.";

      // Only keep the encoder weights in host memory when running on CPU.
      if (audio_executor_settings_ptr->GetBackend() != Backend::CPU) {
        streaming_model_resources->ReleaseWeights(
            proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW);
        ABSL_LOG(INFO) << "Released host memory for audio encoder weights.";
      }

#if !defined(LITERTLM_CPU_ONLY)
      // Like when creating the artisan executor from (non-streamed) model
      // resources, set the end-of-audio embedding from the end-of-audio model,
      // if any.
      if (artisan_executor != nullptr) {
        absl::StatusOr<const litert::Model*> end_of_audio_model =
            streaming_model_resources->GetTFLiteModel(
                proto::LlmMetadata::TF_LITE_END_OF_AUDIO);
        if (end_of_audio_model.ok()) {
          ABSL_ASSIGN_OR_RETURN(
              std::vector<float> end_of_audio_embedding,
              oi::LlmGpuArtisanExecutor::ComputeEndOfAudioEmbedding(
                  *end_of_audio_model, artisan_embedding_dimension));
          artisan_executor->SetEndOfAudioEmbedding(
              std::move(end_of_audio_embedding));
          ABSL_LOG(INFO) << "Set end-of-audio embedding.";
        }
      }
#endif  // !defined(LITERTLM_CPU_ONLY)
    } else {
      // Audio encoders that are part of the artisan model's weights are not
      // supported yet when streaming legacy models. Clear the settings so the
      // engine doesn't report audio support (see the vision case above).
      ABSL_LOG(WARNING) << "Audio executor settings are set, but the model has "
                           "no LiteRT audio encoder. Audio is disabled.";
      engine_settings.GetMutableAudioExecutorSettings().reset();
    }
  }

  ABSL_VLOG(1) << "Creating execution manager for streaming model.";

  std::unique_ptr<ExecutionManager> execution_manager;
  if (!engine_settings.GetSingleThreadedExecution()) {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        ThreadedExecutionManager::Create(
            tokenizer.get(), streaming_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env,
            std::move(audio_executor), /*runtime_debugger=*/nullptr,
            std::move(vision_executor)));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), streaming_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env,
            std::move(audio_executor), /*runtime_debugger=*/nullptr,
            std::move(vision_executor)));
  }

  // The engine owns the streaming model resources, since the execution manager
  // and the vision and audio executors reference them.
  auto model_resources = std::make_unique<LegacyEngineModelResources>();
  model_resources->litert_lm_model_resources =
      std::move(streaming_model_resources);

  auto llm_impl = absl::WrapUnique(new EngineAdvancedLegacyImpl(
      std::move(engine_settings), std::move(model_resources),
      std::move(owned_env), std::move(execution_manager), std::move(tokenizer),
      std::move(benchmark_info)));
  return llm_impl;
}

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> EngineAdvancedLegacyImpl::Create(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  ABSL_VLOG(1) << "Constructing legacy EngineImpl...";
  if (engine_settings.GetVisionExecutorSettings().has_value() &&
      engine_settings.GetMaxVisionTokensPerImage().has_value()) {
    if (*engine_settings.GetMaxVisionTokensPerImage() <= 0) {
      return absl::InvalidArgumentError(
          "max_vision_tokens_per_image must be positive.");
    }
  }
  std::optional<BenchmarkInfo> benchmark_info;
  if (engine_settings.GetMainExecutorSettings()
          .GetModelAssets()
          .HasDataStream()) {
    return CreateStreamingWeights(engine_settings, input_prompt_as_hint);
  }

  if (engine_settings.IsBenchmarkEnabled()) {
    benchmark_info = std::make_optional<BenchmarkInfo>(
        engine_settings.GetBenchmarkParams().value());
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  }
  ABSL_ASSIGN_OR_RETURN(auto scoped_model_file,
                        engine_settings.GetMainExecutorSettings()
                            .GetModelAssets()
                            .GetOrCreateScopedFile());
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        BuildLegacyEngineModelResources(scoped_model_file));

  // Get llm metadata first
  proto::LlmMetadata llm_metadata;
  if (model_resources->litert_lm_model_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        auto metadata,
        model_resources->litert_lm_model_resources->GetLlmMetadata());
    llm_metadata = *metadata;
  } else {
    return absl::InternalError("Model resources are not initialized.");
  }

  bool hasLlmModelType = llm_metadata.has_llm_model_type();
  absl::Duration tokenizer_duration = absl::ZeroDuration();
  // This lambda is used to create the tokenizer asynchronously if the model
  // type is available, such that the tokenizer can be created in parallel with
  // the executor.
  auto create_tokenizer = [&]() -> absl::StatusOr<std::unique_ptr<Tokenizer>> {
    if (model_resources->litert_lm_model_resources != nullptr) {
      // Handle the .litert_lm file format.
      absl::Time start_time = absl::Now();
      ABSL_ASSIGN_OR_RETURN(
          auto litert_tokenizer,
          model_resources->litert_lm_model_resources->GetTokenizer());
      tokenizer_duration = absl::Now() - start_time;
      return litert_tokenizer;
    }
    return absl::InternalError("Model resources are not initialized.");
  };
  std::future<absl::StatusOr<std::unique_ptr<Tokenizer>>> tokenizer_future;
  std::unique_ptr<Tokenizer> tokenizer;
  if (!hasLlmModelType) {
    ABSL_VLOG(1)
        << "Legacy model files don't have LlmModelType, loading tokenizer now";
    // Update and load the parameters from the model file and convert the tokens
    // to ids.
    ABSL_ASSIGN_OR_RETURN(tokenizer, create_tokenizer());
    if (model_resources->litert_lm_model_resources != nullptr) {
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          tokenizer.get(), &llm_metadata, input_prompt_as_hint,
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)));
    } else {
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          tokenizer.get(), &llm_metadata, input_prompt_as_hint));
    }
  } else {
    // If the model type is available, wait for the tokenizer to be created
    // after the model is loaded.
    ABSL_VLOG(1) << "New model files have LlmModelType, loading tokenizer "
                    "asynchronously";
    if (engine_settings.GetParallelFileSectionLoading()) {
      tokenizer_future = std::async(std::launch::async, create_tokenizer);
    } else {
      tokenizer_future = std::async(std::launch::deferred, create_tokenizer);
    }
    if (model_resources->litert_lm_model_resources != nullptr) {
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          nullptr, &llm_metadata, input_prompt_as_hint,
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelBackendConstraint(
                  proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_PREFILL_DECODE),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_VISION_ENCODER),
          model_resources->litert_lm_model_resources
              ->GetTFLiteModelPreferActivationType(
                  proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)));
    } else {
      ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
          nullptr, &llm_metadata, input_prompt_as_hint));
    }
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }

  ABSL_ASSIGN_OR_RETURN(auto executor,
                        BuildExecutor(*model_resources, engine_settings));

  std::unique_ptr<OwnedEnvironment> owned_env;
  {
    ABSL_ASSIGN_OR_RETURN(
        auto temp_owned_env,
        CreateEnvironment(engine_settings,
                          model_resources->litert_lm_model_resources.get()));
    owned_env = std::make_unique<OwnedEnvironment>(std::move(temp_owned_env));
  }

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
        tokenizer.get(), &llm_metadata, input_prompt_as_hint));
    // As we load the tokenizer asynchronously, we need to update the executor
    // settings after the tokenizer is loaded.
    ABSL_RETURN_IF_ERROR(executor->UpdateExecutorSettings(
        engine_settings.GetMainExecutorSettings()));
  }
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kTokenizer, tokenizer_duration));
  }
  ABSL_RETURN_IF_ERROR(
      executor->UpdateRuntimeConfig(MakeDefaultRuntimeConfig()));

  std::unique_ptr<ExecutionManager> execution_manager;
  if (!engine_settings.GetSingleThreadedExecution()) {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        ThreadedExecutionManager::Create(
            tokenizer.get(), model_resources->litert_lm_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), model_resources->litert_lm_model_resources.get(),
            std::move(executor), std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  auto llm_impl = absl::WrapUnique(new EngineAdvancedLegacyImpl(
      std::move(engine_settings), std::move(model_resources),
      std::move(owned_env), std::move(execution_manager), std::move(tokenizer),
      std::move(benchmark_info)));
  return llm_impl;
};

LITERT_LM_REGISTER_ENGINE(EngineFactory::EngineType::kAdvancedLegacyTfLite,
                          [](EngineSettings settings,
                             absl::string_view input_prompt_as_hint) {
                            return EngineAdvancedLegacyImpl::Create(
                                std::move(settings), input_prompt_as_hint);
                          });
}  // namespace litert::lm

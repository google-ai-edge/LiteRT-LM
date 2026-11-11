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

#include "runtime/core/engine_init_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources_streaming.h"
#include "runtime/core/engine_vision_utils.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/executor/vision_litert_compiled_model_executor.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_streaming_loader.h"
#include "schema/core/litertlm_read.h"
#include "support/tokenizer/huggingface_tokenizer.h"
#if defined(ENABLE_SENTENCEPIECE_TOKENIZER) && \
    !defined(LITERT_LM_SENTENCEPIECE_DISABLED)
#include "support/tokenizer/sentencepiece_tokenizer.h"
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER && !LITERT_LM_SENTENCEPIECE_DISABLED
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

void CopyStreamingModelHints(
    LitertLmStreamingLoader& loader,
    absl::Span<const proto::LlmMetadata::TfLiteModelType> model_types,
    ModelResourcesStreaming& resources) {
  for (const proto::LlmMetadata::TfLiteModelType model_type : model_types) {
    if (std::optional<std::string> backend_constraint =
            loader.GetTFLiteModelBackendConstraint(model_type);
        backend_constraint.has_value()) {
      resources.SetTFLiteModelBackendConstraint(model_type,
                                                *std::move(backend_constraint));
    }
    if (std::optional<std::string> prefer_activation_type =
            loader.GetTFLiteModelPreferActivationType(model_type);
        prefer_activation_type.has_value()) {
      resources.SetTFLiteModelPreferActivationType(
          model_type, *std::move(prefer_activation_type));
    }
  }
}

absl::Status CacheStreamedModelSection(
    const SectionInfo& section, proto::LlmMetadata::TfLiteModelType model_type,
    ModelResourcesStreaming& resources) {
  if (section.backend_constraint.has_value()) {
    resources.SetTFLiteModelBackendConstraint(model_type,
                                              *section.backend_constraint);
  }
  if (section.prefer_activation_type.has_value()) {
    resources.SetTFLiteModelPreferActivationType(
        model_type, *section.prefer_activation_type);
  }
  ABSL_LOG(INFO) << "Caching model (" << section.Size()
                 << " bytes) from stream for type: "
                 << static_cast<int>(model_type);
  ABSL_ASSIGN_OR_RETURN(std::vector<char> buffer, section.ReadBuffer());
  resources.SetModelBuffer(model_type, std::move(buffer));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<support::Tokenizer>>
LoadStreamedSentencePieceTokenizer(
    const SectionInfo& section, std::optional<BenchmarkInfo>& benchmark_info) {
#if defined(ENABLE_SENTENCEPIECE_TOKENIZER) && \
    !defined(LITERT_LM_SENTENCEPIECE_DISABLED)
  ABSL_VLOG(1) << "Processing section data type: SP_Tokenizer";
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kTokenizer));
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<char> buffer, section.ReadBuffer());
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<support::SentencePieceTokenizer> tokenizer,
      support::SentencePieceTokenizer::CreateFromBuffer(
          absl::string_view(buffer.data(), buffer.size())));
  ABSL_VLOG(1) << "SentencePieceTokenizer created.";
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  }
  return tokenizer;
#else
  static_cast<void>(benchmark_info);
  ABSL_RETURN_IF_ERROR(section.Discard());
  return absl::UnimplementedError(
      "SentencePiece tokenizer support is not enabled.");
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
}

absl::StatusOr<std::unique_ptr<support::Tokenizer>>
LoadStreamedHuggingFaceTokenizer(const SectionInfo& section,
                                 std::optional<BenchmarkInfo>& benchmark_info) {
  ABSL_VLOG(1) << "Processing section data type: HF_Tokenizer_Zlib";
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kTokenizer));
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<char> compressed_buffer,
                        section.ReadBuffer());
  std::vector<uint8_t> decompressed_buffer;
  ABSL_RETURN_IF_ERROR(schema::DecompressData(
      reinterpret_cast<const uint8_t*>(compressed_buffer.data()),
      compressed_buffer.size(), &decompressed_buffer));

  std::string json_data(
      reinterpret_cast<const char*>(decompressed_buffer.data()),
      decompressed_buffer.size());
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<support::HuggingFaceTokenizer> tokenizer,
      support::HuggingFaceTokenizer::CreateFromJson(json_data));
  ABSL_VLOG(1) << "HuggingFaceTokenizer created.";
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  }
  return tokenizer;
}

ModalityExecutorSettings ExtractModalityExecutorSettings(
    const EngineSettings& engine_settings) {
  ModalityExecutorSettings modality_settings;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    modality_settings.vision = std::make_unique<VisionExecutorSettings>(
        *engine_settings.GetVisionExecutorSettings());
    if (modality_settings.vision->GetAdapterBackend() != Backend::CPU) {
      ABSL_LOG(WARNING) << "Vision adapter backend is not CPU, which may cause "
                           "precision loss.";
    }
  }
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    modality_settings.audio = std::make_unique<AudioExecutorSettings>(
        *engine_settings.GetAudioExecutorSettings());
  }
  return modality_settings;
}

absl::StatusOr<StreamedVisionExecutorInit> InitializeStreamedVisionExecutor(
    EngineSettings& engine_settings, ModelResourcesStreaming& resources,
    const proto::LlmMetadata* absl_nullable llm_metadata, Environment& env) {
  StreamedVisionExecutorInit result;
  if (!engine_settings.GetVisionExecutorSettings().has_value()) {
    return result;
  }
  if (!resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_VISION_ENCODER)
           .ok()) {
    ABSL_LOG(WARNING) << "Vision executor settings are set, but the model "
                         "has no vision encoder. Vision is disabled.";
    // Clear the settings so the engine does not report vision support.
    // Otherwise sessions that enable the vision modality (which the Web SDK
    // does by default when vision settings are present) fail to load the
    // vision executor.
    engine_settings.GetMutableVisionExecutorSettings().reset();
    return result;
  }

  ABSL_RETURN_IF_ERROR(
      MaybeSelectVisionSignatures(engine_settings, resources, llm_metadata));
  result.settings = std::make_unique<VisionExecutorSettings>(
      *engine_settings.GetVisionExecutorSettings());
  if (result.settings->GetAdapterBackend() != Backend::CPU) {
    ABSL_LOG(WARNING) << "Vision adapter backend is not CPU, which may "
                         "cause precision loss.";
  }

  // The vision executor must be created now, while the vision encoder weights
  // are still in host memory. It cannot be lazily created later since the
  // model cannot be reloaded from the already consumed stream.
  //
  // The encoder and adapter keep a reference to the settings they are created
  // with, so pass the heap-allocated settings that ResourceManager takes
  // ownership of, which outlive the executor.
  ABSL_LOG(INFO) << "Creating vision executor from stream...";
  ABSL_ASSIGN_OR_RETURN(result.executor,
                        VisionLiteRtCompiledModelExecutor::Create(
                            *result.settings, env, resources));
  ABSL_LOG(INFO) << "Vision executor created.";

  // We only need to keep the encoder weights in host memory when running on
  // CPU. Do not release the vision adapter weights, as it always runs on CPU.
  if (result.settings->GetBackend() != Backend::CPU) {
    resources.ReleaseWeights(proto::LlmMetadata::TF_LITE_VISION_ENCODER);
    ABSL_LOG(INFO) << "Released host memory for vision encoder weights.";
  }
  return result;
}

absl::StatusOr<StreamedAudioExecutorInit> InitializeStreamedAudioExecutor(
    EngineSettings& engine_settings, ModelResourcesStreaming& resources,
    Environment& env) {
  StreamedAudioExecutorInit result;
  if (!engine_settings.GetAudioExecutorSettings().has_value()) {
    return result;
  }
  if (!resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW)
           .ok()) {
    ABSL_LOG(WARNING) << "Audio executor settings are set, but the model "
                         "has no audio encoder. Audio is disabled.";
    // Clear the settings so the engine does not report audio support (see the
    // vision case above).
    engine_settings.GetMutableAudioExecutorSettings().reset();
    return result;
  }

  result.settings = std::make_unique<AudioExecutorSettings>(
      *engine_settings.GetAudioExecutorSettings());

  // Create the audio executor now, while the audio encoder weights are still
  // in host memory. It cannot be lazily created later since the model cannot
  // be reloaded from the already consumed stream.
  ABSL_LOG(INFO) << "Creating audio executor from stream...";
  ABSL_ASSIGN_OR_RETURN(result.executor,
                        AudioLiteRtCompiledModelExecutor::Create(
                            *result.settings, env, resources));
  ABSL_LOG(INFO) << "Audio executor created.";

  // Only keep the encoder weights in host memory when running on CPU. The
  // audio adapter always runs on CPU, so keep its weights.
  if (result.settings->GetBackend() != Backend::CPU) {
    resources.ReleaseWeights(proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW);
    ABSL_LOG(INFO) << "Released host memory for audio encoder weights.";
  }
  return result;
}

}  // namespace litert::lm

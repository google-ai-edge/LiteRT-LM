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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_INIT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_INIT_UTILS_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources_streaming.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/vision/vision_executor.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_streaming_loader.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Copies any TFLite backend constraints and preferred activation types for
// `model_types` from `loader`'s parsed header into `resources`.
void CopyStreamingModelHints(
    LitertLmStreamingLoader& loader,
    absl::Span<const proto::LlmMetadata::TfLiteModelType> model_types,
    ModelResourcesStreaming& resources);

// Copies section-level backend constraints and preferred activation types into
// `resources`, then reads and caches the streamed TFLite flatbuffer bytes for
// `model_type`.
absl::Status CacheStreamedModelSection(
    const SectionInfo& section, proto::LlmMetadata::TfLiteModelType model_type,
    ModelResourcesStreaming& resources);

// Reads a streamed `SP_Tokenizer` section and constructs a
// SentencePieceTokenizer, recording the `kTokenizer` init phase in
// `benchmark_info` when present.
absl::StatusOr<std::unique_ptr<support::Tokenizer>>
LoadStreamedSentencePieceTokenizer(
    const SectionInfo& section, std::optional<BenchmarkInfo>& benchmark_info);

// Reads and zlib-decompresses a streamed `HF_Tokenizer_Zlib` section and
// constructs a HuggingFaceTokenizer, recording the `kTokenizer` init phase in
// `benchmark_info` when present.
absl::StatusOr<std::unique_ptr<support::Tokenizer>>
LoadStreamedHuggingFaceTokenizer(const SectionInfo& section,
                                 std::optional<BenchmarkInfo>& benchmark_info);

struct ModalityExecutorSettings {
  std::unique_ptr<VisionExecutorSettings> vision;
  std::unique_ptr<AudioExecutorSettings> audio;
};

// Copies optional vision and audio executor settings out of `engine_settings`
// into heap-allocated instances suitable for `ExecutionManager` ownership.
ModalityExecutorSettings ExtractModalityExecutorSettings(
    const EngineSettings& engine_settings);

struct StreamedVisionExecutorInit {
  std::unique_ptr<VisionExecutorSettings> settings;
  std::unique_ptr<VisionExecutor> executor;
};

// Initializes the vision executor eagerly from `resources` when vision is
// configured in `engine_settings`. If vision settings are set on a model
// without a vision encoder, logs a warning and clears the vision settings on
// `engine_settings` so the engine does not advertise vision support.
absl::StatusOr<StreamedVisionExecutorInit> InitializeStreamedVisionExecutor(
    EngineSettings& engine_settings, ModelResourcesStreaming& resources,
    const proto::LlmMetadata* absl_nullable llm_metadata, Environment& env);

struct StreamedAudioExecutorInit {
  std::unique_ptr<AudioExecutorSettings> settings;
  std::unique_ptr<AudioExecutor> executor;
};

// Initializes the audio executor eagerly from `resources` when audio is
// configured in `engine_settings`. If audio settings are set on a model without
// a LiteRT audio encoder, logs a warning and clears the audio settings on
// `engine_settings` so the engine does not advertise audio support.
absl::StatusOr<StreamedAudioExecutorInit> InitializeStreamedAudioExecutor(
    EngineSettings& engine_settings, ModelResourcesStreaming& resources,
    Environment& env);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_INIT_UTILS_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FACTORY_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"

namespace litert::omni::any2any {

// Populates `config` from embedded or caller-supplied `Any2AnyMetadata`,
// ignoring the sentinel voice `"default"` (case-insensitive), canonicalizing
// built-in voice names (`"Ethan"`, `"Chelsie"`) and resolving them to their
// speaker text BOS token IDs, and clamping special token IDs to fit within
// `llm_vocab_size` and `tts_vocab_size`.
void PopulateQwenOmniConfigFromProto(const lm::proto::Any2AnyMetadata& metadata,
                                     QwenOmniModelConfig& config);

// Applies per-session `Any2AnySessionConfig` overrides onto `config`, ignoring
// the sentinel voice `"default"` (case-insensitive), canonicalizing built-in
// voice names (`"Ethan"`, `"Chelsie"`) and mapping them to
// `tts_text_bos_token_id` when in range of `llm_vocab_size`, and propagating
// `seed`.
void ApplySessionConfigToQwenOmniConfig(
    const lm::proto::Any2AnySessionConfig& session_config,
    QwenOmniModelConfig& config);

// Creates shared `ModelResources` for Qwen2.5-Omni-3B, compiling the
// sub-models (`text_embedder`, optional `vision_encoder`, optional
// `audio_encoder`, `llm_backbone`, and, when `config.generate_audio` is true,
// required `tts_projector`, `tts_decoder`, and `audio_vocoder`) from a
// `.litertlm` bundle in a new `ModelResources` owning `env`. If `lm_resources`
// is non-null, it is reused directly and `model_path` is not reopened.
absl::StatusOr<ModelResources> CreateQwenOmniResources(
    const QwenOmniModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

// Wires the Qwen2.5-Omni Any2Any pipeline (`Any2AnyInputFanOutStage` ->
// `QwenOmniTextStage` + `QwenOmniAudioStage` + `QwenOmniImageStage` ->
// `QwenOmniFrontendStage` -> `QwenOmniLlmStage` -> `QwenOmniTtsStage` ->
// `QwenOmniVocoderStage`) and appends all stages to `stages`.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateQwenOmniComponents(
    const QwenOmniModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_FACTORY_H_

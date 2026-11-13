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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_FACTORY_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/any2any/lfm2_audio/lfm2_audio_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::any2any {

// Compiles and populates all LFM2.5-Audio-1.5B LiteRT models from a `.litertlm`
// file into shared `ModelResources`.
//
// Args:
//   config: Model configuration and backend settings for LFM2.5-Audio-1.5B.
//   model_path: Path to the `.litertlm` model file (used when `resources` does
//     not already hold an `lm::ModelResources`).
//   cache_dir: Directory for compiled model cache artifacts, or `":nocache"`.
//   backend: Default execution backend when a per-component backend override is
//     not set in `config`.
//   num_threads: Number of CPU execution threads.
//   env: LiteRT environment used for model compilation.
//   resources: Output `ModelResources` populated with compiled sub-models and
//     the shared LFM2 LLM backbone runner.
//
// Returns:
//   `absl::OkStatus()` on success, or an error status if any required model
//   section is missing or fails to compile.
absl::Status InitLfm2AudioResources(const Lfm2AudioModelConfig& config,
                                    absl::string_view model_path,
                                    absl::string_view cache_dir,
                                    lm::Backend backend, int num_threads,
                                    Environment& env,
                                    ModelResources& resources);

// Instantiates all stage components for an LFM2.5-Audio-1.5B Any2Any inference
// session and appends them in upstream-to-downstream order to `stages`.
//
// Args:
//   config: Session-level model configuration for LFM2.5-Audio-1.5B.
//   input_source: Non-null source stage providing `OmniSession::Input` items;
//     ownership is transferred into `stages[0]`.
//   resources: Non-null initialized `ModelResources` containing the compiled
//     LFM2.5-Audio sub-models, tokenizer, and LLM backbone runner.
//   stages: Output vector that takes ownership of all created pipeline stages.
//
// Returns:
//   A non-owning pointer to the terminal `Stage<Output>` owned by `stages`
//   (`stages.back().get()`), or an error status if stage creation fails.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateLfm2AudioComponents(
    const Lfm2AudioModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_FACTORY_H_

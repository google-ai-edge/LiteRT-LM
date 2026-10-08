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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_FACTORY_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::text2image {

// Compiles and populates all FLUX.2 LiteRT models (`textenc`, `dit`, `vae_dec`)
// into shared `ModelResources`.
absl::Status InitFlux2Resources(Flux2ModelConfig& config,
                                absl::string_view model_folder,
                                absl::string_view cache_dir,
                                lm::Backend backend, int num_threads,
                                ::litert::Environment& env,
                                ModelResources& resources);

// Instantiates all stage components for a FLUX.2 image generation inference
// session.
// The first stage (`stages[0]`) is `prompt_source`, and `*output_stage` is set
// to the final `Flux2VaeDecoderStage`.
absl::Status CreateFlux2Components(
    const Flux2ModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages,
    Stage<Output>* absl_nullable* absl_nonnull output_stage);

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_FLUX2_FACTORY_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_VISION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_VISION_UTILS_H_

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

// Selects the vision encoder and adapter signatures to load, based on
// max_vision_tokens_per_image if it is set by the user, or else the vision
// token limit derived from `llm_metadata` (if any). The selected signatures
// are stored in the vision executor settings of `engine_settings`.
//
// On web, only a single signature is selected by default, unless the user
// sets min_vision_tokens_per_image.
//
// Does nothing if vision is not enabled in `engine_settings`.
absl::Status MaybeSelectVisionSignatures(
    EngineSettings& engine_settings, ModelResources& model_resources,
    const proto::LlmMetadata* absl_nullable llm_metadata);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_VISION_UTILS_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_UTIL_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/embedding_engine_settings.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/magic_number_configs_helper.h"

namespace litert::lm {

struct OwnedEnvironment {
  std::unique_ptr<MagicNumberConfigsHelper> magic_number_configs_helper;
  Environment env;
};

// Creates a new LiteRT Environment.
// The caller takes ownership of the environment and its helper.
absl::StatusOr<OwnedEnvironment> CreateEnvironment(
    EngineSettings& engine_settings, ModelResources* model_resources);

absl::StatusOr<OwnedEnvironment> CreateEnvironment(
    EmbeddingEngineSettings& settings, ModelResources* model_resources);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_UTIL_H_

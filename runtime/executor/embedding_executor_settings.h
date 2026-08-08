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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_

#include <ostream>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

// Settings for configuring an embedding executor.
class EmbeddingExecutorSettings : public ExecutorSettingsBase {
 public:
  // Creates a default configuration for the embedding executor.
  //
  // Args:
  //   model_assets: The model assets to be used by the executor.
  //   backend: The execution backend to target.
  //
  // Returns:
  //   An EmbeddingExecutorSettings instance with default settings for the
  //   specified backend, or an error status if the default settings could not
  //   be created.
  static absl::StatusOr<EmbeddingExecutorSettings> CreateDefault(
      const ModelAssets& model_assets, Backend backend);

  // Constructs an EmbeddingExecutorSettings with the given model assets.
  //
  // Args:
  //   model_assets: The model assets to be used by the executor.
  explicit EmbeddingExecutorSettings(const ModelAssets& model_assets)
      : ExecutorSettingsBase(model_assets) {}
};

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingExecutorSettings& settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_EMBEDDING_EXECUTOR_SETTINGS_H_

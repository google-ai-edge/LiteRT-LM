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

#include "runtime/executor/embedding/embedding_executor_settings.h"

#include <cstddef>
#include <ostream>

#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::lm {

absl::StatusOr<EmbeddingExecutorSettings>
EmbeddingExecutorSettings::CreateDefault(const ModelAssets& model_assets,
                                         Backend backend) {
  EmbeddingExecutorSettings settings(model_assets);
  ABSL_RETURN_IF_ERROR(settings.SetBackend(backend));
  return settings;
}

std::ostream& operator<<(std::ostream& os,
                         const EmbeddingExecutorSettings& settings) {
  os << "backend: " << settings.GetBackend() << std::endl;
  os << "npu_options: " << settings.GetNpuOptions() << std::endl;
  const auto& per_token = settings.GetPerTokenOutputs();
  os << "per_token_outputs: {token_embeddings: "
     << (per_token.token_embeddings ? "true" : "false")
     << ", hidden_state_layers: [";
  for (size_t i = 0; i < per_token.hidden_state_layers.size(); ++i) {
    if (i > 0) os << ", ";
    os << per_token.hidden_state_layers[i];
  }
  os << "]}" << std::endl;
  return os;
}

}  // namespace litert::lm

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_INTERNAL_H_

#include <memory>
#include <vector>

#include "runtime/engine/embedding_engine.h"
#include "runtime/engine/embedding_engine_settings.h"

struct LiteRtLmEmbeddingEngineSettings {
  std::unique_ptr<litert::lm::EmbeddingEngineSettings> settings;
};

struct LiteRtLmEmbeddingOptions {
  litert::lm::EmbeddingOptions options;
};

struct LiteRtLmEmbeddingResponse {
  litert::lm::EmbeddingResponse response;
};

struct LiteRtLmEmbeddingResponses {
  std::vector<litert::lm::EmbeddingResponse> responses;
};

struct LiteRtLmEmbeddingEngine {
  std::unique_ptr<litert::lm::EmbeddingEngine> engine;
};

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_EMBEDDING_ENGINE_INTERNAL_H_

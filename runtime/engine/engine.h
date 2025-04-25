// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_

#include <memory>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/odml/litert_lm/runtime/engine/io_types.h"
#include "third_party/odml/litert_lm/runtime/engine/llm_model_settings.h"

namespace litert::lm {

class Engine {
 public:
  virtual ~Engine() = default;

  // Session is responsible for hosting the internal state (e.g. conversation
  // history) of each separate interaction with LLM.
  class Session {
   public:
    virtual ~Session() = default;

    // Adds the input prompt/query to the model for starting the prefilling
    // process. Note that the user can break down their prompt/query into
    // multiple chunks and call this function multiple times.
    virtual absl::Status AddTextPrompt(absl::string_view input) = 0;

    // Starts the decoding process for the model to predict the response based
    // on the input prompt/query added after using AddTextPrompt (or
    // AddImagePrompt) functions.
    virtual absl::StatusOr<Responses> PredictSync() = 0;
  };

  // Method to create Engine.
  static absl::StatusOr<std::unique_ptr<Engine>> CreateEngine(
      const LlmModelSettings& settings_struct);

  // Method to create the Session.
  virtual absl::StatusOr<std::unique_ptr<Session>> CreateSession() const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_

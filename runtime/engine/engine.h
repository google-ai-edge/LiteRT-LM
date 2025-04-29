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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

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
    //
    // This is a blocking call and the function will return when the prefill
    // process is done.
    virtual absl::Status RunPrefill(absl::string_view input) = 0;
    // This is a not blocking call and the function will return right away.
    virtual absl::Status RunPrefillAsync(absl::string_view input) = 0;

    // Starts the decoding process for the model to predict the response based
    // on the input prompt/query added after using RunPrefill* functions.
    // This is a blocking call and the function will return when the decoding
    // process is done.
    virtual absl::StatusOr<Responses> RunDecode() = 0;
  };

  // Method to create Engine.
  static absl::StatusOr<std::unique_ptr<Engine>> CreateEngine(
      const EngineSettings& settings);

  // Method to create the Session.
  virtual absl::StatusOr<std::unique_ptr<Session>> CreateSession() const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_

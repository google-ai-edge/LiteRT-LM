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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_ENGINE_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_ENGINE_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::omni {

// Interface for LiteRT-LM Engine execution in omni pipelines.
// Handles multimodal prefill and autoregressive decoding backed by
// litert::lm::Engine and litert::lm::SessionInterface.
class LiteRtLmEngineRunner {
 public:
  virtual ~LiteRtLmEngineRunner() = default;

  // Runs the prefill phase on a sequence of multimodal inputs.
  virtual absl::Status Prefill(std::vector<lm::InputData> inputs) = 0;

  // Runs decode and returns the Responses container from the engine.
  virtual absl::StatusOr<lm::Responses> Decode(
      const lm::DecodeConfig& decode_config) = 0;

  // Resets the runner state and session.
  virtual absl::Status Reset() = 0;

  // Returns a pointer to ModelResources if available, or nullptr otherwise.
  virtual const lm::ModelResources* model_resources() const { return nullptr; }
  virtual lm::ModelResources* mutable_model_resources() { return nullptr; }
};

// Implementation of LiteRtLmEngineRunner backed by litert::lm::Engine and
// litert::lm::SessionInterface for models whose full multimodal generation is
// handled by the LiteRT-LM Engine.
class LiteRtLmEngineRunnerImpl : public LiteRtLmEngineRunner {
 public:
  LiteRtLmEngineRunnerImpl(std::unique_ptr<lm::Engine> engine,
                           std::unique_ptr<lm::SessionInterface> session,
                           const lm::SessionConfig& session_config,
                           std::unique_ptr<lm::ModelResources> model_resources);

  ~LiteRtLmEngineRunnerImpl() override = default;

  absl::Status Prefill(std::vector<lm::InputData> inputs) override;

  absl::StatusOr<lm::Responses> Decode(
      const lm::DecodeConfig& decode_config) override;

  absl::Status Reset() override;

  const lm::ModelResources* model_resources() const override {
    return model_resources_.get();
  }
  lm::ModelResources* mutable_model_resources() override {
    return model_resources_.get();
  }

  const lm::SessionConfig& session_config() const { return session_config_; }

 private:
  std::unique_ptr<lm::Engine> engine_;
  std::unique_ptr<lm::SessionInterface> session_;
  lm::SessionConfig session_config_;
  std::unique_ptr<lm::ModelResources> model_resources_;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_LITERT_LM_ENGINE_RUNNER_H_

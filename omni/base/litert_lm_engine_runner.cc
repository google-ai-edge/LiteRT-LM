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

#include "omni/base/litert_lm_engine_runner.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::omni {

LiteRtLmEngineRunnerImpl::LiteRtLmEngineRunnerImpl(
    std::unique_ptr<lm::Engine> engine,
    std::unique_ptr<lm::SessionInterface> session,
    const lm::SessionConfig& session_config,
    std::unique_ptr<lm::ModelResources> model_resources)
    : engine_(std::move(engine)),
      session_(std::move(session)),
      session_config_(session_config),
      model_resources_(std::move(model_resources)) {}

absl::Status LiteRtLmEngineRunnerImpl::Prefill(
    std::vector<lm::InputData> inputs) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError(
        "LiteRtLmEngineRunnerImpl::Prefill received empty inputs.");
  }
  return session_->RunPrefill(inputs);
}

absl::StatusOr<lm::Responses> LiteRtLmEngineRunnerImpl::Decode(
    const lm::DecodeConfig& decode_config) {
  return session_->RunDecode(decode_config);
}

absl::Status LiteRtLmEngineRunnerImpl::Reset() {
  auto status = session_->RewindToStep(0);
  if (!status.ok()) {
    if (engine_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(session_, engine_->CreateSession(session_config_));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::omni

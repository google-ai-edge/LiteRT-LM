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

#include "c/c_api_internal.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter_internal.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"

namespace litert::lm::c {

bool IsValidEngineSettings(const LiteRtLmEngineSettings* settings) {
  if (settings != nullptr && settings->settings != nullptr) {
    return true;
  }
  SetLastError(absl::StatusCode::kInvalidArgument, "Invalid engine settings.");
  return false;
}

bool IsValidSessionConfig(const LiteRtLmSessionConfig* config) {
  if (config != nullptr && config->config != nullptr) {
    return true;
  }
  SetLastError(absl::StatusCode::kInvalidArgument, "Invalid session config.");
  return false;
}

bool IsValidEngine(const LiteRtLmEngine* engine) {
  if (engine != nullptr && engine->engine != nullptr) {
    return true;
  }
  SetLastError(absl::StatusCode::kInvalidArgument, "Invalid engine.");
  return false;
}

bool IsValidSession(const LiteRtLmSession* session) {
  if (session != nullptr && session->session != nullptr) {
    return true;
  }
  SetLastError(absl::StatusCode::kInvalidArgument, "Invalid session.");
  return false;
}

bool IsValidActivationDataType(const LiteRtLmActivationDataType& type) {
  switch (RawEnumValue(type)) {
    case kLiteRtLmActivationDataTypeFloat32:
    case kLiteRtLmActivationDataTypeFloat16:
    case kLiteRtLmActivationDataTypeInt16:
    case kLiteRtLmActivationDataTypeInt8:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<std::vector<InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs) {
  if (inputs == nullptr && num_inputs > 0) {
    return absl::InvalidArgumentError(
        "inputs must not be NULL when num_inputs is non-zero.");
  }
  std::vector<InputData> engine_inputs;
  if (inputs == nullptr || num_inputs == 0) {
    return engine_inputs;
  }
  engine_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    if (inputs[i] != nullptr) {
      absl::StatusOr<InputData> copy = CreateInputDataCopy(inputs[i]->data);
      if (!copy.ok()) {
        return copy.status();
      }
      engine_inputs.push_back(std::move(*copy));
    }
  }
  return engine_inputs;
}

absl::StatusOr<std::optional<Backend>> ParseOptionalBackend(
    absl::string_view backend_str) {
  if (backend_str.empty()) {
    return std::nullopt;
  }
  absl::StatusOr<Backend> backend = GetBackendFromString(backend_str);
  if (!backend.ok()) {
    return backend.status();
  }
  return *backend;
}

void EnableConfiguredModalities(const EngineSettings& engine_settings,
                                SessionConfig& session_config) {
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    session_config.SetAudioModalityEnabled(true);
  }
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    session_config.SetVisionModalityEnabled(true);
  }
}

void SetExecutorCacheDirs(
    ExecutorSettingsBase& main_settings,
    std::optional<VisionExecutorSettings>& vision_settings,
    std::optional<AudioExecutorSettings>& audio_settings,
    absl::string_view cache_dir) {
  const std::string dir(cache_dir);
  main_settings.SetCacheDir(dir);
  if (vision_settings.has_value()) {
    vision_settings->SetCacheDir(dir);
  }
  if (audio_settings.has_value()) {
    audio_settings->SetCacheDir(dir);
  }
}

}  // namespace litert::lm::c

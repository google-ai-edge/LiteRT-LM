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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_C_API_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_C_API_INTERNAL_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/engine_internal.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/vision/vision_executor_settings.h"

namespace litert::lm::c {

// Handle validation helpers. Each returns true if the handle and its wrapped
// C++ instance are non-null; otherwise records a `kInvalidArgument`
// thread-local error and returns false.
bool IsValidEngineSettings(const LiteRtLmEngineSettings* settings);
bool IsValidSessionConfig(const LiteRtLmSessionConfig* config);
bool IsValidEngine(const LiteRtLmEngine* engine);
bool IsValidSession(const LiteRtLmSession* session);

// Returns true if `type` holds a declared `LiteRtLmActivationDataType` value,
// inspecting its raw integer representation without C++ enum-load UB.
bool IsValidActivationDataType(const LiteRtLmActivationDataType& type);

// Copies an array of C `LiteRtLmInputData` pointers into C++ `InputData`
// objects, skipping null entries. Fails with `kInvalidArgument` if `inputs` is
// null while `num_inputs` is non-zero.
absl::StatusOr<std::vector<InputData>> ToEngineInputData(
    const LiteRtLmInputData* const* inputs, size_t num_inputs);

// Parses an optional backend name string, returning `std::nullopt` when
// `backend_str` is empty.
absl::StatusOr<std::optional<Backend>> ParseOptionalBackend(
    absl::string_view backend_str);

// Enables audio and vision modalities on `session_config` whenever the
// corresponding executor settings are present in `engine_settings`.
void EnableConfiguredModalities(const EngineSettings& engine_settings,
                                SessionConfig& session_config);

// Sets `cache_dir` on `main_settings` and on `vision_settings` and
// `audio_settings` when present.
void SetExecutorCacheDirs(
    ExecutorSettingsBase& main_settings,
    std::optional<VisionExecutorSettings>& vision_settings,
    std::optional<AudioExecutorSettings>& audio_settings,
    absl::string_view cache_dir);

// Invokes a C streaming callback with a single `LiteRtLmStreamChunk`.
inline void EmitStreamChunk(LiteRtLmStreamCallback callback,
                            void* callback_data, const char* text,
                            bool is_final, const char* error_msg) {
  LiteRtLmStreamChunk chunk{text, is_final, error_msg};
  callback(callback_data, &chunk);
}

}  // namespace litert::lm::c

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_C_API_INTERNAL_H_

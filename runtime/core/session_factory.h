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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_FACTORY_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

// Factory method to create and initialize a Engine::Session from the given
// settings. Note that this function should be updated to take in the
// SessionConfig and be refactored with registry pattern.
// image_preprocessor and vision_executor are optional and can be nullptr.
// If image input is used in the session, the vision_executor must be provided.
// If audio input is used in the session, the audio_executor must be provided.
absl::StatusOr<std::unique_ptr<Engine::Session>> InitializeSession(
    LlmExecutor* absl_nonnull executor, Tokenizer* absl_nonnull tokenizer,
    VisionExecutor* vision_executor, AudioExecutor* audio_executor,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* absl_nonnull worker_thread_pool);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_FACTORY_H_

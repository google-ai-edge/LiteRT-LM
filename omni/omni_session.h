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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/asr/text_merger.h"
#include "omni/base/io_types.h"

namespace litert::omni {

// Pure interface for a unified session processing multimodal inputs (`Input`)
// and producing multimodal outputs (`Output`).
class OmniSession {
 public:
  // Text input payload for TTS synthesis.
  struct TextInput {
    std::string text;
  };

  // Audio input payload for ASR transcription.
  struct AudioInput {
    std::vector<float> pcm_samples;
    int sample_rate_hz = 16000;
    int num_channels = 1;
  };

  // Unified input variant for Process and ProcessAsync.
  using Input = std::variant<TextInput, AudioInput>;

  using TextOutput = asr::TextMerger::MergeResult;
  using AudioOutput = ::litert::omni::AudioOutput;
  using Output = std::variant<std::monostate, TextOutput, AudioOutput>;
  using AsyncCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<Output>)>;

  virtual ~OmniSession() = default;

  // Resets session state for a new stream.
  virtual void Reset() = 0;

  // Processes `input` synchronously (`AudioInput` -> `TextOutput` for ASR;
  // `TextInput` -> `AudioOutput` for TTS).
  virtual absl::StatusOr<Output> Process(const Input& input) = 0;

  // Processes `input` asynchronously using the underlying session's thread
  // pool and emits `Output` chunks (`TextOutput` for ASR, `AudioOutput` for
  // TTS) to `callback`.
  virtual absl::Status ProcessAsync(const Input& input,
                                    AsyncCallback callback) = 0;

  // Flushes remaining buffered output (`TextOutput` for ASR, `AudioOutput` for
  // TTS) at the end of a stream.
  virtual absl::StatusOr<Output> Flush() = 0;
};

// Pure interface for creating `OmniSession` instances.
class OmniSessionFactory {
 public:
  virtual ~OmniSessionFactory() = default;

  virtual absl::StatusOr<std::unique_ptr<OmniSession>> Create() = 0;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_OMNI_SESSION_H_

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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_AUDIO_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_AUDIO_PREPROCESSOR_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "support/preprocessor/audio_preprocessor.h"
#include "support/util/io_types.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep

namespace litert::support {

// Audio preprocessor implementation that bypasses the preprocessing and returns
// the input audio directly.
class ByPassAudioPreprocessor : public AudioPreprocessor {
 public:
  static absl::StatusOr<std::unique_ptr<ByPassAudioPreprocessor>> Create(
      const AudioPreprocessorConfig& config) {
    return std::make_unique<ByPassAudioPreprocessor>();
  }

  absl::StatusOr<InputAudio> Preprocess(
      const InputAudio& input_audio) override {
    if (input_audio.IsTensorBuffer()) {
      ASSIGN_OR_RETURN(auto processed_audio_tensor,
                       input_audio.GetPreprocessedAudioTensor());
      LITERT_ASSIGN_OR_RETURN(auto processed_audio_tensor_with_reference,
                              processed_audio_tensor->Duplicate());
      InputAudio processed_audio(
          std::move(processed_audio_tensor_with_reference));
      return processed_audio;
    }
    return absl::InvalidArgumentError("Input audio is not preprocessed.");
  };

  // No-op for bypass preprocessor.
  void Reset() override {}
};

}  // namespace litert::support

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_PREPROCESSOR_BY_PASS_AUDIO_PREPROCESSOR_H_

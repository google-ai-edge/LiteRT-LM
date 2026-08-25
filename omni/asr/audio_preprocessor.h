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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"

namespace litert::omni::asr {

// Abstract interface for preprocessing raw PCM audio into features.
class AudioPreprocessor
    : public SingleThreadedStageWithDeque<std::vector<float>> {
 public:
  explicit AudioPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source)
      : audio_source_(*audio_source) {}

  ~AudioPreprocessor() override = default;

 protected:
  bool NeedScheduleInternal() const override {
    return audio_source_.HasOutput();
  }

  Stage<std::vector<float>>& audio_source_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_PREPROCESSOR_H_

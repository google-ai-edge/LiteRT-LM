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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_DUMMY_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_DUMMY_PREPROCESSOR_H_

#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/audio_source.h"

namespace litert::omni::asr {

// Dummy audio preprocessor that passes raw PCM speech audio directly as
// features.
// TODO: b/524681030 - Consider a way to pass data without copying.
class DummyPreprocessor : public AudioPreprocessor {
 public:
  // Constructs a raw-PCM preprocessor connected to `audio_source`.
  //
  // args:
  //   audio_source: Upstream stage emitting raw PCM audio chunks.
  //   pad_to_num_samples: Fixed PCM input window size expected by the model (in
  //     samples). Short follow-up chunks are shifted back over the previous
  //     full window when available, and standalone short chunks are padded up
  //     to this length; 0 disables fixed-window shifting and padding.
  explicit DummyPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source,
      int pad_to_num_samples = 0);

  explicit DummyPreprocessor(AudioSource* absl_nonnull audio_source,
                             int pad_to_num_samples = 0);

  ~DummyPreprocessor() override = default;

 protected:
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

 private:
  // Fixed PCM input window size expected by the model (in samples). Short
  // follow-up chunks are shifted back over `previous_window_`, and standalone
  // short chunks are padded up to this length; 0 disables padding.
  const int pad_to_num_samples_;
  // Last `pad_to_num_samples_` samples of the most recent full-size non-silent
  // chunk.
  std::vector<float> previous_window_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_DUMMY_PREPROCESSOR_H_

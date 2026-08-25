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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_

#include <vector>

#include "omni/base/stage.h"

namespace litert::omni::asr {

// Abstract interface for providing raw PCM audio data.
class AudioSource : public SingleThreadedStageWithDeque<std::vector<float>> {
 public:
  ~AudioSource() override = default;

  // Returns the audio sampling rate in Hertz (e.g. 16000).
  virtual int GetSampleRateHz() const = 0;

  // Returns the number of audio channels (e.g. 1 for mono).
  virtual int GetNumChannels() const = 0;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_AUDIO_SOURCE_H_

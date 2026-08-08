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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_VOCODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_VOCODER_H_

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/latent_decoder.h"

namespace litert::omni::tts {

// Abstract interface for Stage 4: Neural vocoder audio waveform synthesis.
class Vocoder : public SingleThreadedStageWithDeque<AudioOutput> {
 public:
  explicit Vocoder(
      Stage<LatentDecoder::LatentOutput>* absl_nonnull latent_decoder)
      : latent_decoder_(*latent_decoder) {}

  ~Vocoder() override = default;

  // Resets internal state for a new TTS stream.
  virtual void Reset() = 0;

  // Flushes remaining synthesized audio at end of stream.
  virtual absl::Status Flush() = 0;

 protected:
  bool NeedScheduleInternal() const override {
    return latent_decoder_.HasOutput();
  }

  Stage<LatentDecoder::LatentOutput>& latent_decoder_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_VOCODER_H_

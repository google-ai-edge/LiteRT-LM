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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_AUDIO_PREPROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_AUDIO_PREPROCESSOR_H_

#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/speech2text/audio_source.h"

namespace litert::omni::asr {

// Preprocessed audio features for a single chunk and optional chunk metadata.
struct PreprocessedAudio {
  // Portion of the encoder input (along the time axis) that should be decoded
  // and reported for a chunk, as fractions of the model window.
  struct DecodeRange {
    // The final chunk of a long stream is shifted back to end exactly at the
    // end of the audio, so the model always sees a full window of real audio;
    // this is the fraction in [0, 1) of the window, from its start, that was
    // already covered by the previous window. Tokens emitted there are dropped
    // and the remaining timestamps are made relative to this point.
    float start_fraction = 0.0f;
    // Fraction in (0, 1] of the window backed by real audio (a stream shorter
    // than the window is padded); nothing is decoded past it.
    float valid_fraction = 1.0f;
  };

  std::vector<float> features;
  // Optional alternative features extracted with centered 25 ms STFT framing
  // for retrying full windows whose unpadded framing collapses in the
  // encoder/decoder.
  std::optional<std::vector<float>> centered_features;
  std::optional<DecodeRange> decode_range;
  // True when this chunk is a progressive warm-start prefix emitted before the
  // first full model window is accumulated.
  bool is_warm_start = false;
};

// Abstract interface for preprocessing raw PCM audio into features.
class AudioPreprocessor
    : public SingleThreadedStageWithDeque<PreprocessedAudio> {
 public:
  using DecodeRange = PreprocessedAudio::DecodeRange;
  using PreprocessedAudio = ::litert::omni::asr::PreprocessedAudio;

  explicit AudioPreprocessor(
      Stage<std::vector<float>>* absl_nonnull audio_source)
      : audio_source_(*audio_source) {}

  explicit AudioPreprocessor(AudioSource* absl_nonnull audio_source)
      : audio_source_(*audio_source), typed_audio_source_(audio_source) {}

  ~AudioPreprocessor() override = default;

 protected:
  bool NeedScheduleInternal() const override {
    return audio_source_.HasOutput();
  }

  bool IsLastInputWarmStart() const {
    return typed_audio_source_ != nullptr &&
           typed_audio_source_->IsLastOutputWarmStart();
  }

  Stage<std::vector<float>>& audio_source_;
  AudioSource* absl_nullable typed_audio_source_ = nullptr;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_AUDIO_PREPROCESSOR_H_

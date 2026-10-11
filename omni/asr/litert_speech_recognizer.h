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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LITERT_SPEECH_RECOGNIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LITERT_SPEECH_RECOGNIZER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {

// SpeechRecognizer stage implementation powered by LiteRtRunner
// and a LiteRtSpeechRecognizer::Decoder.
class LiteRtSpeechRecognizer : public SpeechRecognizer {
 public:
  // Interface for decoding the ASR encoder output into token IDs.
  class Decoder {
   public:
    virtual ~Decoder() = default;

    // Decodes encoder output buffers into token IDs and optional timestamps.
    virtual absl::StatusOr<std::vector<DecodedToken>> Decode(
        std::vector<::litert::TensorBuffer>& encoder_outputs) = 0;

    // Portion of the encoder input (along the time axis) that the next
    // `Decode()` call should report tokens for, as fractions of the window.
    // Applies to one `Decode()` call only; decoders fall back to the whole
    // window afterwards.
    struct DecodeRange {
      // The final chunk of a long stream is shifted back to end exactly at the
      // end of the audio, so the model always sees a full window of real
      // audio; this is the fraction of the window, from its start, that was
      // already covered by the previous window. Tokens emitted there are
      // dropped and the remaining timestamps are made relative to this point.
      float start_fraction = 0.0f;
      // Fraction of the window backed by real audio (a stream shorter than the
      // window is padded); nothing is decoded past it.
      float valid_fraction = 1.0f;
    };
    // Defaults to a no-op (decode and report the whole window).
    virtual void SetDecodeRange(const DecodeRange& range) {}
  };

  // `audio_preprocessor` is the feature stage to pull from. If `chunk_info` is
  // non-null, its `last_chunk_start_fraction()` / `last_chunk_valid_fraction()`
  // are forwarded to the decoder (`Decoder::SetDecodeRange`) before every
  // `Decode()` call.
  static absl::StatusOr<std::unique_ptr<LiteRtSpeechRecognizer>> Create(
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
      std::unique_ptr<Decoder> absl_nonnull decoder,
      const AudioPreprocessor* absl_nullable chunk_info = nullptr);

  ~LiteRtSpeechRecognizer() override = default;

  void PushOutputForTesting(std::vector<DecodedToken> output);

 protected:
  void ResetInternal() override;
  absl::Status ScheduleInternal() override;

 private:
  LiteRtSpeechRecognizer(
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
      std::unique_ptr<Decoder> absl_nonnull decoder,
      const AudioPreprocessor* absl_nullable chunk_info,
      std::vector<::litert::TensorBuffer> encode_input_buffers,
      std::vector<::litert::TensorBuffer> encode_output_buffers);

  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  const std::unique_ptr<Decoder> absl_nonnull decoder_;
  const AudioPreprocessor* absl_nullable chunk_info_;
  std::vector<::litert::TensorBuffer> encode_input_buffers_;
  std::vector<::litert::TensorBuffer> encode_output_buffers_;
  bool prev_chunk_tail_empty_ = false;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LITERT_SPEECH_RECOGNIZER_H_

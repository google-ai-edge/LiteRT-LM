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
  };

  static absl::StatusOr<std::unique_ptr<LiteRtSpeechRecognizer>> Create(
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
      std::unique_ptr<Decoder> absl_nonnull decoder);

  ~LiteRtSpeechRecognizer() override = default;

  void PushOutputForTesting(std::vector<DecodedToken> output);

 protected:
  absl::Status ScheduleInternal() override;

 private:
  LiteRtSpeechRecognizer(
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
      std::unique_ptr<Decoder> absl_nonnull decoder,
      std::vector<::litert::TensorBuffer> encode_input_buffers,
      std::vector<::litert::TensorBuffer> encode_output_buffers);

  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  const std::unique_ptr<Decoder> absl_nonnull decoder_;
  std::vector<::litert::TensorBuffer> encode_input_buffers_;
  std::vector<::litert::TensorBuffer> encode_output_buffers_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_LITERT_SPEECH_RECOGNIZER_H_

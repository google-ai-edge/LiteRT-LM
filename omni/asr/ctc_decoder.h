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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_CTC_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_CTC_DECODER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/speech_recognizer.h"

namespace litert::omni::asr {

// Connectionist Temporal Classification (CTC) decoder implementation.
class CtcDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  static absl::StatusOr<std::unique_ptr<CtcDecoder>> Create(int vocab_size);

  ~CtcDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs) override;

 private:
  explicit CtcDecoder(int vocab_size);

  const int vocab_size_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_CTC_DECODER_H_

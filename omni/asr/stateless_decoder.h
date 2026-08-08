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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_STATELESS_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_STATELESS_DECODER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"

namespace litert::omni::asr {

// Stateless single-signature autoregressive decoder implementation.
class StatelessDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  static absl::StatusOr<std::unique_ptr<StatelessDecoder>> Create(
      LiteRtRunner* absl_nonnull runner, int decode_start_token_id = -1,
      int decode_stop_token_id = -1, int decode_skip_until_token_id = -1);

  ~StatelessDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs) override;

 private:
  StatelessDecoder(LiteRtRunner* absl_nonnull runner,
                   std::vector<::litert::TensorBuffer> decode_input_buffers,
                   std::vector<::litert::TensorBuffer> decode_output_buffers,
                   size_t num_logits_per_token, size_t num_token_ids,
                   int decode_start_token_id, int decode_stop_token_id,
                   int decode_skip_until_token_id);

  LiteRtRunner* const absl_nonnull runner_;
  std::vector<::litert::TensorBuffer> decode_input_buffers_;
  std::vector<::litert::TensorBuffer> decode_output_buffers_;
  const size_t num_logits_per_token_;
  const size_t num_token_ids_;
  const int decode_start_token_id_;
  const int decode_stop_token_id_;
  const int decode_skip_until_token_id_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_STATELESS_DECODER_H_

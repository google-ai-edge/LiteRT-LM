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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TDT_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TDT_DECODER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/litert_speech_recognizer.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"

namespace litert::omni::asr {

// Token Duration Transducer (TDT) decoder implementation supporting optional
// stateful decoding.
class TdtDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  static absl::StatusOr<std::unique_ptr<TdtDecoder>> Create(
      LiteRtRunner* absl_nonnull runner, int decode_start_token_id = 8192,
      int decode_statefully_after = 4);

  ~TdtDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<TensorBuffer>& encoder_outputs) override;

 private:
  TdtDecoder(
      LiteRtRunner* absl_nonnull runner,
      std::vector<TensorBuffer> decode_input_buffers,
      std::vector<TensorBuffer> decode_output_buffers,
      std::optional<std::vector<TensorBuffer>> stateful_decode_input_buffers,
      std::optional<std::vector<TensorBuffer>> stateful_decode_output_buffers,
      size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
      int decode_start_token_id, int decode_statefully_after);

  // Prepares input and output buffers for a single inference call.
  absl::StatusOr<std::vector<TensorBuffer>> GetInputBuffersForInference(
      std::vector<TensorBuffer>& encoder_outputs,
      TensorBuffer& token_ids_buffer);
  absl::StatusOr<std::vector<TensorBuffer>> GetOutputBuffersForInference(
      TensorBuffer& logits_buffer);

  LiteRtRunner* const absl_nonnull runner_;
  std::vector<TensorBuffer> decode_input_buffers_;
  std::vector<TensorBuffer> decode_output_buffers_;
  std::optional<std::vector<TensorBuffer>> stateful_decode_input_buffers_;
  std::optional<std::vector<TensorBuffer>> stateful_decode_output_buffers_;
  std::vector<TensorBuffer*> input_states_buffers_;
  std::vector<TensorBuffer*> output_states_buffers_;
  const size_t max_time_index_;
  const size_t num_token_ids_;
  const size_t num_logits_per_token_;
  const int decode_start_token_id_;
  const int decode_statefully_after_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TDT_DECODER_H_

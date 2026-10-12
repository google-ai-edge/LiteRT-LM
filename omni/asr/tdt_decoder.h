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
#include "omni/base/stateful_litert_runner.h"

namespace litert::omni::asr {

// Token Duration Transducer (TDT) decoder implementation supporting optional
// stateful decoding.
class TdtDecoder : public LiteRtSpeechRecognizer::Decoder {
 public:
  static absl::StatusOr<std::unique_ptr<TdtDecoder>> Create(
      LiteRtRunner* absl_nonnull runner, int decode_start_token_id = 8192,
      int decode_statefully_after = 4, int max_symbols_per_step = 3);

  ~TdtDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<TensorBuffer>& encoder_outputs) override;

  // Restricts the next `Decode()` to `range` (see `Decoder::DecodeRange`):
  // encoder frames `[floor(start_fraction * max_time_index),
  // round(valid_fraction * max_time_index) + kValidFrameMargin)` are reported,
  // i.e. frames whose center lies in real audio (frames straddling the
  // padding boundary otherwise emit stray partial tokens), with timestamps
  // relative to the first reported frame. Frames before the start are still
  // decoded (the prediction network is autoregressive) but their tokens are
  // dropped. Reset to the whole window after each call.
  void SetDecodeRange(const DecodeRange& range) override;

  // Extra encoder frames decoded past the last real-audio frame to absorb the
  // encoder's receptive field and TDT duration jitter at the boundary.
  static constexpr size_t kValidFrameMargin = 0;

 private:
  TdtDecoder(
      LiteRtRunner* absl_nonnull runner,
      std::unique_ptr<StatefulLiteRtRunner> absl_nullable stateful_runner,
      std::vector<TensorBuffer> decode_input_buffers,
      std::vector<TensorBuffer> decode_output_buffers,
      std::optional<TensorBuffer> stateful_decode_token_ids_buffer,
      size_t max_time_index, size_t num_token_ids, size_t num_logits_per_token,
      int decode_start_token_id, int decode_statefully_after,
      int max_symbols_per_step);

  // Number of encoder frames to decode for the current chunk, derived from
  // `range_.valid_fraction` and `max_time_index_`.
  size_t EffectiveMaxTimeIndex() const;
  // First encoder frame whose tokens are reported for the current chunk,
  // derived from `range_.start_fraction` and `max_time_index_`.
  size_t FirstReportedTimeIndex() const;

  LiteRtRunner& runner_;
  std::unique_ptr<StatefulLiteRtRunner> absl_nullable stateful_runner_;
  std::vector<TensorBuffer> decode_input_buffers_;
  std::vector<TensorBuffer> decode_output_buffers_;
  std::optional<TensorBuffer> stateful_decode_token_ids_buffer_;
  const size_t max_time_index_;
  const size_t num_token_ids_;
  const size_t num_logits_per_token_;
  const int decode_start_token_id_;
  const int decode_statefully_after_;
  const int max_symbols_per_step_;
  DecodeRange range_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_TDT_DECODER_H_

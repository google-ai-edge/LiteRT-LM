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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_DECODER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/speech2text/speech_recognizer.h"
#include "omni/speech2text/stateful_decoder.h"

namespace litert::omni::asr {

struct WhisperDecoderOptions {
  // If true, parses timestamp tokens (50364..51864) and assigns timestamps
  // in milliseconds to emitted DecodedTokens while suppressing timestamp
  // tokens from text output. If false, timestamp tokens are suppressed from
  // logit selection.
  bool enable_timestamps = true;

  // Token IDs to unconditionally suppress (masked to -infinity).
  std::vector<int> suppress_tokens = {};

  // Token IDs to suppress on the first decoding step after the prompt
  // prefix (e.g. 220 for blank, 50257 for <|endoftext|>).
  std::vector<int> begin_suppress_tokens = {220, 50257};

  // Maximum number of autoregressive decoding steps.
  int max_decode_steps = 256;
};

// Whisper-specific autoregressive decoder that inherits from StatefulDecoder,
// incorporating Whisper token suppression and native acoustic timestamp
// token extraction for precise alignment and text merging.
class WhisperDecoder : public StatefulDecoder {
 public:
  using Options = WhisperDecoderOptions;

  static constexpr int kWhisperEndOfTextTokenId = 50257;
  static constexpr int kWhisperStartOfTranscriptTokenId = 50258;
  static constexpr int kWhisperTranscribeTokenId = 50359;
  static constexpr int kWhisperNoTimestampsTokenId = 50363;
  static constexpr int kWhisperTimestampBeginTokenId = 50364;
  static constexpr int kWhisperTimestampEndTokenId = 51864;
  static constexpr float kWhisperTimeStepSeconds = 0.02f;  // 20ms per step

  // Creates a WhisperDecoder instance.
  static absl::StatusOr<std::unique_ptr<WhisperDecoder>> Create(
      LiteRtRunner* absl_nonnull runner,
      absl::string_view signature_name = "decode",
      size_t num_non_state_inputs = 2, size_t num_non_state_outputs = 1,
      std::vector<int> decode_start_token_ids = {},
      Options options = {});

  ~WhisperDecoder() override = default;

  absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> Decode(
      std::vector<::litert::TensorBuffer>& encoder_outputs,
      std::optional<DecodeRange> decode_range) override;

  using Decoder::Decode;

  const Options& options() const { return options_; }
  using StatefulDecoder::decode_start_token_ids;

 private:
  WhisperDecoder(std::unique_ptr<StatefulLiteRtRunner> stateful_runner,
                 size_t token_input_index, size_t num_logits_per_token,
                 std::vector<int> decode_start_token_ids,
                 Options options);

  // Selects next token from logits according to Whisper constraints.
  int32_t SelectNextToken(absl::Span<const float> logits, int step,
                          int min_timestamp_token_id);

  const Options options_;
};

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_SPEECH2TEXT_WHISPER_DECODER_H_

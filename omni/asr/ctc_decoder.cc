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

#include "omni/asr/ctc_decoder.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/asr/speech_recognizer.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<CtcDecoder>> CtcDecoder::Create(int vocab_size) {
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError("vocab_size must be positive");
  }
  return std::unique_ptr<CtcDecoder>(new CtcDecoder(vocab_size));
}

CtcDecoder::CtcDecoder(int vocab_size) : vocab_size_(vocab_size) {}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>> CtcDecoder::Decode(
    std::vector<::litert::TensorBuffer>& encoder_outputs) {
  if (encoder_outputs.empty()) {
    return std::vector<SpeechRecognizer::DecodedToken>();
  }

  LITERT_ASSIGN_OR_RETURN(auto packed_size, encoder_outputs[0].PackedSize());
  const size_t total_floats = packed_size / sizeof(float);
  if (total_floats == 0 || vocab_size_ <= 0) {
    return std::vector<SpeechRecognizer::DecodedToken>();
  }

  std::vector<float> logits(total_floats);
  LITERT_RETURN_IF_ERROR(
      encoder_outputs[0].Read<float>(absl::MakeSpan(logits)));

  // Decodes the model output (FloatArray logits) into an array of token IDs.
  // Performs CTC decoding (collapse repeats -> remove blanks) and text cleanup.
  const int num_frames = static_cast<int>(total_floats / vocab_size_);
  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  int previous_token_id = -1;
  for (int t = 0; t < num_frames; ++t) {
    const int start = t * vocab_size_;
    const int end = start + vocab_size_;
    auto max_it =
        std::max_element(logits.begin() + start, logits.begin() + end);
    int token_id =
        static_cast<int>(std::distance(logits.begin() + start, max_it));
    if (token_id == previous_token_id) {
      continue;
    }
    // Timestamp is the index of the token in the original list of logits
    // as the input of CTC has logits for every sample in the input audio.
    decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
        .token_id = token_id, .timestamp_ms = t});
    previous_token_id = token_id;
  }

  decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
      .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
      .timestamp_ms = num_frames});

  return decoded_tokens;
}

}  // namespace litert::omni::asr

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

#include "omni/speech2text/whisper_decoder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/speech2text/speech_recognizer.h"
#include "omni/speech2text/stateful_decoder.h"
#include "omni/speech2text/utils.h"

namespace litert::omni::asr {

absl::StatusOr<std::unique_ptr<WhisperDecoder>> WhisperDecoder::Create(
    LiteRtRunner* absl_nonnull runner, absl::string_view signature_name,
    size_t num_non_state_inputs, size_t num_non_state_outputs,
    std::vector<int> decode_start_token_ids, Options options) {
  if (runner == nullptr) {
    return absl::InvalidArgumentError("Runner cannot be null.");
  }
  if (options.max_decode_steps <= 0) {
    return absl::InvalidArgumentError("max_decode_steps must be positive.");
  }

  // If timestamps are enabled, ensure <|notimestamps|> is not in start tokens.
  if (options.enable_timestamps) {
    std::vector<int> filtered_starts;
    filtered_starts.reserve(decode_start_token_ids.size());
    for (int id : decode_start_token_ids) {
      if (id != kWhisperNoTimestampsTokenId) {
        filtered_starts.push_back(id);
      }
    }
    decode_start_token_ids = std::move(filtered_starts);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto context,
      InitRunnerContext(runner, signature_name, num_non_state_inputs,
                        num_non_state_outputs));

  return absl::WrapUnique(new WhisperDecoder(
      std::move(context.stateful_runner), context.token_input_index,
      context.num_logits_per_token, std::move(decode_start_token_ids),
      std::move(options)));
}

WhisperDecoder::WhisperDecoder(
    std::unique_ptr<StatefulLiteRtRunner> stateful_runner,
    size_t token_input_index, size_t num_logits_per_token,
    std::vector<int> decode_start_token_ids, Options options)
    : StatefulDecoder(std::move(stateful_runner), token_input_index,
                      num_logits_per_token, decode_start_token_ids,
                      kWhisperEndOfTextTokenId,
                      /*decode_skip_until_token_id=*/-1,
                      options.max_decode_steps),
      options_(std::move(options)) {}

int32_t WhisperDecoder::SelectNextToken(absl::Span<const float> logits,
                                        int step, int min_timestamp_token_id) {
  int32_t best_token = -1;
  float best_logit = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < logits.size(); ++i) {
    int tok = static_cast<int>(i);

    // Apply begin_suppress_tokens on the first step.
    if (step == 0 &&
        absl::c_linear_search(options_.begin_suppress_tokens, tok)) {
      continue;
    }

    // Apply suppress_tokens.
    if (absl::c_linear_search(options_.suppress_tokens, tok)) {
      continue;
    }

    if (!options_.enable_timestamps) {
      // If timestamps disabled, suppress all timestamp tokens.
      if (tok >= kWhisperTimestampBeginTokenId &&
          tok <= kWhisperTimestampEndTokenId) {
        continue;
      }
    } else {
      // If timestamps enabled, enforce monotonicity: timestamp tokens must
      // be >= min_timestamp_token_id.
      if (tok >= kWhisperTimestampBeginTokenId && tok < min_timestamp_token_id) {
        continue;
      }
    }

    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best_token = tok;
    }
  }

  return best_token;
}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>>
WhisperDecoder::Decode(std::vector<::litert::TensorBuffer>& encoder_outputs,
                       std::optional<DecodeRange> decode_range) {
  ABSL_RETURN_IF_ERROR(ResetAndInitStepInputs(encoder_outputs));

  int32_t current_token = -1;
  int min_timestamp_token_id = kWhisperTimestampBeginTokenId;
  const auto& starts = decode_start_token_ids();
  if (starts.empty()) {
    current_token = kWhisperStartOfTranscriptTokenId;
  } else {
    for (size_t i = 0; i < starts.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(absl::Span<const float> logits, Step(starts[i]));
      if (i + 1 == starts.size()) {
        current_token =
            SelectNextToken(logits, /*step=*/0, min_timestamp_token_id);
      }
    }
  }

  std::vector<SpeechRecognizer::DecodedToken> decoded_tokens;
  std::optional<int> current_timestamp_ms = std::nullopt;

  for (int step = 0; step < options_.max_decode_steps; ++step) {
    if (current_token < 0 || current_token == kWhisperEndOfTextTokenId) {
      break;
    }

    // Check if current_token is a timestamp token.
    if (options_.enable_timestamps &&
        current_token >= kWhisperTimestampBeginTokenId &&
        current_token <= kWhisperTimestampEndTokenId) {
      // Convert token to millisecond timestamp: (id - 50364) * 20 ms.
      int ts_ms = static_cast<int>(
          (current_token - kWhisperTimestampBeginTokenId) * 20);
      if (decode_range.has_value() &&
          ts_ms > static_cast<int>(decode_range->valid_fraction * 30000.0f)) {
        break;
      }
      current_timestamp_ms = ts_ms;
      min_timestamp_token_id = current_token;
      // Timestamp tokens are acoustic boundaries and omitted from text words.
    } else {
      if (TruncateOnTrailingRepetition(decoded_tokens, current_token)) {
        break;
      }
      int start_ms = decode_range.has_value()
          ? static_cast<int>(decode_range->start_fraction * 30000.0f)
          : 0;
      if (!current_timestamp_ms.has_value() ||
          *current_timestamp_ms >= start_ms) {
        int relative_ts_ms = current_timestamp_ms.has_value()
            ? *current_timestamp_ms - start_ms
            : -1;
        decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
            .token_id = current_token,
            .timestamp_ms = relative_ts_ms >= 0
                ? std::optional<int>(relative_ts_ms)
                : std::nullopt,
        });
      }
    }

    ABSL_ASSIGN_OR_RETURN(absl::Span<const float> logits, Step(current_token));
    current_token = SelectNextToken(logits, step + 1, min_timestamp_token_id);
  }

  if (options_.enable_timestamps) {
    int chunk_end_ms = decode_range.has_value()
        ? static_cast<int>(
              (decode_range->valid_fraction - decode_range->start_fraction) *
              30000.0f)
        : (current_timestamp_ms.has_value() ? *current_timestamp_ms : 30000);
    decoded_tokens.push_back(SpeechRecognizer::DecodedToken{
        .token_id = SpeechRecognizer::DecodedToken::kEndOfChunkTokenId,
        .timestamp_ms = std::max(0, chunk_end_ms),
    });
  }

  return decoded_tokens;
}

}  // namespace litert::omni::asr

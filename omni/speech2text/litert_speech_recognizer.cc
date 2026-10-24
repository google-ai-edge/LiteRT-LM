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

#include "omni/speech2text/litert_speech_recognizer.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/speech2text/audio_preprocessor.h"
#include "omni/speech2text/speech_recognizer.h"

namespace litert::omni::asr {
namespace {
constexpr absl::string_view kEncodeSignatureName = "encode";
constexpr int kCollapsedLeadingFramesThreshold = 35;
constexpr int kCollapsedOverlapFramesLimit = 25;
constexpr int kCollapsedTrailingFramesThreshold = 15;
}  // namespace

absl::StatusOr<std::unique_ptr<LiteRtSpeechRecognizer>>
LiteRtSpeechRecognizer::Create(
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    Stage<PreprocessedAudio>* absl_nonnull audio_preprocessor,
    std::unique_ptr<Decoder> absl_nonnull decoder) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          runner->CreateInputBuffers(kEncodeSignatureName));
  LITERT_RETURN_IF_ERROR(!inputs.empty());
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          runner->CreateOutputBuffers(kEncodeSignatureName));
  LITERT_RETURN_IF_ERROR(!outputs.empty());
  return std::unique_ptr<LiteRtSpeechRecognizer>(new LiteRtSpeechRecognizer(
      std::move(runner), audio_preprocessor, std::move(decoder),
      std::move(inputs), std::move(outputs)));
}

LiteRtSpeechRecognizer::LiteRtSpeechRecognizer(
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    Stage<PreprocessedAudio>* absl_nonnull audio_preprocessor,
    std::unique_ptr<Decoder> absl_nonnull decoder,
    std::vector<::litert::TensorBuffer> encode_input_buffers,
    std::vector<::litert::TensorBuffer> encode_output_buffers)
    : SpeechRecognizer(audio_preprocessor),
      runner_(std::move(runner)),
      decoder_(std::move(decoder)),
      encode_input_buffers_(std::move(encode_input_buffers)),
      encode_output_buffers_(std::move(encode_output_buffers)) {}

absl::Status LiteRtSpeechRecognizer::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto preprocessed_audio,
                        audio_preprocessor_.GetOutput());
  const auto& mel_features = preprocessed_audio.features;
  if (mel_features.empty()) {
    // Audio preprocessors emit an empty feature vector for silent chunks. Push
    // an end-of-chunk token without running the encoder or decoder so
    // downstream stages (Detokenizer and TextMerger) still advance by one
    // chunk for this schedule step.
    PushOutput({DecodedToken{
        .token_id = DecodedToken::kEndOfChunkTokenId,
        .is_warm_start = preprocessed_audio.is_warm_start,
    }});
    return absl::OkStatus();
  }
  LITERT_RETURN_IF_ERROR(
      encode_input_buffers_[0].Write<float>(absl::MakeConstSpan(mel_features)));
  LITERT_RETURN_IF_ERROR(runner_->Run(
      kEncodeSignatureName, encode_input_buffers_, encode_output_buffers_));
  ABSL_ASSIGN_OR_RETURN(auto decoded_tokens,
                        decoder_->Decode(encode_output_buffers_,
                                         preprocessed_audio.decode_range));
  if (preprocessed_audio.decode_range.has_value()) {
    ABSL_ASSIGN_OR_RETURN(decoded_tokens,
                          MaybeRetryWithCenteredFeatures(
                              preprocessed_audio, std::move(decoded_tokens)));
  }
  if (preprocessed_audio.is_warm_start) {
    if (!decoded_tokens.empty() &&
        decoded_tokens.back().token_id == DecodedToken::kEndOfChunkTokenId) {
      decoded_tokens.back().is_warm_start = true;
    } else {
      decoded_tokens.push_back(DecodedToken{
          .token_id = DecodedToken::kEndOfChunkTokenId, .is_warm_start = true});
    }
  }
  PushOutput(std::move(decoded_tokens));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<SpeechRecognizer::DecodedToken>>
LiteRtSpeechRecognizer::MaybeRetryWithCenteredFeatures(
    const PreprocessedAudio& preprocessed_audio,
    std::vector<DecodedToken> decoded_tokens) {
  if (!preprocessed_audio.decode_range.has_value() ||
      !preprocessed_audio.centered_features.has_value()) {
    return decoded_tokens;
  }

  const Decoder::DecodeRange& range = *preprocessed_audio.decode_range;
  const auto& retry_features = *preprocessed_audio.centered_features;
  const bool empty_chunk =
      decoded_tokens.size() == 1 &&
      decoded_tokens.front().token_id == DecodedToken::kEndOfChunkTokenId;
  const bool leading_collapsed =
      range.start_fraction == 0.0f &&
      (empty_chunk || (decoded_tokens.size() > 1 &&
                       decoded_tokens.front().timestamp_ms.value_or(0) >=
                           kCollapsedLeadingFramesThreshold));
  const bool shifted_tail_collapsed =
      range.start_fraction > 0.0f &&
      (empty_chunk ||
       (decoded_tokens.size() > 1 &&
        decoded_tokens.back().timestamp_ms.has_value() &&
        decoded_tokens[decoded_tokens.size() - 2].timestamp_ms.has_value() &&
        *decoded_tokens.back().timestamp_ms -
                *decoded_tokens[decoded_tokens.size() - 2].timestamp_ms >=
            kCollapsedTrailingFramesThreshold));
  // Primary features use unpadded 512-sample (32 ms) STFT windows, which can
  // occasionally cause encoders trained on centered 25 ms (400-sample) STFT
  // framing (such as Parakeet) to collapse the leading or trailing tokens on a
  // full valid window. Retrying with centered 25 ms STFT features matches the
  // training framing and recovers the dropped span.
  if (range.valid_fraction >= 1.0f &&
      (leading_collapsed || shifted_tail_collapsed) &&
      retry_features.size() == preprocessed_audio.features.size()) {
    LITERT_RETURN_IF_ERROR(encode_input_buffers_[0].Write<float>(
        absl::MakeConstSpan(retry_features)));
    LITERT_RETURN_IF_ERROR(runner_->Run(
        kEncodeSignatureName, encode_input_buffers_, encode_output_buffers_));
    ABSL_ASSIGN_OR_RETURN(auto retry_tokens,
                          decoder_->Decode(encode_output_buffers_,
                                           preprocessed_audio.decode_range));
    const bool recovered_leading =
        leading_collapsed && retry_tokens.size() > 1 &&
        retry_tokens.front().timestamp_ms.value_or(63) <
            kCollapsedOverlapFramesLimit;
    if (retry_tokens.size() > decoded_tokens.size() || recovered_leading) {
      decoded_tokens = std::move(retry_tokens);
    }
  }
  return decoded_tokens;
}

void LiteRtSpeechRecognizer::PushOutputForTesting(
    std::vector<DecodedToken> output) {
  PushOutput(std::move(output));
}

}  // namespace litert::omni::asr

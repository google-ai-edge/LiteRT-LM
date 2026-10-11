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

#include "omni/asr/litert_speech_recognizer.h"

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
#include "omni/asr/audio_preprocessor.h"
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {
namespace {
constexpr absl::string_view kEncodeSignatureName = "encode";
constexpr int kCollapsedLeadingFramesThreshold = 35;
constexpr int kCollapsedOverlapFramesLimit = 25;
constexpr int kShiftedChunkTrailingFrameLimit = 48;
}  // namespace

absl::StatusOr<std::unique_ptr<LiteRtSpeechRecognizer>>
LiteRtSpeechRecognizer::Create(
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
    std::unique_ptr<Decoder> absl_nonnull decoder,
    const AudioPreprocessor* absl_nullable chunk_info) {
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          runner->CreateInputBuffers(kEncodeSignatureName));
  LITERT_RETURN_IF_ERROR(!inputs.empty());
  LITERT_ASSIGN_OR_RETURN(auto outputs,
                          runner->CreateOutputBuffers(kEncodeSignatureName));
  LITERT_RETURN_IF_ERROR(!outputs.empty());
  return std::unique_ptr<LiteRtSpeechRecognizer>(new LiteRtSpeechRecognizer(
      std::move(runner), audio_preprocessor, std::move(decoder), chunk_info,
      std::move(inputs), std::move(outputs)));
}

LiteRtSpeechRecognizer::LiteRtSpeechRecognizer(
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
    std::unique_ptr<Decoder> absl_nonnull decoder,
    const AudioPreprocessor* absl_nullable chunk_info,
    std::vector<::litert::TensorBuffer> encode_input_buffers,
    std::vector<::litert::TensorBuffer> encode_output_buffers)
    : SpeechRecognizer(audio_preprocessor),
      runner_(std::move(runner)),
      decoder_(std::move(decoder)),
      chunk_info_(chunk_info),
      encode_input_buffers_(std::move(encode_input_buffers)),
      encode_output_buffers_(std::move(encode_output_buffers)) {}

void LiteRtSpeechRecognizer::ResetInternal() { prev_chunk_tail_empty_ = false; }

absl::Status LiteRtSpeechRecognizer::ScheduleInternal() {
  auto cleanup = absl::MakeCleanup([this]() { SetState(State::kIdle); });
  ABSL_ASSIGN_OR_RETURN(auto mel_features, audio_preprocessor_.GetOutput());
  if (mel_features.empty()) {
    // Audio preprocessors emit an empty feature vector for silent chunks. Push
    // an end-of-chunk token without running the encoder or decoder so
    // downstream stages (Detokenizer and TextMerger) still advance by one
    // chunk for this schedule step.
    prev_chunk_tail_empty_ = false;
    PushOutput({DecodedToken{.token_id = DecodedToken::kEndOfChunkTokenId}});
    return absl::OkStatus();
  }
  LITERT_RETURN_IF_ERROR(
      encode_input_buffers_[0].Write<float>(absl::MakeConstSpan(mel_features)));
  LITERT_RETURN_IF_ERROR(runner_->Run(
      kEncodeSignatureName, encode_input_buffers_, encode_output_buffers_));
  // Tell the decoder which part of the window to report tokens for: the final
  // chunk of a stream is either shifted back to fill the window (long stream)
  // or padded (short stream) by the preprocessor.
  Decoder::DecodeRange range;
  if (chunk_info_ != nullptr) {
    range.start_fraction = chunk_info_->last_chunk_start_fraction();
    range.valid_fraction = chunk_info_->last_chunk_valid_fraction();
  }
  decoder_->SetDecodeRange(range);
  ABSL_ASSIGN_OR_RETURN(auto decoded_tokens,
                        decoder_->Decode(encode_output_buffers_));
  const bool leading_collapsed =
      range.start_fraction == 0.0f && decoded_tokens.size() > 1 &&
      decoded_tokens.front().timestamp_ms.value_or(0) >=
          kCollapsedLeadingFramesThreshold;
  const bool shifted_tail_collapsed =
      range.start_fraction > 0.0f && decoded_tokens.size() > 1 &&
      decoded_tokens[decoded_tokens.size() - 2].timestamp_ms.value_or(63) <=
          kShiftedChunkTrailingFrameLimit;
  if (chunk_info_ != nullptr && range.valid_fraction >= 1.0f &&
      (decoded_tokens.size() <= 1 || leading_collapsed ||
       shifted_tail_collapsed ||
       (range.start_fraction == 0.0f && prev_chunk_tail_empty_))) {
    auto retry_features = chunk_info_->last_chunk_centered_features();
    if (retry_features.size() == mel_features.size()) {
      LITERT_RETURN_IF_ERROR(encode_input_buffers_[0].Write<float>(
          absl::MakeConstSpan(retry_features)));
      LITERT_RETURN_IF_ERROR(runner_->Run(
          kEncodeSignatureName, encode_input_buffers_, encode_output_buffers_));
      decoder_->SetDecodeRange(range);
      ABSL_ASSIGN_OR_RETURN(auto retry_tokens,
                            decoder_->Decode(encode_output_buffers_));
      const bool recovered_leading =
          leading_collapsed && retry_tokens.size() > 1 &&
          retry_tokens.front().timestamp_ms.value_or(63) <
              kCollapsedOverlapFramesLimit;
      if (retry_tokens.size() > decoded_tokens.size() || recovered_leading) {
        decoded_tokens = std::move(retry_tokens);
      }
    }
  }
  prev_chunk_tail_empty_ =
      range.start_fraction == 0.0f && range.valid_fraction >= 1.0f &&
      (decoded_tokens.size() <= 1 ||
       decoded_tokens[decoded_tokens.size() - 2].timestamp_ms.value_or(63) <=
           kCollapsedOverlapFramesLimit);
  PushOutput(std::move(decoded_tokens));
  return absl::OkStatus();
}

void LiteRtSpeechRecognizer::PushOutputForTesting(
    std::vector<DecodedToken> output) {
  PushOutput(std::move(output));
}

}  // namespace litert::omni::asr

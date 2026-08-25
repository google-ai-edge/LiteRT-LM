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
#include "omni/asr/speech_recognizer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::asr {
namespace {
constexpr absl::string_view kEncodeSignatureName = "encode";
}  // namespace

absl::StatusOr<std::unique_ptr<LiteRtSpeechRecognizer>>
LiteRtSpeechRecognizer::Create(
    std::unique_ptr<LiteRtRunner> absl_nonnull runner,
    Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
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
    Stage<std::vector<float>>* absl_nonnull audio_preprocessor,
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
  ABSL_ASSIGN_OR_RETURN(auto mel_features, audio_preprocessor_.GetOutput());
  LITERT_RETURN_IF_ERROR(
      encode_input_buffers_[0].Write<float>(absl::MakeConstSpan(mel_features)));
  LITERT_RETURN_IF_ERROR(runner_->Run(
      kEncodeSignatureName, encode_input_buffers_, encode_output_buffers_));
  ABSL_ASSIGN_OR_RETURN(auto decoded_tokens,
                        decoder_->Decode(encode_output_buffers_));
  PushOutput(std::move(decoded_tokens));
  return absl::OkStatus();
}

void LiteRtSpeechRecognizer::PushOutputForTesting(
    std::vector<DecodedToken> output) {
  PushOutput(std::move(output));
}

}  // namespace litert::omni::asr

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

#include "omni/tts/qwen3_tts/qwen3_latent_decoder_stage.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/base/stage.h"
#include "omni/tts/acoustic_predictor.h"
#include "omni/tts/latent_decoder.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<Qwen3LatentDecoderStage>>
Qwen3LatentDecoderStage::Create(
    Stage<AcousticOutput>* absl_nonnull acoustic_predictor) {
  return std::make_unique<Qwen3LatentDecoderStage>(acoustic_predictor);
}

Qwen3LatentDecoderStage::Qwen3LatentDecoderStage(
    Stage<AcousticOutput>* absl_nonnull acoustic_predictor)
    : LatentDecoder(acoustic_predictor) {}

absl::Status Qwen3LatentDecoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto acoustic = acoustic_predictor_.GetOutput();
  if (absl::IsNotFound(acoustic.status())) {
    return absl::OkStatus();
  } else if (!acoustic.ok()) {
    return acoustic.status();
  }

  LatentOutput out;
  out.codec_features = std::move(acoustic->codec_features);
  out.rvq_frames = std::move(acoustic->rvq_frames);
  PushOutput(std::move(out));
  return absl::OkStatus();
}

void Qwen3LatentDecoderStage::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  ClearOutputsThenSetState(State::kIdle);
}

}  // namespace litert::omni::tts

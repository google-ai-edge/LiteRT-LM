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

#include "omni/any2any/minicpm_o/minicpm_o_vocoder_stage.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_vocoder_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

// Short seam de-clicking crossfade cap (2 ms at 24 kHz) to prevent boundary
// clicks without superimposing independently phased HiFTGenerator harmonics.
constexpr size_t kMiniCpmOMaxSeamCrossfadeSamples = 48;

}  // namespace

MiniCpmOVocoderStage::MiniCpmOVocoderStage(
    std::unique_ptr<Any2AnyVocoderStage> absl_nonnull vocoder_stage)
    : vocoder_stage_(std::move(vocoder_stage)) {}

absl::StatusOr<std::unique_ptr<MiniCpmOVocoderStage>>
MiniCpmOVocoderStage::Create(
    Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage,
    const MiniCpmOModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner) {
  Any2AnyVocoderStage::Options options{
      .history_tokens = config.vocoder_history_tokens,
      .chunk_tokens = config.vocoder_chunk_tokens,
      .lookahead_tokens = config.vocoder_lookahead_tokens,
      .samples_per_token = config.vocoder_samples_per_token,
      .audio_output_sample_rate_hz = config.audio_output_sample_rate_hz,
      .generate_audio = config.generate_audio,
      .padding_mode = Any2AnyVocoderStage::PaddingMode::kConstant,
      .pad_token_id = minicpm_o::kTtsSilenceTokenId,
      .max_seam_crossfade_samples = kMiniCpmOMaxSeamCrossfadeSamples,
      .max_valid_code = 0,
      .allow_inferred_layout = false,
  };
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Any2AnyVocoderStage> vocoder_stage,
                        Any2AnyVocoderStage::Create(tts_stage, options,
                                                    std::move(vocoder_runner)));
  return absl::WrapUnique(new MiniCpmOVocoderStage(std::move(vocoder_stage)));
}

bool MiniCpmOVocoderStage::NeedScheduleInternal() const {
  return vocoder_stage_->NeedSchedule() || vocoder_stage_->HasOutput();
}

void MiniCpmOVocoderStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  vocoder_stage_->Reset();
}

absl::Status MiniCpmOVocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  absl::Status status = vocoder_stage_->Schedule();
  while (vocoder_stage_->HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(Output out, vocoder_stage_->GetOutput());
    PushOutput(std::move(out));
  }
  return status;
}

absl::Status MiniCpmOVocoderStage::FlushInternal() {
  absl::Status status = vocoder_stage_->Flush();
  while (vocoder_stage_->HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(Output out, vocoder_stage_->GetOutput());
    PushOutput(std::move(out));
  }
  return status;
}

}  // namespace litert::omni::any2any

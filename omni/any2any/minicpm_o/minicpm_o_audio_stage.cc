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

#include "omni/any2any/minicpm_o/minicpm_o_audio_stage.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/any2any/any2any_whisper_audio_stage.h"
#include "omni/any2any/io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"

namespace litert::omni::any2any {
namespace {

MiniCpmOEmbeddingSegment BuildAudioSegment(
    std::vector<EncodedAudioChunk> chunks) {
  MiniCpmOEmbeddingSegment segment;
  if (chunks.empty()) {
    return segment;
  }

  size_t total_tokens = 0;
  size_t total_floats = 0;
  for (const auto& chunk : chunks) {
    total_tokens += static_cast<size_t>(chunk.valid_tokens);
    total_floats += chunk.embeddings.size();
  }
  segment.token_ids.reserve(total_tokens + 2 * chunks.size());
  for (const auto& chunk : chunks) {
    segment.token_ids.push_back(minicpm_o::kAudioStartTokenId);
    segment.token_ids.insert(segment.token_ids.end(), chunk.valid_tokens,
                             minicpm_o::kAudioEmbedPlaceholderTokenId);
    segment.token_ids.push_back(minicpm_o::kAudioEndTokenId);
    segment.num_tokens += chunk.valid_tokens;
  }

  if (chunks.size() == 1) {
    segment.embeddings = std::move(chunks[0].embeddings);
  } else {
    segment.embeddings.reserve(total_floats);
    for (const auto& chunk : chunks) {
      segment.embeddings.insert(segment.embeddings.end(),
                                chunk.embeddings.begin(),
                                chunk.embeddings.end());
    }
  }
  return segment;
}

}  // namespace

MiniCpmOAudioStage::MiniCpmOAudioStage(
    std::unique_ptr<Any2AnyWhisperAudioStage> absl_nonnull audio_stage)
    : audio_stage_(std::move(audio_stage)) {}

MiniCpmOAudioStage::~MiniCpmOAudioStage() = default;

absl::StatusOr<std::unique_ptr<MiniCpmOAudioStage>> MiniCpmOAudioStage::Create(
    FanOutOutputQueue<OmniSession::Input, AudioInput>* absl_nonnull audio_queue,
    FanOutOutputQueue<OmniSession::Input, AudioInputMetadata>* absl_nonnull
        audio_input_metadata_queue,
    std::unique_ptr<LiteRtRunner> absl_nonnull audio_encoder_runner,
    const MiniCpmOModelConfig& config) {
  Any2AnyWhisperAudioStage::Options options{
      .llm_hidden_size = config.llm_hidden_size,
      .initial_input_sample_rate_hz = config.audio_input_sample_rate_hz,
      .target_sample_rate_hz = minicpm_o::kAudioInputSampleRateHz,
      .num_mel_bins = minicpm_o::kAudioMelBins,
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Any2AnyWhisperAudioStage> audio_stage,
      Any2AnyWhisperAudioStage::Create(audio_queue, audio_input_metadata_queue,
                                       std::move(audio_encoder_runner),
                                       std::move(options)));
  return absl::WrapUnique(new MiniCpmOAudioStage(std::move(audio_stage)));
}

bool MiniCpmOAudioStage::NeedScheduleInternal() const {
  return audio_stage_->NeedSchedule() || audio_stage_->HasOutput();
}

absl::Status MiniCpmOAudioStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  if (!NeedScheduleInternal()) {
    return absl::NotFoundError("Audio queues have no output.");
  }
  absl::Status schedule_status = absl::OkStatus();
  if (audio_stage_->NeedSchedule()) {
    schedule_status = audio_stage_->Schedule();
  }
  while (audio_stage_->HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<std::vector<EncodedAudioChunk>> encoded,
                          audio_stage_->GetOutput());
    PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
        .seqnum = encoded.seqnum,
        .item = BuildAudioSegment(std::move(encoded.item)),
    });
  }
  return schedule_status;
}

void MiniCpmOAudioStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  audio_stage_->Reset();
}

}  // namespace litert::omni::any2any

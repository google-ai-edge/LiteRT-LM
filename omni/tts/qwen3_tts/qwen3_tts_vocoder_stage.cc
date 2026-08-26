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

#include "omni/tts/qwen3_tts/qwen3_tts_vocoder_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/common.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<Qwen3TtsVocoderStage>>
Qwen3TtsVocoderStage::Create(
    Stage<Qwen3TtsLatentOutput>* absl_nonnull latent_decoder,
    const Qwen3TtsModelConfig& config,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = absl::WrapUnique(
      new Qwen3TtsVocoderStage(latent_decoder, config, std::move(resources)));

  LITERT_ASSIGN_OR_RETURN(stage->codec_model_,
                          stage->resources_->GetCompiledModel("codec"));

  LITERT_ASSIGN_OR_RETURN(stage->input_buffers_,
                          stage->codec_model_->CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(stage->output_buffers_,
                          stage->codec_model_->CreateOutputBuffers(0));

  LITERT_ASSIGN_OR_RETURN(size_t size_bytes, stage->input_buffers_[0].Size());
  stage->codec_chunk_ =
      size_bytes / (sizeof(int32_t) * qwen3_tts::kNumCodeGroups);
  if (stage->codec_chunk_ == 0) {
    return absl::InvalidArgumentError("codec_chunk cannot be zero");
  }
  ABSL_VLOG(2) << "Dynamically set codec_chunk_ = " << stage->codec_chunk_;

  LITERT_ASSIGN_OR_RETURN(size_t out_size_bytes,
                          stage->output_buffers_[0].Size());
  int total_samples = out_size_bytes / sizeof(float);
  stage->upsample_ = total_samples / stage->codec_chunk_;
  if (stage->codec_chunk_ == 0) {
    return absl::InvalidArgumentError("codec_chunk cannot be zero");
  }
  if (stage->upsample_ == 0) {
    return absl::InvalidArgumentError("upsample cannot be zero");
  }

  return stage;
}

absl::StatusOr<std::vector<float>> Qwen3TtsVocoderStage::DecodeChunk(
    const std::vector<std::vector<int>>& window_frames, int c,
    int valid_frames_count) {
  int chunk = codec_chunk_;
  std::vector<int32_t> buf(qwen3_tts::kNumCodeGroups * chunk, 0);
  int num_frames = std::min(chunk, static_cast<int>(window_frames.size()));
  for (int k = 0; k < num_frames; ++k) {
    const auto& frame = window_frames[k];
    for (int g = 0;
         g < qwen3_tts::kNumCodeGroups && g < static_cast<int>(frame.size());
         ++g) {
      buf[g * chunk + k] = frame[g];
    }
  }
  LITERT_RETURN_IF_ERROR(
      input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(buf)));

  LITERT_RETURN_IF_ERROR(codec_model_->Run(input_buffers_, output_buffers_));

  LITERT_ASSIGN_OR_RETURN(
      auto copy_wav, support::CopyFromTensorBuffer<float>(output_buffers_[0]));
  const float* wav_ptr = copy_wav.data();
  size_t total_samples = copy_wav.size();

  int slice_start = c * upsample_;
  int slice_end = valid_frames_count * upsample_;
  if (slice_end > static_cast<int>(total_samples)) {
    slice_end = static_cast<int>(total_samples);
  }

  std::vector<float> waveform;
  if (slice_start < slice_end) {
    waveform.assign(wav_ptr + slice_start, wav_ptr + slice_end);
  }
  return waveform;
}

absl::Status Qwen3TtsVocoderStage::ProcessPendingChunks(bool flush_remaining) {
  const int chunk = codec_chunk_;
  const int ctx = qwen3_tts::kVocoderOverlapContext;
  const int step_size = chunk - ctx;

  while (static_cast<int>(pending_frames_.size()) >= chunk) {
    std::vector<std::vector<int>> window_frames(
        pending_frames_.begin(), pending_frames_.begin() + chunk);
    ABSL_ASSIGN_OR_RETURN(
        auto pcm, DecodeChunk(window_frames, is_first_chunk_ ? 0 : ctx, chunk));
    is_first_chunk_ = false;

    pending_frames_.erase(pending_frames_.begin(),
                          pending_frames_.begin() + step_size);

    AudioOutput out;
    out.pcm_samples = std::move(pcm);
    out.sample_rate_hz = qwen3_tts::kAudioSampleRateHz;
    PushOutput(std::move(out));
  }

  if (flush_remaining && !pending_frames_.empty()) {
    int num_frames = static_cast<int>(pending_frames_.size());
    std::vector<std::vector<int>> window_frames = std::move(pending_frames_);
    pending_frames_.clear();
    while (static_cast<int>(window_frames.size()) < chunk) {
      window_frames.push_back(std::vector<int>(qwen3_tts::kNumCodeGroups, 0));
    }
    int c = is_first_chunk_ ? 0 : std::min(ctx, num_frames - 1);
    ABSL_ASSIGN_OR_RETURN(auto pcm, DecodeChunk(window_frames, c, num_frames));

    AudioOutput out;
    out.pcm_samples = std::move(pcm);
    out.sample_rate_hz = qwen3_tts::kAudioSampleRateHz;
    PushOutput(std::move(out));
  }
  return absl::OkStatus();
}

absl::Status Qwen3TtsVocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2) << "[TRACE] Starting Qwen3TtsVocoderStage::ScheduleInternal";

  auto latent_out = latent_decoder_.GetOutput();
  if (absl::IsNotFound(latent_out.status())) {
    return absl::OkStatus();
  } else if (!latent_out.ok()) {
    return latent_out.status();
  }

  pending_frames_.insert(pending_frames_.end(), latent_out->rvq_frames.begin(),
                         latent_out->rvq_frames.end());

  return ProcessPendingChunks(/*flush_remaining=*/false);
}

absl::Status Qwen3TtsVocoderStage::Flush() {
  {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kIdle) {
      return absl::FailedPreconditionError(
          "Flush() called while Schedule() is in progress.");
    }
    state_ = State::kRunning;
  }
  while (true) {
    auto latent_out = latent_decoder_.GetOutput();
    if (!latent_out.ok()) break;
    pending_frames_.insert(pending_frames_.end(),
                           latent_out->rvq_frames.begin(),
                           latent_out->rvq_frames.end());
  }

  absl::Status status = ProcessPendingChunks(/*flush_remaining=*/true);
  SetState(State::kIdle);
  return status;
}

void Qwen3TtsVocoderStage::ResetInternal() {
  pending_frames_.clear();
  is_first_chunk_ = true;
}

}  // namespace litert::omni::tts

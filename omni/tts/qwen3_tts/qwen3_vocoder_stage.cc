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

#include "omni/tts/qwen3_tts/qwen3_vocoder_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/tts/latent_decoder.h"
#include "omni/tts/qwen3_tts/common.h"
#include "omni/tts/qwen3_tts/qwen3_stage_options.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<Qwen3VocoderStage>> Qwen3VocoderStage::Create(
    Stage<LatentOutput>* absl_nonnull latent_decoder, Qwen3StageOptions options,
    std::shared_ptr<Environment> absl_nonnull env) {
  auto stage = absl::WrapUnique(new Qwen3VocoderStage(
      latent_decoder, std::move(options), std::move(env)));

  ABSL_ASSIGN_OR_RETURN(auto codec,
                        CreateCompiledModel(*stage->env_, stage->options_,
                                            stage->options_.codec_file,
                                            stage->options_.num_threads));
  stage->codec_model_ = std::move(codec);

  LITERT_ASSIGN_OR_RETURN(stage->input_buffers_,
                          stage->codec_model_->CreateInputBuffers(0));
  LITERT_ASSIGN_OR_RETURN(stage->output_buffers_,
                          stage->codec_model_->CreateOutputBuffers(0));

  LITERT_ASSIGN_OR_RETURN(size_t size_bytes, stage->input_buffers_[0].Size());
  stage->codec_chunk_ =
      size_bytes / (sizeof(int32_t) * qwen3_tts::kNumCodeGroups);
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

absl::StatusOr<std::vector<float>> Qwen3VocoderStage::DecodeCodes(
    const std::vector<std::vector<int>>& frames) {
  ABSL_VLOG(2) << "[TRACE] Qwen3VocoderStage::DecodeCodes frames.size="
               << frames.size();
  int num_frames = frames.size();
  if (num_frames == 0) return std::vector<float>{};

  int chunk = codec_chunk_;
  int ctx = 25;

  std::vector<float> waveform;
  int i = 0;

  while (i < num_frames) {
    int c = std::min({ctx, i, chunk - 1});
    int j = std::min(i + chunk - c, num_frames);
    int window_len = j - (i - c);

    std::vector<int32_t> buf(qwen3_tts::kNumCodeGroups * chunk, 0);
    for (int k = 0; k < window_len; ++k) {
      int frame_idx = (i - c) + k;
      const auto& frame = frames[frame_idx];
      for (int g = 0;
           g < qwen3_tts::kNumCodeGroups && g < static_cast<int>(frame.size());
           ++g) {
        buf[g * chunk + k] = frame[g];
      }
    }
    LITERT_RETURN_IF_ERROR(
        input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(buf)));

    LITERT_RETURN_IF_ERROR(codec_model_->Run(input_buffers_, output_buffers_));

    LITERT_ASSIGN_OR_RETURN(auto copy_wav, support::CopyFromTensorBuffer<float>(
                                               output_buffers_[0]));
    const float* wav_ptr = copy_wav.data();
    size_t total_samples = copy_wav.size();

    int slice_start = c * upsample_;
    int slice_end = window_len * upsample_;
    if (slice_end > static_cast<int>(total_samples)) {
      slice_end = static_cast<int>(total_samples);
    }
    for (int idx = slice_start; idx < slice_end; ++idx) {
      waveform.push_back(wav_ptr[idx]);
    }

    i = j;
  }

  return waveform;
}

absl::Status Qwen3VocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  auto latent = latent_decoder_.GetOutput();
  if (absl::IsNotFound(latent.status())) {
    return absl::OkStatus();
  } else if (!latent.ok()) {
    return latent.status();
  }

  pending_frames_.insert(pending_frames_.end(), latent->rvq_frames.begin(),
                         latent->rvq_frames.end());

  if (static_cast<int>(pending_frames_.size()) >= codec_chunk_) {
    ABSL_ASSIGN_OR_RETURN(auto pcm, DecodeCodes(pending_frames_));
    pending_frames_.clear();
    AudioOutput out;
    out.pcm_samples = std::move(pcm);
    out.sample_rate_hz = 24000;
    PushOutput(std::move(out));
  }
  return absl::OkStatus();
}

absl::Status Qwen3VocoderStage::Flush() {
  {
    absl::MutexLock lock(mutex_);
    if (state_ != State::kIdle) {
      return absl::FailedPreconditionError(
          "Flush() called while Schedule() is in progress.");
    }
    state_ = State::kRunning;
  }
  while (true) {
    auto latent_or = latent_decoder_.GetOutput();
    if (!latent_or.ok()) break;
    pending_frames_.insert(pending_frames_.end(), latent_or->rvq_frames.begin(),
                           latent_or->rvq_frames.end());
  }

  if (!pending_frames_.empty()) {
    ABSL_ASSIGN_OR_RETURN(auto pcm, DecodeCodes(pending_frames_));
    pending_frames_.clear();
    AudioOutput out;
    out.pcm_samples = std::move(pcm);
    out.sample_rate_hz = 24000;
    PushOutput(std::move(out));
  }
  SetState(State::kIdle);
  return absl::OkStatus();
}

void Qwen3VocoderStage::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  pending_frames_.clear();
  ClearOutputsThenSetState(State::kIdle);
}

}  // namespace litert::omni::tts

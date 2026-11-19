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

#include "omni/any2any/any2any_vocoder_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/io_types.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {

Any2AnyVocoderStage::Any2AnyVocoderStage(
    Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage, Options options,
    std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner)
    : tts_stage_(*tts_stage),
      options_(options),
      vocoder_runner_(std::move(vocoder_runner)) {}

absl::StatusOr<std::unique_ptr<Any2AnyVocoderStage>>
Any2AnyVocoderStage::Create(
    Stage<Any2AnyTtsOutput>* absl_nonnull tts_stage, Options options,
    std::unique_ptr<LiteRtRunner> absl_nullable vocoder_runner) {
  if (options.history_tokens < 0 || options.chunk_tokens <= 0 ||
      options.lookahead_tokens < 0 || options.samples_per_token <= 0 ||
      options.audio_output_sample_rate_hz <= 0) {
    return absl::InvalidArgumentError(
        "Any2AnyVocoderStage token window, sample count, and output sample "
        "rate must be valid.");
  }

  auto stage = absl::WrapUnique(new Any2AnyVocoderStage(
      tts_stage, options,
      options.generate_audio ? std::move(vocoder_runner) : nullptr));

  if (stage->options_.generate_audio) {
    if (stage->vocoder_runner_ == nullptr) {
      return absl::NotFoundError(
          "Any2AnyVocoderStage requires vocoder_runner when generate_audio is "
          "enabled.");
    }
    ABSL_ASSIGN_OR_RETURN(stage->input_buffers_,
                          stage->vocoder_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(stage->output_buffers_,
                          stage->vocoder_runner_->CreateOutputBuffers(""));
    if (stage->input_buffers_.empty() || stage->output_buffers_.empty()) {
      return absl::InternalError(
          "audio_vocoder model must have at least 1 input and 1 output "
          "buffer.");
    }

    LITERT_ASSIGN_OR_RETURN(const RankedTensorType in_type,
                            stage->input_buffers_[0].TensorType());
    if (in_type.ElementType() != ElementType::Int32) {
      return absl::InvalidArgumentError(
          "audio_vocoder input buffer element type must be Int32.");
    }
    LITERT_ASSIGN_OR_RETURN(const RankedTensorType out_type,
                            stage->output_buffers_[0].TensorType());
    if (out_type.ElementType() != ElementType::Float32) {
      return absl::InvalidArgumentError(
          "audio_vocoder output buffer element type must be Float32.");
    }

    LITERT_ASSIGN_OR_RETURN(size_t in_bytes,
                            stage->input_buffers_[0].PackedSize());
    LITERT_ASSIGN_OR_RETURN(size_t out_bytes,
                            stage->output_buffers_[0].PackedSize());

    const size_t window_tokens = static_cast<size_t>(
        stage->options_.history_tokens + stage->options_.chunk_tokens +
        stage->options_.lookahead_tokens);
    const size_t trimmed_tokens = static_cast<size_t>(
        stage->options_.history_tokens + stage->options_.chunk_tokens);
    const size_t expected_in_bytes = window_tokens * sizeof(int32_t);
    const size_t expected_trimmed_out_bytes =
        trimmed_tokens *
        static_cast<size_t>(stage->options_.samples_per_token) * sizeof(float);
    const size_t expected_full_out_bytes =
        window_tokens * static_cast<size_t>(stage->options_.samples_per_token) *
        sizeof(float);

    if (!stage->options_.allow_inferred_layout) {
      if (in_bytes != expected_in_bytes ||
          out_bytes != expected_trimmed_out_bytes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "audio_vocoder buffer packed sizes (input=", in_bytes,
            " bytes, output=", out_bytes,
            " bytes) do not match configured vocoder token window and PCM "
            "sample dimensions (expected input=",
            expected_in_bytes, " bytes, output=", expected_trimmed_out_bytes,
            " bytes)."));
      }
    } else if (in_bytes != expected_in_bytes ||
               (out_bytes != expected_trimmed_out_bytes &&
                out_bytes != expected_full_out_bytes)) {
      const size_t in_tokens = in_bytes / sizeof(int32_t);
      const size_t out_samples = out_bytes / sizeof(float);
      if (in_bytes == expected_in_bytes && out_bytes > 0 &&
          out_bytes % sizeof(float) == 0 && window_tokens > 0 &&
          out_samples % window_tokens == 0) {
        stage->options_.samples_per_token =
            static_cast<int>(out_samples / window_tokens);
      } else if (in_bytes == expected_in_bytes && out_bytes > 0 &&
                 out_bytes % sizeof(float) == 0 && trimmed_tokens > 0 &&
                 out_samples % trimmed_tokens == 0) {
        stage->options_.samples_per_token =
            static_cast<int>(out_samples / trimmed_tokens);
      } else if (in_bytes > 0 && in_bytes % sizeof(int32_t) == 0 &&
                 out_bytes > 0 && out_bytes % sizeof(float) == 0 &&
                 in_tokens > 0 && out_samples % in_tokens == 0) {
        stage->options_.history_tokens = 0;
        stage->options_.chunk_tokens = static_cast<int>(in_tokens);
        stage->options_.lookahead_tokens = 0;
        stage->options_.samples_per_token =
            static_cast<int>(out_samples / in_tokens);
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "audio_vocoder buffer packed sizes (input=", in_bytes,
            " bytes, output=", out_bytes,
            " bytes) do not match configured vocoder token window and PCM "
            "sample dimensions (expected input=",
            expected_in_bytes, " bytes, output=", expected_trimmed_out_bytes,
            " or ", expected_full_out_bytes, " bytes)."));
      }
    }

    stage->output_num_samples_ = out_bytes / sizeof(float);
  }

  return stage;
}

absl::StatusOr<std::vector<float>> Any2AnyVocoderStage::SynthesizeWaveform(
    const Any2AnyTtsOutput& tts_out) {
  if (!options_.generate_audio || vocoder_runner_ == nullptr ||
      tts_out.audio_codes.empty()) {
    return std::vector<float>{};
  }

  if (options_.max_valid_code > 0) {
    for (int code : tts_out.audio_codes) {
      if (code < 0 || code >= options_.max_valid_code) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Any2AnyVocoderStage received out-of-range audio codec token ID (",
            code, "); valid range is [0, ", options_.max_valid_code, ")."));
      }
    }
  }

  const int total_tokens = static_cast<int>(tts_out.audio_codes.size());
  const int window_tokens = options_.history_tokens + options_.chunk_tokens +
                            options_.lookahead_tokens;
  const size_t history_samples =
      static_cast<size_t>(options_.history_tokens) * options_.samples_per_token;

  std::vector<float> waveform;
  waveform.reserve(static_cast<size_t>(total_tokens) *
                   options_.samples_per_token);

  const int32_t first_code = static_cast<int32_t>(tts_out.audio_codes.front());
  const int32_t last_code = static_cast<int32_t>(tts_out.audio_codes.back());
  const bool edge_clamp = options_.padding_mode == PaddingMode::kEdgeClamp;
  int emitted_tokens = 0;
  while (emitted_tokens < total_tokens) {
    const int valid_chunk_tokens =
        std::min(options_.chunk_tokens, total_tokens - emitted_tokens);
    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          TensorBufferScopedLock::Create<int32_t>(
              input_buffers_[0], TensorBuffer::LockMode::kWrite));
      int32_t* window_codes = lock_and_addr.second;
      for (int i = 0; i < window_tokens; ++i) {
        const int src_idx = emitted_tokens - options_.history_tokens + i;
        if (src_idx < 0) {
          window_codes[i] = edge_clamp ? first_code : options_.pad_token_id;
        } else if (src_idx < total_tokens) {
          window_codes[i] = static_cast<int32_t>(tts_out.audio_codes[src_idx]);
        } else {
          window_codes[i] = edge_clamp ? last_code : options_.pad_token_id;
        }
      }
    }

    ABSL_RETURN_IF_ERROR(
        vocoder_runner_->Run("", input_buffers_, output_buffers_));

    {
      LITERT_ASSIGN_OR_RETURN(
          auto lock_and_addr,
          TensorBufferScopedLock::Create<const float>(
              output_buffers_[0], TensorBuffer::LockMode::kRead));
      const float* chunk_pcm = lock_and_addr.second;

      const size_t crossfade_samples =
          std::min(history_samples, options_.max_seam_crossfade_samples);
      if (emitted_tokens > 0 && crossfade_samples > 0 &&
          waveform.size() >= crossfade_samples &&
          output_num_samples_ >= history_samples) {
        const size_t waveform_seam_start = waveform.size() - crossfade_samples;
        const size_t chunk_seam_start = history_samples - crossfade_samples;
        for (size_t i = 0; i < crossfade_samples; ++i) {
          const float alpha = static_cast<float>(i + 1) /
                              static_cast<float>(crossfade_samples + 1);
          waveform[waveform_seam_start + i] =
              (1.0f - alpha) * waveform[waveform_seam_start + i] +
              alpha * chunk_pcm[chunk_seam_start + i];
        }
      }

      const size_t valid_new_samples =
          static_cast<size_t>(valid_chunk_tokens) * options_.samples_per_token;
      const size_t copy_start = std::min(history_samples, output_num_samples_);
      const size_t copy_end =
          std::min(history_samples + valid_new_samples, output_num_samples_);
      waveform.insert(waveform.end(), chunk_pcm + copy_start,
                      chunk_pcm + copy_end);
    }
    emitted_tokens += valid_chunk_tokens;
  }

  return waveform;
}

absl::Status Any2AnyVocoderStage::ProcessTtsOutput(
    const Any2AnyTtsOutput& tts_out) {
  if (tts_out.has_text && !tts_out.text.empty()) {
    PushOutput(TextOutput{.confirmed_text = tts_out.text});
  }

  if (options_.generate_audio && !tts_out.audio_codes.empty() &&
      vocoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::vector<float> pcm, SynthesizeWaveform(tts_out));
    if (!pcm.empty()) {
      PushOutput(AudioOutput{
          .pcm_samples = std::move(pcm),
          .sample_rate_hz = options_.audio_output_sample_rate_hz,
      });
    }
  }

  if (tts_out.is_text_final) {
    PushOutput(EndOfOutputTurn{});
  }
  return absl::OkStatus();
}

absl::Status Any2AnyVocoderStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  absl::StatusOr<Any2AnyTtsOutput> tts_out = tts_stage_.GetOutput();
  if (absl::IsNotFound(tts_out.status())) {
    return absl::OkStatus();
  }
  if (!tts_out.ok()) {
    return tts_out.status();
  }
  return ProcessTtsOutput(*tts_out);
}

absl::Status Any2AnyVocoderStage::FlushInternal() {
  while (true) {
    absl::StatusOr<Any2AnyTtsOutput> tts_out = tts_stage_.GetOutput();
    if (absl::IsNotFound(tts_out.status())) {
      break;
    }
    if (!tts_out.ok()) {
      return tts_out.status();
    }
    ABSL_RETURN_IF_ERROR(ProcessTtsOutput(*tts_out));
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::any2any

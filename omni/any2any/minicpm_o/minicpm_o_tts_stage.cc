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

#include "omni/any2any/minicpm_o/minicpm_o_tts_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_llm_stage.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

// Transposes frame-major mel features `[num_frames, num_bins]` into bin-major
// layout `[num_bins, num_frames]`.
std::vector<float> TransposeFrameMajorToBinMajor(
    absl::Span<const float> frame_major_mel, int num_frames, int num_bins) {
  std::vector<float> bin_major(static_cast<size_t>(num_bins) * num_frames,
                               0.0f);
  for (int f = 0; f < num_frames; ++f) {
    for (int b = 0; b < num_bins; ++b) {
      bin_major[b * num_frames + f] = frame_major_mel[f * num_bins + b];
    }
  }
  return bin_major;
}

// Selects the next speech token ID from `logits` using greedy argmax when
// `!config.do_sample || config.tts_temperature <= 0.0f`, or temperature-scaled
// top-k (`config.tts_top_k`) and nucleus (`config.tts_top_p`) sampling
// otherwise.
int SampleAudioCode(absl::Span<const float> logits,
                    const MiniCpmOModelConfig& config, std::mt19937_64& rng) {
  if (logits.empty()) {
    return 0;
  }
  if (!config.do_sample || config.tts_temperature <= 0.0f) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const double temp = std::max<double>(config.tts_temperature, 1e-6);
  const float max_logit = *std::max_element(logits.begin(), logits.end());

  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const double p = std::exp((logits[i] - max_logit) / temp);
    candidates.push_back({p, static_cast<int>(i)});
  }

  const int k =
      (config.tts_top_k > 0)
          ? std::min(config.tts_top_k, static_cast<int>(candidates.size()))
          : static_cast<int>(candidates.size());
  absl::c_partial_sort(
      candidates, candidates.begin() + k,
      [](const auto& a, const auto& b) { return a.first > b.first; });
  candidates.resize(k);

  double sum = 0.0;
  for (const auto& c : candidates) {
    sum += c.first;
  }
  if (sum <= 0.0) {
    return candidates.front().second;
  }

  if (config.tts_top_p > 0.0f && config.tts_top_p < 1.0f &&
      candidates.size() > 1) {
    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
      cumulative += candidates[keep].first / sum;
      if (cumulative >= config.tts_top_p) {
        ++keep;
        break;
      }
    }
    candidates.resize(std::max<size_t>(1, keep));
  }

  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const auto& c : candidates) {
    weights.push_back(c.first);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return candidates[dist(rng)].second;
}

}  // namespace

MiniCpmOTtsStage::MiniCpmOTtsStage(
    Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
    MiniCpmOModelConfig config,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner)
    : llm_stage_(*llm_stage),
      config_(std::move(config)),
      tts_projector_runner_(std::move(tts_projector_runner)),
      tts_decoder_runner_(std::move(tts_decoder_runner)) {
  rng_.seed(config_.seed.value_or(42));
}

absl::StatusOr<std::unique_ptr<MiniCpmOTtsStage>> MiniCpmOTtsStage::Create(
    Stage<MiniCpmOLlmOutput>* absl_nonnull llm_stage,
    const MiniCpmOModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_projector_runner,
    std::unique_ptr<LiteRtRunner> absl_nullable tts_decoder_runner) {
  if (config.llm_hidden_size <= 0 || config.tts_hidden_size <= 0 ||
      config.tts_vocab_size <= 0 || config.tts_num_mel_bins <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOTtsStage dimensions (llm_hidden_size, tts_hidden_size, "
        "tts_vocab_size, tts_num_mel_bins) must be positive.");
  }

  auto stage = absl::WrapUnique(
      new MiniCpmOTtsStage(llm_stage, config, std::move(tts_projector_runner),
                           std::move(tts_decoder_runner)));

  if (stage->tts_projector_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        stage->proj_input_buffers_,
        stage->tts_projector_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        stage->proj_output_buffers_,
        stage->tts_projector_runner_->CreateOutputBuffers(""));

    if (stage->proj_input_buffers_.size() != 1 ||
        stage->proj_output_buffers_.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector model must have exactly 1 input and 1 output buffer; "
          "got ",
          stage->proj_input_buffers_.size(), " inputs and ",
          stage->proj_output_buffers_.size(), " outputs."));
    }
    LITERT_ASSIGN_OR_RETURN(size_t in_bytes,
                            stage->proj_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t out_bytes,
                            stage->proj_output_buffers_[0].Size());
    const size_t expected_in_bytes =
        static_cast<size_t>(stage->config_.llm_hidden_size) * sizeof(float);
    const size_t expected_out_bytes =
        static_cast<size_t>(stage->config_.tts_hidden_size) * sizeof(float);
    if (in_bytes != expected_in_bytes || out_bytes != expected_out_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_projector buffer sizes (", in_bytes, " -> ", out_bytes,
          " bytes) do not match configured llm_hidden_size -> tts_hidden_size "
          "(",
          expected_in_bytes, " -> ", expected_out_bytes, " bytes)."));
    }
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOTtsStage requires tts_projector_runner when generate_audio is "
        "enabled.");
  }

  if (stage->tts_decoder_runner_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        stage->decoder_input_buffers_,
        stage->tts_decoder_runner_->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(
        stage->decoder_output_buffers_,
        stage->tts_decoder_runner_->CreateOutputBuffers(""));
    if (stage->decoder_input_buffers_.size() != 3 ||
        stage->decoder_output_buffers_.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder model must have 3 inputs (cond_emb, prev_audio_code, "
          "step_idx) and 2 outputs (code_logits, mel_frame); got ",
          stage->decoder_input_buffers_.size(), " inputs and ",
          stage->decoder_output_buffers_.size(), " outputs."));
    }

    LITERT_ASSIGN_OR_RETURN(size_t cond_bytes,
                            stage->decoder_input_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t code_bytes,
                            stage->decoder_input_buffers_[1].Size());
    LITERT_ASSIGN_OR_RETURN(size_t step_bytes,
                            stage->decoder_input_buffers_[2].Size());
    LITERT_ASSIGN_OR_RETURN(size_t logits_bytes,
                            stage->decoder_output_buffers_[0].Size());
    LITERT_ASSIGN_OR_RETURN(size_t mel_bytes,
                            stage->decoder_output_buffers_[1].Size());

    const size_t expected_cond_bytes =
        static_cast<size_t>(stage->config_.tts_hidden_size) * sizeof(float);
    const size_t expected_logits_bytes =
        static_cast<size_t>(stage->config_.tts_vocab_size) * sizeof(float);
    const size_t expected_mel_bytes =
        static_cast<size_t>(stage->config_.tts_num_mel_bins) * sizeof(float);

    if (cond_bytes != expected_cond_bytes || code_bytes != sizeof(int32_t) ||
        step_bytes != sizeof(int32_t) ||
        logits_bytes != expected_logits_bytes ||
        mel_bytes != expected_mel_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "tts_decoder buffer sizes do not match configured TTS dimensions: "
          "inputs=(",
          cond_bytes, ", ", code_bytes, ", ", step_bytes,
          ") bytes vs expected (", expected_cond_bytes, ", ", sizeof(int32_t),
          ", ", sizeof(int32_t), ") bytes; outputs=(", logits_bytes, ", ",
          mel_bytes, ") bytes vs expected (", expected_logits_bytes, ", ",
          expected_mel_bytes, ") bytes."));
    }
  } else if (stage->config_.generate_audio) {
    return absl::NotFoundError(
        "MiniCpmOTtsStage requires tts_decoder_runner when generate_audio is "
        "enabled.");
  }

  return stage;
}

absl::StatusOr<std::vector<float>> MiniCpmOTtsStage::ProjectHiddenStates(
    absl::Span<const float> llm_hidden, int num_tokens) {
  if (num_tokens <= 0 || llm_hidden.empty()) {
    return std::vector<float>{};
  }
  if (tts_projector_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "MiniCpmOTtsStage requires tts_projector_runner to project LLM hidden "
        "states.");
  }

  const size_t llm_dim = static_cast<size_t>(config_.llm_hidden_size);
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);
  const size_t expected_size = static_cast<size_t>(num_tokens) * llm_dim;
  if (llm_hidden.size() != expected_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOLlmOutput llm_hidden_states size (", llm_hidden.size(),
        ") does not match num_tokens * llm_hidden_size (", expected_size,
        ")."));
  }

  std::vector<float> projected(static_cast<size_t>(num_tokens) * tts_dim, 0.0f);
  for (int t = 0; t < num_tokens; ++t) {
    absl::Span<const float> token_hidden =
        llm_hidden.subspan(static_cast<size_t>(t) * llm_dim, llm_dim);
    LITERT_RETURN_IF_ERROR(proj_input_buffers_[0].Write<float>(token_hidden));
    ABSL_RETURN_IF_ERROR(
        tts_projector_runner_->Run("", proj_input_buffers_,
                                   proj_output_buffers_));
    LITERT_RETURN_IF_ERROR(
        proj_output_buffers_[0].Read<float>(absl::MakeSpan(projected).subspan(
            static_cast<size_t>(t) * tts_dim, tts_dim)));
  }
  return projected;
}

absl::Status MiniCpmOTtsStage::DecodeAudioCodesAndMelFramesAutoregressively(
    absl::Span<const float> cond_hidden, int num_tokens,
    MiniCpmOTtsOutput& out) {
  const size_t tts_dim = static_cast<size_t>(config_.tts_hidden_size);
  const int vocab_size = config_.tts_vocab_size;
  const int num_bins = config_.tts_num_mel_bins;
  const int max_steps = std::max(1, config_.max_tts_tokens);

  if (cond_hidden.size() != static_cast<size_t>(num_tokens) * tts_dim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Projected TTS conditioning size (", cond_hidden.size(),
        ") does not match num_tokens * tts_hidden_size (",
        static_cast<size_t>(num_tokens) * tts_dim, ")."));
  }

  std::vector<float> frame_major_mel;
  std::vector<float> logits(vocab_size, 0.0f);
  std::vector<float> step_mel(num_bins, 0.0f);
  int32_t prev_code_id = minicpm_o::kTtsBosTokenId;

  for (int step = 0; step < max_steps; ++step) {
    const int cond_idx = std::min(step, num_tokens - 1);
    absl::Span<const float> step_cond =
        cond_hidden.subspan(static_cast<size_t>(cond_idx) * tts_dim, tts_dim);
    const int32_t step_idx = static_cast<int32_t>(step);

    LITERT_RETURN_IF_ERROR(decoder_input_buffers_[0].Write<float>(step_cond));
    LITERT_RETURN_IF_ERROR(decoder_input_buffers_[1].Write<int32_t>(
        absl::MakeConstSpan(&prev_code_id, 1)));
    LITERT_RETURN_IF_ERROR(decoder_input_buffers_[2].Write<int32_t>(
        absl::MakeConstSpan(&step_idx, 1)));
    ABSL_RETURN_IF_ERROR(tts_decoder_runner_->Run("", decoder_input_buffers_,
                                                  decoder_output_buffers_));

    LITERT_ASSIGN_OR_RETURN(size_t logits_bytes,
                            decoder_output_buffers_[0].Size());
    if (logits_bytes != static_cast<size_t>(vocab_size) * sizeof(float)) {
      return absl::InternalError(absl::StrCat(
          "tts_decoder code_logits buffer size (", logits_bytes,
          " bytes) does not match tts_vocab_size (", vocab_size, " floats)."));
    }
    LITERT_RETURN_IF_ERROR(
        decoder_output_buffers_[0].Read<float>(absl::MakeSpan(logits)));

    const int code_id = SampleAudioCode(logits, config_, rng_);
    if (code_id == minicpm_o::kTtsEosTokenId) {
      break;
    }
    out.audio_codes.push_back(code_id);
    prev_code_id = static_cast<int32_t>(code_id);

    LITERT_ASSIGN_OR_RETURN(size_t mel_bytes,
                            decoder_output_buffers_[1].Size());
    if (mel_bytes != static_cast<size_t>(num_bins) * sizeof(float)) {
      return absl::InternalError(absl::StrCat(
          "tts_decoder mel_frame buffer size (", mel_bytes,
          " bytes) does not match tts_num_mel_bins (", num_bins, " floats)."));
    }
    LITERT_RETURN_IF_ERROR(
        decoder_output_buffers_[1].Read<float>(absl::MakeSpan(step_mel)));
    frame_major_mel.insert(frame_major_mel.end(), step_mel.begin(),
                           step_mel.end());
  }

  const int num_frames = static_cast<int>(frame_major_mel.size()) / num_bins;
  out.num_frames = num_frames;
  out.mel_features =
      TransposeFrameMajorToBinMajor(frame_major_mel, num_frames, num_bins);
  return absl::OkStatus();
}

absl::StatusOr<MiniCpmOTtsOutput> MiniCpmOTtsStage::PredictLogMelSpectrogram(
    const MiniCpmOLlmOutput& llm_out) {
  MiniCpmOTtsOutput out;
  out.text = llm_out.text;
  out.has_text = !llm_out.text.empty();
  out.is_text_final = llm_out.is_final;
  out.num_mel_bins = config_.tts_num_mel_bins;

  if (!llm_out.generate_audio || !config_.generate_audio ||
      llm_out.num_tokens <= 0 || llm_out.llm_hidden_states.empty()) {
    return out;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> cond_hidden,
      ProjectHiddenStates(llm_out.llm_hidden_states, llm_out.num_tokens));
  if (cond_hidden.empty()) {
    return out;
  }

  ABSL_RETURN_IF_ERROR(DecodeAudioCodesAndMelFramesAutoregressively(
      cond_hidden, llm_out.num_tokens, out));
  return out;
}

absl::Status MiniCpmOTtsStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  auto llm_out = llm_stage_.GetOutput();
  if (absl::IsNotFound(llm_out.status())) {
    return absl::OkStatus();
  }
  if (!llm_out.ok()) {
    return llm_out.status();
  }

  ABSL_ASSIGN_OR_RETURN(MiniCpmOTtsOutput tts_out,
                        PredictLogMelSpectrogram(*llm_out));
  PushOutput(std::move(tts_out));
  return absl::OkStatus();
}

void MiniCpmOTtsStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  rng_.seed(config_.seed.value_or(42));
}

}  // namespace litert::omni::any2any

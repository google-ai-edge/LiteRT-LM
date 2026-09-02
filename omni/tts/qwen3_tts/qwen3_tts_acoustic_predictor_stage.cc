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

#include "omni/tts/qwen3_tts/qwen3_tts_acoustic_predictor_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/base/stateful_litert_runner.h"
#include "omni/tts/qwen3_tts/common.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "support/util/convert_tensor_buffer.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<Qwen3TtsAcousticPredictorStage>>
Qwen3TtsAcousticPredictorStage::Create(
    Stage<Qwen3TtsFrontendOutput>* absl_nonnull text_frontend,
    const Qwen3TtsModelConfig& config,
    std::shared_ptr<ModelResources> absl_nonnull resources) {
  auto stage = absl::WrapUnique(new Qwen3TtsAcousticPredictorStage(
      text_frontend, config, std::move(resources)));

  // Retrieve LM Runner for Talker model.
  LITERT_ASSIGN_OR_RETURN(stage->talker_runner_,
                          stage->resources_->GetLmRunner("talker"));

  // Retrieve auxiliary compiled models.
  LITERT_ASSIGN_OR_RETURN(auto mtp_model,
                          stage->resources_->GetCompiledModel("mtp"));
  LITERT_ASSIGN_OR_RETURN(
      stage->codec_embedding_model_,
      stage->resources_->GetCompiledModel("codec_embedding"));
  LITERT_ASSIGN_OR_RETURN(stage->mtp_embedding_model_,
                          stage->resources_->GetCompiledModel("mtp_embedding"));

  // Pre-allocate embedding model TensorBuffers
  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_input_buffers_,
                          stage->codec_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->codec_emb_output_buffers_,
                          stage->codec_embedding_model_->CreateOutputBuffers());

  LITERT_ASSIGN_OR_RETURN(stage->mtp_emb_input_buffers_,
                          stage->mtp_embedding_model_->CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(stage->mtp_emb_output_buffers_,
                          stage->mtp_embedding_model_->CreateOutputBuffers());

  // Initialize MTP stateful runner
  stage->mtp_runner_raw_ = std::make_unique<LiteRtRunnerImpl>(mtp_model.get());
  LITERT_ASSIGN_OR_RETURN(
      stage->mtp_runner_,
      StatefulLiteRtRunnerImpl::Create(
          stage->mtp_runner_raw_.get(), /*signature_name=*/"",
          /*num_non_state_inputs=*/3, /*num_non_state_outputs=*/1));

  auto non_state_in_bufs = stage->mtp_runner_->GetNonStateInputBuffers();
  if (non_state_in_bufs.size() < 3) {
    return absl::InternalError(
        "MTP model missing required non-state input buffers.");
  }

  LITERT_ASSIGN_OR_RETURN(size_t mask_size_bytes, non_state_in_bufs[2].Size());
  if (mask_size_bytes > 0) {
    stage->mtp_cache_len_ = mask_size_bytes / sizeof(float);
  }

  return stage;
}

// Runs the Talker model prefill step on input prompt embeddings to populate
// the KV cache.
absl::Status Qwen3TtsAcousticPredictorStage::RunPrefill(
    const std::vector<float>& prefill, int p) {
  LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                        {1, p, qwen3_tts::kHiddenDim}));
  LITERT_RETURN_IF_ERROR(buf.Write<float>(
      absl::MakeConstSpan(prefill.data(), p * qwen3_tts::kHiddenDim)));
  LITERT_ASSIGN_OR_RETURN(auto inputs, CreateExecutorInputsWithAudio(buf));
  return talker_runner_->Prefill(inputs);
}

// Runs Multi-Token Predictor (MTP) model to generate remaining codebook tokens.
absl::StatusOr<std::vector<int>> Qwen3TtsAcousticPredictorStage::RunMtp(
    const std::vector<float>& hidden, int cb0) {
  LITERT_RETURN_IF_ERROR(mtp_runner_->Reset());

  std::vector<int> codes;
  codes.reserve(qwen3_tts::kNumCodeGroups - 1);

  for (int t = 0; t < qwen3_tts::kNumCodeGroups; ++t) {
    auto non_state_inputs = mtp_runner_->GetNonStateInputBuffers();
    if (non_state_inputs.size() < 3) {
      return absl::InternalError("MTP runner missing non-state input buffers.");
    }
    auto& embed_buf = non_state_inputs[0];
    auto& pos_buf = non_state_inputs[1];
    auto& mask_buf = non_state_inputs[2];

    std::vector<float> embed(qwen3_tts::kHiddenDim, 0.0f);
    if (t == 0) {
      embed = hidden;
    } else if (t == 1) {
      ABSL_ASSIGN_OR_RETURN(embed, EmbedCodecToken(cb0));
    } else {
      int head_idx = t - 2;
      int prev_code = codes[head_idx];
      int32_t global_id = head_idx * 2048 + prev_code;
      LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
          absl::MakeConstSpan(&global_id, 1)));
      LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(
          mtp_emb_input_buffers_, mtp_emb_output_buffers_));
      LITERT_RETURN_IF_ERROR(mtp_emb_output_buffers_[0].Read<float>(
          absl::MakeSpan(embed.data(), qwen3_tts::kHiddenDim)));
    }

    LITERT_RETURN_IF_ERROR(embed_buf.Write<float>(absl::MakeConstSpan(embed)));
    int32_t t_val = t;
    LITERT_RETURN_IF_ERROR(
        pos_buf.Write<int32_t>(absl::MakeConstSpan(&t_val, 1)));
    std::vector<float> mask(mtp_cache_len_, qwen3_tts::kNegInf);
    for (int j = 0; j <= t && j < mtp_cache_len_; ++j) mask[j] = 0.0f;
    LITERT_RETURN_IF_ERROR(mask_buf.Write<float>(absl::MakeConstSpan(mask)));

    LITERT_ASSIGN_OR_RETURN(
        auto step_outputs,
        mtp_runner_->Step(/*non_state_inputs=*/{}, /*auto_commit_state=*/true));
    if (step_outputs.empty()) {
      return absl::InternalError("MTP step returned empty outputs.");
    }

    if (t >= 1) {
      LITERT_ASSIGN_OR_RETURN(
          auto lock, TensorBufferScopedLock::Create<const float>(
                         step_outputs[0], TensorBuffer::LockMode::kRead));
      int head_idx = t - 1;
      const float* logits_ptr = lock.second + head_idx * 2048;
      const float* max_it = std::max_element(logits_ptr, logits_ptr + 2048);
      int picked_code = std::distance(logits_ptr, max_it);
      codes.push_back(picked_code);
    }
  }

  return codes;
}

// Embeds a single audio codec token ID using the codec embedding model.
absl::StatusOr<std::vector<float>>
Qwen3TtsAcousticPredictorStage::EmbedCodecToken(int code_id) {
  int32_t cid = code_id;
  LITERT_RETURN_IF_ERROR(
      codec_emb_input_buffers_[0].Write<int32_t>(absl::MakeConstSpan(&cid, 1)));
  LITERT_RETURN_IF_ERROR(codec_embedding_model_->Run(
      codec_emb_input_buffers_, codec_emb_output_buffers_));
  std::vector<float> vec(qwen3_tts::kHiddenDim);
  LITERT_RETURN_IF_ERROR(codec_emb_output_buffers_[0].Read<float>(
      absl::MakeSpan(vec.data(), qwen3_tts::kHiddenDim)));
  return vec;
}

// Embeds a sequence of MTP codebook token IDs using the MTP embedding model.
absl::StatusOr<std::vector<float>>
Qwen3TtsAcousticPredictorStage::EmbedMtpTokens(
    const std::vector<int>& mtp_codes) {
  std::vector<float> total_vec(qwen3_tts::kHiddenDim, 0.0f);
  for (size_t i = 0; i < mtp_codes.size(); ++i) {
    int32_t global_id = i * 2048 + mtp_codes[i];
    LITERT_RETURN_IF_ERROR(mtp_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(&global_id, 1)));
    LITERT_RETURN_IF_ERROR(mtp_embedding_model_->Run(mtp_emb_input_buffers_,
                                                     mtp_emb_output_buffers_));
    std::vector<float> vec(qwen3_tts::kHiddenDim);
    LITERT_RETURN_IF_ERROR(mtp_emb_output_buffers_[0].Read<float>(
        absl::MakeSpan(vec.data(), qwen3_tts::kHiddenDim)));
    for (int d = 0; d < qwen3_tts::kHiddenDim; ++d) {
      total_vec[d] += vec[d];
    }
  }
  return total_vec;
}

int Qwen3TtsAcousticPredictorStage::PickToken(const std::vector<float>& logits,
                                              bool do_sample) {
  if (!do_sample) {
    return std::distance(logits.begin(),
                         std::max_element(logits.begin(), logits.end()));
  }

  std::vector<double> probs(logits.size());
  double max_logit = -1e30;
  for (float val : logits) {
    if (val > max_logit) max_logit = val;
  }
  double temp = std::max(static_cast<double>(config_.temperature), 1e-6);

  double sum_exp = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] <= qwen3_tts::kNegInf / 2) {
      probs[i] = 0.0;
    } else {
      probs[i] = std::exp((logits[i] - max_logit) / temp);
      sum_exp += probs[i];
    }
  }

  if (sum_exp <= 0.0) {
    return 0;
  }

  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] /= sum_exp;
  }

  int top_k = std::min(config_.top_k, static_cast<int>(probs.size()));
  if (top_k > 0) {
    std::vector<std::pair<double, int>> pairs(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
      pairs[i] = {probs[i], static_cast<int>(i)};
    }
    absl::c_partial_sort(
        pairs, pairs.begin() + top_k,
        [](const auto& a, const auto& b) { return a.first > b.first; });

    double top_k_sum = 0.0;
    for (int i = 0; i < top_k; ++i) {
      top_k_sum += pairs[i].first;
    }
    for (int i = 0; i < top_k; ++i) {
      pairs[i].first /= top_k_sum;
    }

    std::vector<double> top_k_probs(top_k);
    for (int i = 0; i < top_k; ++i) {
      top_k_probs[i] = pairs[i].first;
    }
    std::discrete_distribution<int> dist(top_k_probs.begin(),
                                         top_k_probs.end());
    return pairs[dist(rng_)].second;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng_);
}

absl::Status Qwen3TtsAcousticPredictorStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  ABSL_VLOG(2)
      << "[TRACE] Starting Qwen3TtsAcousticPredictorStage::ScheduleInternal";

  auto frontend_out = text_frontend_.GetOutput();
  if (absl::IsNotFound(frontend_out.status())) {
    return absl::OkStatus();
  } else if (!frontend_out.ok()) {
    ABSL_LOG(ERROR) << "Frontend GetOutput error: " << frontend_out.status();
    return frontend_out.status();
  }
  const auto& frontend = *frontend_out;
  ABSL_VLOG(2) << "Acoustic stage got frontend output with prompt_len="
               << frontend.prompt_len;

  // 1. Reset Talker LM KV cache for new prompt
  LITERT_RETURN_IF_ERROR(talker_runner_->Reset());
  ABSL_VLOG(2) << "Acoustic stage reset talker_runner";

  // 2. Prefill prompt embeddings
  auto prefill_status =
      RunPrefill(frontend.prompt_embeddings, frontend.prompt_len);
  if (!prefill_status.ok()) {
    ABSL_LOG(ERROR) << "RunPrefill failed: " << prefill_status;
    return prefill_status;
  }
  ABSL_VLOG(2) << "Acoustic stage finished RunPrefill";

  std::vector<std::vector<int>> chunk_frames;
  std::vector<float> chunk_codec_features;

  std::vector<float> suppress(qwen3_tts::kCodecVocab, 0.0f);
  for (int i = 2048; i < qwen3_tts::kCodecVocab; ++i) {
    suppress[i] = qwen3_tts::kNegInf;
  }
  suppress[qwen3_tts::kCodecEos] = 0.0f;

  std::vector<float> cb0_logits(qwen3_tts::kCodecVocab);
  std::vector<float> hidden(qwen3_tts::kHiddenDim);
  absl::flat_hash_set<int> history;

  int total_generated_frames = 0;
  lm::ExecutorInputs next_inputs;

  while (total_generated_frames < config_.max_frames) {
    auto decode_res = talker_runner_->Decode(next_inputs);
    if (!decode_res.ok()) {
      ABSL_LOG(ERROR) << "talker_runner_->Decode failed at frame "
                      << total_generated_frames << ": " << decode_res.status();
      return decode_res.status();
    }
    auto decode_logits_buf = std::move(*decode_res);

    std::vector<float> raw_output(qwen3_tts::kCodecVocab +
                                  qwen3_tts::kHiddenDim);
    LITERT_RETURN_IF_ERROR(decode_logits_buf.Read<float>(
        absl::MakeSpan(raw_output.data(), raw_output.size())));
    std::copy(raw_output.begin(), raw_output.begin() + qwen3_tts::kCodecVocab,
              cb0_logits.begin());
    std::copy(raw_output.begin() + qwen3_tts::kCodecVocab, raw_output.end(),
              hidden.begin());

    std::vector<float> scores(qwen3_tts::kCodecVocab);
    for (int i = 0; i < qwen3_tts::kCodecVocab; ++i) {
      scores[i] = cb0_logits[i] + suppress[i];
    }

    if (total_generated_frames < 2) {
      scores[qwen3_tts::kCodecEos] = qwen3_tts::kNegInf;
    }

    for (int token : history) {
      if (scores[token] > 0) {
        scores[token] /= config_.repetition_penalty;
      } else {
        scores[token] *= config_.repetition_penalty;
      }
    }

    int cb0 = PickToken(scores, config_.do_sample);
    history.insert(cb0);

    ABSL_VLOG(2) << "[TRACE] Generated frame " << total_generated_frames
                 << " with cb0=" << cb0;
    if (cb0 == qwen3_tts::kCodecEos) break;

    ABSL_ASSIGN_OR_RETURN(auto mtp_codes, RunMtp(hidden, cb0));
    std::vector<int> frame;
    frame.reserve(qwen3_tts::kNumCodeGroups);
    frame.push_back(cb0);
    frame.insert(frame.end(), mtp_codes.begin(), mtp_codes.end());
    chunk_frames.push_back(std::move(frame));

    ABSL_ASSIGN_OR_RETURN(auto codec_vec, EmbedCodecToken(cb0));
    ABSL_ASSIGN_OR_RETURN(auto mtp_vec, EmbedMtpTokens(mtp_codes));

    std::vector<float> embed(qwen3_tts::kHiddenDim);
    for (int i = 0; i < qwen3_tts::kHiddenDim; ++i) {
      float val = codec_vec[i] + mtp_vec[i];
      embed[i] = val;
      chunk_codec_features.push_back(val);
    }

    int step = total_generated_frames;
    total_generated_frames++;

    if (step < frontend.trailing_len) {
      const float* tr_ptr =
          frontend.trailing_embeddings.data() + step * qwen3_tts::kHiddenDim;
      for (int i = 0; i < qwen3_tts::kHiddenDim; ++i) embed[i] += tr_ptr[i];
    } else {
      for (int i = 0; i < qwen3_tts::kHiddenDim; ++i)
        embed[i] += frontend.tts_pad_embedding[i];
    }

    if (static_cast<int>(chunk_frames.size()) >= qwen3_tts::kFrameChunkSize) {
      Qwen3TtsAcousticOutput out;
      out.rvq_frames = std::move(chunk_frames);
      out.codec_features = std::move(chunk_codec_features);
      PushOutput(std::move(out));
      chunk_frames.clear();
      chunk_codec_features.clear();
    }

    LITERT_ASSIGN_OR_RETURN(auto buf, support::CreateTensorBuffer<float>(
                                          {1, 1, qwen3_tts::kHiddenDim}));
    LITERT_RETURN_IF_ERROR(
        buf.Write<float>(absl::MakeConstSpan(embed.data(), embed.size())));
    LITERT_ASSIGN_OR_RETURN(next_inputs, CreateExecutorInputsWithAudio(buf));
  }

  if (!chunk_frames.empty()) {
    Qwen3TtsAcousticOutput out;
    out.rvq_frames = std::move(chunk_frames);
    out.codec_features = std::move(chunk_codec_features);
    PushOutput(std::move(out));
  }

  return absl::OkStatus();
}

void Qwen3TtsAcousticPredictorStage::Reset() {
  WaitForStateThenSetState(State::kIdle, State::kRunning);
  ClearOutputsThenSetState(State::kIdle);
}

}  // namespace litert::omni::tts

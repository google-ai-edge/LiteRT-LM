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

#include "omni/any2any/qwen_omni/qwen_omni_frontend_stage.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "omni/any2any/io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_io_types.h"
#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"
#include "omni/any2any/qwen_omni/qwen_omni_text_stage.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

// Reserves capacity for `extra` appended elements while preserving geometric
// growth across repeated chunks, avoiding both repeated intermediate
// reallocations within a multi-token loop and exact-fit O(N^2) reallocations
// across chunks.
template <typename T>
void ReserveGeometricForAppend(std::vector<T>& vec, size_t extra) {
  if (extra == 0 || extra > vec.max_size() - vec.size()) {
    return;
  }
  const size_t needed = vec.size() + extra;
  if (needed > vec.capacity()) {
    const size_t doubled =
        vec.capacity() <= std::numeric_limits<size_t>::max() / 2
            ? vec.capacity() * 2
            : needed;
    vec.reserve(std::max(doubled, needed));
  }
}

}  // namespace

QwenOmniFrontendStage::QwenOmniFrontendStage(
    QwenOmniTextStage* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<QwenOmniEmbeddingSegment>>* absl_nullable audio_stage,
    SingleThreadedStageWithDeque<
        Sequenced<QwenOmniEmbeddingSegment>>* absl_nullable image_stage,
    const QwenOmniModelConfig& config)
    : text_stage_(*text_stage),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      generate_audio_(config.generate_audio) {
  fan_in_.AddInputStage(text_stage);
  if (audio_stage != nullptr) {
    fan_in_.AddInputStage(audio_stage);
  }
  if (image_stage != nullptr) {
    fan_in_.AddInputStage(image_stage);
  }
}

bool QwenOmniFrontendStage::NeedScheduleInternal() const {
  return fan_in_.NeedSchedule() || fan_in_.HasOutput();
}

absl::Status QwenOmniFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (fan_in_.NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(fan_in_.Schedule());
  }

  bool processed_any = false;
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(QwenOmniEmbeddingSegment segment,
                          fan_in_.GetOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(segment));
    processed_any = true;
  }

  if (!processed_any && !this->HasOutput()) {
    return absl::NotFoundError(
        "QwenOmniFrontendStage has no ready sequence items.");
  }
  return absl::OkStatus();
}

absl::Status QwenOmniFrontendStage::FlushInternal() {
  ABSL_RETURN_IF_ERROR(fan_in_.Flush());
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(QwenOmniEmbeddingSegment segment,
                          fan_in_.GetOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(segment));
  }
  if (in_turn_) {
    ABSL_RETURN_IF_ERROR(CloseOpenTurn());
  }
  return absl::OkStatus();
}

void QwenOmniFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  fan_in_.Reset();
  is_first_turn_ = true;
  in_turn_ = false;
  ResetTurn();
}

absl::StatusOr<const QwenOmniEmbeddingSegment* absl_nonnull>
QwenOmniFrontendStage::GetOrEmbedBoundary(
    absl::string_view boundary_control_text,
    std::optional<QwenOmniEmbeddingSegment>& cache) {
  if (!cache.has_value()) {
    ABSL_ASSIGN_OR_RETURN(QwenOmniEmbeddingSegment boundary,
                          text_stage_.EmbedControlText(boundary_control_text));
    if (boundary.token_ids.size() != 2 ||
        boundary.embeddings.size() != static_cast<size_t>(2) * hidden_dim_) {
      return absl::InternalError(absl::StrCat(
          "QwenOmniFrontendStage expected 2 boundary tokens for ",
          boundary_control_text, ", got ", boundary.token_ids.size(), "."));
    }
    cache = std::move(boundary);
  }
  return &(*cache);
}

absl::StatusOr<const QwenOmniEmbeddingSegment* absl_nonnull>
QwenOmniFrontendStage::GetOrEmbedTurnSuffix() {
  if (!cached_turn_suffix_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        cached_turn_suffix_,
        text_stage_.EmbedControlText("<|im_end|>\n<|im_start|>assistant\n"));
  }
  return &(*cached_turn_suffix_);
}

absl::Status QwenOmniFrontendStage::EnsureTurnStarted() {
  if (in_turn_) {
    return absl::OkStatus();
  }
  if (is_first_turn_ && !config_.system_prompt.empty()) {
    const std::string prefix_text =
        absl::StrCat("<|im_start|>system\n", config_.system_prompt,
                     "<|im_end|>\n<|im_start|>user\n");
    ABSL_ASSIGN_OR_RETURN(QwenOmniEmbeddingSegment prefix,
                          text_stage_.EmbedControlText(prefix_text));
    if (prompt_embeddings_.empty() && token_ids_.empty()) {
      prompt_embeddings_ = std::move(prefix.embeddings);
      token_ids_ = std::move(prefix.token_ids);
    } else {
      AppendRawSegment(prefix);
    }
  } else {
    if (!cached_user_turn_prefix_.has_value()) {
      ABSL_ASSIGN_OR_RETURN(cached_user_turn_prefix_,
                            text_stage_.EmbedControlText("<|im_start|>user\n"));
    }
    AppendRawSegment(*cached_user_turn_prefix_);
  }
  is_first_turn_ = false;
  in_turn_ = true;
  return absl::OkStatus();
}

absl::Status QwenOmniFrontendStage::CloseOpenTurn() {
  ABSL_ASSIGN_OR_RETURN(const QwenOmniEmbeddingSegment* absl_nonnull suffix,
                        GetOrEmbedTurnSuffix());
  AppendRawSegment(*suffix);
  FlushTurn();
  in_turn_ = false;
  return absl::OkStatus();
}

absl::Status QwenOmniFrontendStage::AppendFramedModalitySegment(
    const QwenOmniEmbeddingSegment& segment, absl::string_view modality,
    int bos_token_id, int eos_token_id, int placeholder_token_id,
    absl::string_view boundary_control_text,
    std::optional<QwenOmniEmbeddingSegment>& boundary_cache) {
  if (hidden_dim_ <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "QwenOmniFrontendStage config.llm_hidden_size must be positive, got ",
        hidden_dim_, "."));
  }
  const size_t hidden_size = static_cast<size_t>(hidden_dim_);
  if (segment.embeddings.empty() ||
      segment.embeddings.size() % hidden_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("QwenOmniFrontendStage ", modality,
                     " segment embedding size (", segment.embeddings.size(),
                     ") must be a positive multiple of llm_hidden_size (",
                     hidden_dim_, ")."));
  }
  size_t placeholder_count = 0;
  for (int tok : segment.token_ids) {
    if (tok == placeholder_token_id) {
      ++placeholder_count;
    } else if (tok != bos_token_id && tok != eos_token_id) {
      return absl::InvalidArgumentError(
          absl::StrCat("QwenOmniFrontendStage ", modality,
                       " segment contains unexpected token ID (", tok, ")."));
    }
  }
  const size_t total_emb_tokens = segment.embeddings.size() / hidden_size;
  if (placeholder_count != total_emb_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "QwenOmniFrontendStage ", modality, " segment token count (",
        placeholder_count, ") does not match embedding token count (",
        total_emb_tokens, ")."));
  }

  ABSL_ASSIGN_OR_RETURN(
      const QwenOmniEmbeddingSegment* absl_nonnull boundary,
      GetOrEmbedBoundary(boundary_control_text, boundary_cache));
  const absl::Span<const float> bos_emb =
      absl::MakeConstSpan(boundary->embeddings).subspan(0, hidden_size);
  const absl::Span<const float> eos_emb =
      absl::MakeConstSpan(boundary->embeddings)
          .subspan(hidden_size, hidden_size);

  ABSL_RETURN_IF_ERROR(EnsureTurnStarted());

  const size_t added_tokens = segment.token_ids.size();
  ReserveGeometricForAppend(prompt_embeddings_, added_tokens * hidden_size);
  ReserveGeometricForAppend(token_ids_, added_tokens);

  size_t emb_offset = 0;
  for (int tok : segment.token_ids) {
    if (tok == bos_token_id) {
      prompt_embeddings_.insert(prompt_embeddings_.end(), bos_emb.begin(),
                                bos_emb.end());
      token_ids_.push_back(boundary->token_ids[0]);
    } else if (tok == eos_token_id) {
      prompt_embeddings_.insert(prompt_embeddings_.end(), eos_emb.begin(),
                                eos_emb.end());
      token_ids_.push_back(boundary->token_ids[1]);
    } else {
      prompt_embeddings_.insert(
          prompt_embeddings_.end(), segment.embeddings.begin() + emb_offset,
          segment.embeddings.begin() + emb_offset + hidden_size);
      token_ids_.push_back(placeholder_token_id);
      emb_offset += hidden_size;
    }
  }
  return absl::OkStatus();
}

absl::Status QwenOmniFrontendStage::ProcessSegment(
    const QwenOmniEmbeddingSegment& segment) {
  if (segment.is_turn_end) {
    if (!segment.token_ids.empty()) {
      ABSL_RETURN_IF_ERROR(EnsureTurnStarted());
      AppendRawSegment(segment);
      FlushTurn();
      in_turn_ = false;
    } else if (in_turn_) {
      ABSL_RETURN_IF_ERROR(CloseOpenTurn());
    }
    return absl::OkStatus();
  }

  if (segment.embeddings.empty() && segment.token_ids.empty()) {
    return absl::OkStatus();
  }

  if (segment.modality == QwenOmniEmbeddingSegment::Modality::kAudio) {
    return AppendFramedModalitySegment(
        segment, "audio", config_.audio_bos_token_id,
        config_.audio_eos_token_id, config_.audio_token_id,
        "<|audio_bos|><|audio_eos|>", cached_audio_boundary_);
  }

  if (segment.modality == QwenOmniEmbeddingSegment::Modality::kVision) {
    return AppendFramedModalitySegment(
        segment, "vision", config_.vision_bos_token_id,
        config_.vision_eos_token_id, config_.image_token_id,
        "<|vision_bos|><|vision_eos|>", cached_vision_boundary_);
  }

  ABSL_RETURN_IF_ERROR(EnsureTurnStarted());
  AppendRawSegment(segment);
  return absl::OkStatus();
}

void QwenOmniFrontendStage::AppendRawSegment(
    const QwenOmniEmbeddingSegment& segment) {
  ReserveGeometricForAppend(prompt_embeddings_, segment.embeddings.size());
  prompt_embeddings_.insert(prompt_embeddings_.end(),
                            segment.embeddings.begin(),
                            segment.embeddings.end());
  ReserveGeometricForAppend(token_ids_, segment.token_ids.size());
  token_ids_.insert(token_ids_.end(), segment.token_ids.begin(),
                    segment.token_ids.end());
}

void QwenOmniFrontendStage::FlushTurn() {
  if (!prompt_embeddings_.empty() || !token_ids_.empty()) {
    const int prompt_len = config_.llm_hidden_size > 0
                               ? static_cast<int>(prompt_embeddings_.size() /
                                                  config_.llm_hidden_size)
                               : static_cast<int>(token_ids_.size());
    this->PushOutput(Any2AnyFrontendOutput{
        .prompt_embeddings = std::move(prompt_embeddings_),
        .prompt_len = prompt_len,
        .token_ids = std::move(token_ids_),
        .generate_audio = generate_audio_,
    });
  }
  ResetTurn();
}

void QwenOmniFrontendStage::ResetTurn() {
  prompt_embeddings_.clear();
  token_ids_.clear();
}

}  // namespace litert::omni::any2any

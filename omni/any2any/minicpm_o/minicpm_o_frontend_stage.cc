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

#include "omni/any2any/minicpm_o/minicpm_o_frontend_stage.h"

#include <cstddef>
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
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/any2any/minicpm_o/minicpm_o_text_stage.h"
#include "omni/base/fan_in_stage.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"

namespace litert::omni::any2any {
namespace {

bool IsAudioSegment(const MiniCpmOEmbeddingSegment& segment) {
  for (int tok : segment.token_ids) {
    if (tok == minicpm_o::kAudioStartTokenId ||
        tok == minicpm_o::kAudioEndTokenId ||
        tok == minicpm_o::kAudioEmbedPlaceholderTokenId) {
      return true;
    }
  }
  return false;
}

bool IsImageSegment(const MiniCpmOEmbeddingSegment& segment) {
  for (int tok : segment.token_ids) {
    if (tok == minicpm_o::kImageStartTokenId ||
        tok == minicpm_o::kImageEndTokenId ||
        tok == minicpm_o::kSliceStartTokenId ||
        tok == minicpm_o::kSliceEndTokenId ||
        tok == minicpm_o::kImageEmbedPlaceholderTokenId) {
      return true;
    }
  }
  return false;
}

}  // namespace

MiniCpmOFrontendStage::MiniCpmOFrontendStage(
    MiniCpmOTextStage* absl_nonnull text_stage,
    SingleThreadedStageWithDeque<
        Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable audio_stage,
    SingleThreadedStageWithDeque<
        Sequenced<MiniCpmOEmbeddingSegment>>* absl_nullable image_stage,
    const MiniCpmOModelConfig& config)
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

MiniCpmOFrontendStage::~MiniCpmOFrontendStage() = default;

bool MiniCpmOFrontendStage::NeedScheduleInternal() const {
  return fan_in_.NeedSchedule() || fan_in_.HasOutput();
}

absl::Status MiniCpmOFrontendStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (fan_in_.NeedSchedule()) {
    ABSL_RETURN_IF_ERROR(fan_in_.Schedule());
  }

  bool processed_any = false;
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(const MiniCpmOEmbeddingSegment* segment,
                          fan_in_.PeekOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(*segment));
    ABSL_RETURN_IF_ERROR(fan_in_.GetOutput().status());
    processed_any = true;
  }

  if (!processed_any && !this->HasOutput()) {
    return absl::NotFoundError(
        "MiniCpmOFrontendStage has no ready sequence items.");
  }
  return absl::OkStatus();
}

absl::StatusOr<MiniCpmOEmbeddingSegment>
MiniCpmOFrontendStage::BuildTurnStartPrefix() {
  std::string prefix;
  if (is_first_turn_ && !config_.system_prompt.empty()) {
    absl::StrAppend(&prefix, "<|im_start|>system\n", config_.system_prompt,
                    "<|im_end|>\n");
  }
  absl::StrAppend(&prefix, "<|im_start|>user\n");
  return text_stage_.EmbedControlText(prefix);
}

absl::StatusOr<MiniCpmOEmbeddingSegment>
MiniCpmOFrontendStage::FrameAudioSegment(
    const MiniCpmOEmbeddingSegment& audio) {
  if (audio.embeddings.empty() ||
      audio.embeddings.size() % static_cast<size_t>(hidden_dim_) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("MiniCpmOFrontendStage audio segment embedding size (",
                     audio.embeddings.size(),
                     ") must be a positive multiple of llm_hidden_size (",
                     hidden_dim_, ")."));
  }
  size_t placeholder_count = 0;
  for (int tok : audio.token_ids) {
    if (tok == minicpm_o::kAudioEmbedPlaceholderTokenId) {
      ++placeholder_count;
    } else if (tok != minicpm_o::kAudioStartTokenId &&
               tok != minicpm_o::kAudioEndTokenId) {
      return absl::InvalidArgumentError(
          absl::StrCat("MiniCpmOFrontendStage audio segment contains "
                       "unexpected token ID (",
                       tok, ")."));
    }
  }
  const size_t total_audio_tokens =
      audio.embeddings.size() / static_cast<size_t>(hidden_dim_);
  if (placeholder_count != total_audio_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOFrontendStage audio segment placeholder token count (",
        placeholder_count, ") does not match audio embedding token count (",
        total_audio_tokens, ")."));
  }

  ABSL_ASSIGN_OR_RETURN(
      const MiniCpmOEmbeddingSegment boundary,
      text_stage_.EmbedControlText("<|audio_start|><|audio_end|>\n"));
  const int audio_start_id = boundary.token_ids[0];
  const int audio_end_id = boundary.token_ids[1];
  const absl::Span<const int> newline_tokens =
      absl::MakeConstSpan(boundary.token_ids).subspan(2);
  const absl::Span<const float> start_emb =
      absl::MakeConstSpan(boundary.embeddings).subspan(0, hidden_dim_);
  const absl::Span<const float> end_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(hidden_dim_, hidden_dim_);
  const absl::Span<const float> newline_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(2 * static_cast<size_t>(hidden_dim_));

  std::vector<float> framed_embeddings;
  std::vector<int> framed_token_ids;
  size_t emb_offset = 0;
  for (int tok : audio.token_ids) {
    if (tok == minicpm_o::kAudioStartTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), start_emb.begin(),
                               start_emb.end());
      framed_token_ids.push_back(audio_start_id);
    } else if (tok == minicpm_o::kAudioEndTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), end_emb.begin(),
                               end_emb.end());
      framed_token_ids.push_back(audio_end_id);
    } else {
      framed_embeddings.insert(
          framed_embeddings.end(), audio.embeddings.begin() + emb_offset,
          audio.embeddings.begin() + emb_offset + hidden_dim_);
      framed_token_ids.push_back(minicpm_o::kAudioEmbedPlaceholderTokenId);
      emb_offset += static_cast<size_t>(hidden_dim_);
    }
  }
  framed_embeddings.insert(framed_embeddings.end(), newline_emb.begin(),
                           newline_emb.end());
  framed_token_ids.insert(framed_token_ids.end(), newline_tokens.begin(),
                          newline_tokens.end());

  const int num_tokens = static_cast<int>(framed_token_ids.size());
  return MiniCpmOEmbeddingSegment{
      .embeddings = std::move(framed_embeddings),
      .token_ids = std::move(framed_token_ids),
      .num_tokens = num_tokens,
  };
}

absl::StatusOr<MiniCpmOEmbeddingSegment>
MiniCpmOFrontendStage::FrameImageSegment(const MiniCpmOEmbeddingSegment& image,
                                         int image_index) {
  if (image.embeddings.empty() ||
      image.embeddings.size() % static_cast<size_t>(hidden_dim_) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("MiniCpmOFrontendStage image segment embedding size (",
                     image.embeddings.size(),
                     ") must be a positive multiple of llm_hidden_size (",
                     hidden_dim_, ")."));
  }
  size_t placeholder_count = 0;
  for (int tok : image.token_ids) {
    if (tok == minicpm_o::kImageEmbedPlaceholderTokenId) {
      ++placeholder_count;
    } else if (tok != minicpm_o::kImageStartTokenId &&
               tok != minicpm_o::kImageEndTokenId &&
               tok != minicpm_o::kSliceStartTokenId &&
               tok != minicpm_o::kSliceEndTokenId &&
               tok != static_cast<int>('\n')) {
      return absl::InvalidArgumentError(
          absl::StrCat("MiniCpmOFrontendStage image segment contains "
                       "unexpected token ID (",
                       tok, ")."));
    }
  }
  const size_t total_image_tokens =
      image.embeddings.size() / static_cast<size_t>(hidden_dim_);
  if (placeholder_count != total_image_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOFrontendStage image segment placeholder token count (",
        placeholder_count, ") does not match image embedding token count (",
        total_image_tokens, ")."));
  }

  ABSL_ASSIGN_OR_RETURN(
      const MiniCpmOEmbeddingSegment boundary,
      text_stage_.EmbedControlText("<image></image><slice></slice>\n"));
  const int image_start_id = boundary.token_ids[0];
  const int image_end_id = boundary.token_ids[1];
  const int slice_start_id = boundary.token_ids[2];
  const int slice_end_id = boundary.token_ids[3];
  const absl::Span<const int> newline_tokens =
      absl::MakeConstSpan(boundary.token_ids).subspan(4);
  const absl::Span<const float> image_start_emb =
      absl::MakeConstSpan(boundary.embeddings).subspan(0, hidden_dim_);
  const absl::Span<const float> image_end_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(hidden_dim_, hidden_dim_);
  const absl::Span<const float> slice_start_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(2 * static_cast<size_t>(hidden_dim_), hidden_dim_);
  const absl::Span<const float> slice_end_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(3 * static_cast<size_t>(hidden_dim_), hidden_dim_);
  const absl::Span<const float> newline_emb =
      absl::MakeConstSpan(boundary.embeddings)
          .subspan(4 * static_cast<size_t>(hidden_dim_));

  ABSL_ASSIGN_OR_RETURN(const MiniCpmOEmbeddingSegment image_id_seg,
                        text_stage_.EmbedControlText(absl::StrCat(
                            "<image_id>", image_index, "</image_id>")));

  std::vector<float> framed_embeddings(image_id_seg.embeddings.begin(),
                                       image_id_seg.embeddings.end());
  std::vector<int> framed_token_ids(image_id_seg.token_ids.begin(),
                                    image_id_seg.token_ids.end());

  size_t emb_offset = 0;
  for (int tok : image.token_ids) {
    if (tok == minicpm_o::kImageStartTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), image_start_emb.begin(),
                               image_start_emb.end());
      framed_token_ids.push_back(image_start_id);
    } else if (tok == minicpm_o::kImageEndTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), image_end_emb.begin(),
                               image_end_emb.end());
      framed_token_ids.push_back(image_end_id);
    } else if (tok == minicpm_o::kSliceStartTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), slice_start_emb.begin(),
                               slice_start_emb.end());
      framed_token_ids.push_back(slice_start_id);
    } else if (tok == minicpm_o::kSliceEndTokenId) {
      framed_embeddings.insert(framed_embeddings.end(), slice_end_emb.begin(),
                               slice_end_emb.end());
      framed_token_ids.push_back(slice_end_id);
    } else if (tok == static_cast<int>('\n')) {
      framed_embeddings.insert(framed_embeddings.end(), newline_emb.begin(),
                               newline_emb.end());
      framed_token_ids.insert(framed_token_ids.end(), newline_tokens.begin(),
                              newline_tokens.end());
    } else {
      framed_embeddings.insert(
          framed_embeddings.end(), image.embeddings.begin() + emb_offset,
          image.embeddings.begin() + emb_offset + hidden_dim_);
      framed_token_ids.push_back(minicpm_o::kImageEmbedPlaceholderTokenId);
      emb_offset += static_cast<size_t>(hidden_dim_);
    }
  }
  framed_embeddings.insert(framed_embeddings.end(), newline_emb.begin(),
                           newline_emb.end());
  framed_token_ids.insert(framed_token_ids.end(), newline_tokens.begin(),
                          newline_tokens.end());

  const int num_tokens = static_cast<int>(framed_token_ids.size());
  return MiniCpmOEmbeddingSegment{
      .embeddings = std::move(framed_embeddings),
      .token_ids = std::move(framed_token_ids),
      .num_tokens = num_tokens,
  };
}

absl::Status MiniCpmOFrontendStage::ProcessSegment(
    const MiniCpmOEmbeddingSegment& segment) {
  if (segment.is_turn_end) {
    if (!segment.token_ids.empty()) {
      // `EndOfInputTurn`: open the turn if it had no content segments, append
      // the turn-end suffix, and emit the turn.
      if (!in_turn_) {
        ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment prefix,
                              BuildTurnStartPrefix());
        AppendRawSegment(prefix);
        is_first_turn_ = false;
      }
      AppendRawSegment(segment);
      FlushTurn();
      in_turn_ = false;
    } else if (in_turn_) {
      // `EndOfInput` while a turn is open: close and emit the open turn.
      ABSL_ASSIGN_OR_RETURN(
          MiniCpmOEmbeddingSegment suffix,
          text_stage_.EmbedControlText("<|im_end|>\n<|im_start|>assistant\n"));
      AppendRawSegment(suffix);
      FlushTurn();
      in_turn_ = false;
    }
    return absl::OkStatus();
  }

  if (segment.embeddings.empty() && segment.token_ids.empty()) {
    return absl::OkStatus();
  }

  if (IsAudioSegment(segment)) {
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment framed_audio,
                          FrameAudioSegment(segment));
    if (!in_turn_) {
      ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment prefix,
                            BuildTurnStartPrefix());
      AppendRawSegment(prefix);
      is_first_turn_ = false;
      in_turn_ = true;
    }
    AppendRawSegment(framed_audio);
    return absl::OkStatus();
  }

  if (IsImageSegment(segment)) {
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment framed_image,
                          FrameImageSegment(segment, current_image_index_));
    if (!in_turn_) {
      ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment prefix,
                            BuildTurnStartPrefix());
      AppendRawSegment(prefix);
      is_first_turn_ = false;
      in_turn_ = true;
    }
    AppendRawSegment(framed_image);
    ++current_image_index_;
    return absl::OkStatus();
  }

  if (!in_turn_) {
    ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment prefix,
                          BuildTurnStartPrefix());
    AppendRawSegment(prefix);
    is_first_turn_ = false;
    in_turn_ = true;
  }
  AppendRawSegment(segment);
  return absl::OkStatus();
}

void MiniCpmOFrontendStage::AppendRawSegment(
    const MiniCpmOEmbeddingSegment& segment) {
  prompt_embeddings_.insert(prompt_embeddings_.end(),
                            segment.embeddings.begin(),
                            segment.embeddings.end());
  token_ids_.insert(token_ids_.end(), segment.token_ids.begin(),
                    segment.token_ids.end());
}

absl::Status MiniCpmOFrontendStage::FlushInternal() {
  ABSL_RETURN_IF_ERROR(fan_in_.Flush());
  while (fan_in_.HasOutput()) {
    ABSL_ASSIGN_OR_RETURN(const MiniCpmOEmbeddingSegment* segment,
                          fan_in_.PeekOutput());
    ABSL_RETURN_IF_ERROR(ProcessSegment(*segment));
    ABSL_RETURN_IF_ERROR(fan_in_.GetOutput().status());
  }
  FlushTurn();
  in_turn_ = false;
  return absl::OkStatus();
}

void MiniCpmOFrontendStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  fan_in_.Reset();
  is_first_turn_ = true;
  in_turn_ = false;
  current_image_index_ = 0;
  ResetTurn();
}

void MiniCpmOFrontendStage::FlushTurn() {
  if (!prompt_embeddings_.empty() || !token_ids_.empty()) {
    const int prompt_len = config_.llm_hidden_size > 0
                               ? static_cast<int>(prompt_embeddings_.size() /
                                                  config_.llm_hidden_size)
                               : static_cast<int>(token_ids_.size());
    this->PushOutput(MiniCpmOFrontendOutput{
        .prompt_embeddings = std::move(prompt_embeddings_),
        .prompt_len = prompt_len,
        .token_ids = std::move(token_ids_),
        .generate_audio = generate_audio_,
    });
  }
  ResetTurn();
}

void MiniCpmOFrontendStage::ResetTurn() {
  prompt_embeddings_.clear();
  token_ids_.clear();
}

}  // namespace litert::omni::any2any

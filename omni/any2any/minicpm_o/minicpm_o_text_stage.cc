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

#include "omni/any2any/minicpm_o/minicpm_o_text_stage.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/any2any/minicpm_o/minicpm_o_io_types.h"
#include "omni/any2any/minicpm_o/minicpm_o_model_config.h"
#include "omni/base/fan_out_stage.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/sequenced.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

// Splits `text` around MiniCPM-o chat and modality control tokens (e.g.,
// `<|im_start|>`, `<|im_end|>`, `<image>`, `<|audio_start|>`) and maps each
// special token directly to its token ID while delegating normal text spans
// to `tokenizer.TextToTokenIds()`. This is necessary because
// `Tokenizer::TextToTokenIds()` treats special tokens in formatted prompts as
// literal characters (splitting them into subwords) unless tokenized
// separately via `TokenToId()` or fallback token IDs.
absl::StatusOr<std::vector<int>> TokenizeWithSpecialTokens(
    support::Tokenizer& tokenizer, absl::string_view text) {
  static constexpr absl::string_view kSpecialTokens[] = {
      "<|im_start|>", "<|im_end|>",      "<|endoftext|>", "<image>",
      "</image>",     "<slice>",         "</slice>",      "<image_id>",
      "</image_id>",  "<|audio_start|>", "<|audio_end|>", "<think>",
      "</think>",     "<tts_bos>",       "<tts_eos>",
  };

  std::vector<int> all_tokens;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next_special_pos = absl::string_view::npos;
    absl::string_view matched_token;

    for (absl::string_view token : kSpecialTokens) {
      size_t found = text.find(token, pos);
      if (found != absl::string_view::npos &&
          (next_special_pos == absl::string_view::npos ||
           found < next_special_pos)) {
        next_special_pos = found;
        matched_token = token;
      }
    }

    if (next_special_pos == absl::string_view::npos) {
      ABSL_ASSIGN_OR_RETURN(support::TokenIds part_ids,
                            tokenizer.TextToTokenIds(text.substr(pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
      break;
    }

    if (next_special_pos > pos) {
      ABSL_ASSIGN_OR_RETURN(
          support::TokenIds part_ids,
          tokenizer.TextToTokenIds(text.substr(pos, next_special_pos - pos)));
      all_tokens.insert(all_tokens.end(), part_ids.begin(), part_ids.end());
    }

    absl::StatusOr<int> special_id = tokenizer.TokenToId(matched_token);
    if (special_id.ok()) {
      all_tokens.push_back(*special_id);
    } else if (matched_token == "<|im_start|>") {
      all_tokens.push_back(minicpm_o::kImStartTokenId);
    } else if (matched_token == "<|im_end|>") {
      all_tokens.push_back(minicpm_o::kImEndTokenId);
    } else if (matched_token == "<|endoftext|>") {
      all_tokens.push_back(minicpm_o::kEosTokenId);
    } else if (matched_token == "<think>") {
      all_tokens.push_back(minicpm_o::kThinkStartTokenId);
    } else if (matched_token == "</think>") {
      all_tokens.push_back(minicpm_o::kThinkEndTokenId);
    } else if (matched_token == "<tts_bos>") {
      all_tokens.push_back(minicpm_o::kTtsBosTokenId);
    } else if (matched_token == "<tts_eos>") {
      all_tokens.push_back(minicpm_o::kTtsEosTokenId);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unmapped special token: ", matched_token));
    }
    pos = next_special_pos + matched_token.size();
  }
  return all_tokens;
}

}  // namespace

MiniCpmOTextStage::MiniCpmOTextStage(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    std::vector<TensorBuffer> text_emb_input_buffers,
    std::vector<TensorBuffer> text_emb_output_buffers,
    const MiniCpmOModelConfig& config)
    : text_queue_(*text_queue),
      end_of_turn_queue_(*end_of_turn_queue),
      config_(config),
      hidden_dim_(config.llm_hidden_size),
      tokenizer_(std::move(tokenizer)),
      text_embedder_runner_(std::move(text_embedder_runner)),
      text_emb_input_buffers_(std::move(text_emb_input_buffers)),
      text_emb_output_buffers_(std::move(text_emb_output_buffers)) {}

absl::StatusOr<std::unique_ptr<MiniCpmOTextStage>> MiniCpmOTextStage::Create(
    FanOutOutputQueue<OmniSession::Input, TextInput>* absl_nonnull text_queue,
    FanOutOutputQueue<OmniSession::Input, EndOfInputTurn>* absl_nonnull
        end_of_turn_queue,
    std::shared_ptr<support::Tokenizer> absl_nonnull tokenizer,
    std::unique_ptr<LiteRtRunner> absl_nonnull text_embedder_runner,
    const MiniCpmOModelConfig& config) {
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> input_buffers,
                        text_embedder_runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(std::vector<TensorBuffer> output_buffers,
                        text_embedder_runner->CreateOutputBuffers(""));
  if (input_buffers.empty() || output_buffers.empty()) {
    return absl::InvalidArgumentError(
        "MiniCpmOTextStage text_embedder_runner must provide non-empty input "
        "and output buffers.");
  }
  if (config.llm_hidden_size <= 0) {
    return absl::InvalidArgumentError(
        "MiniCpmOTextStage config.llm_hidden_size must be positive.");
  }
  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, input_buffers[0].Size());
  LITERT_ASSIGN_OR_RETURN(size_t out_bytes, output_buffers[0].Size());
  const size_t input_elements = in_bytes / sizeof(int32_t);
  const size_t expected_out_bytes =
      input_elements * static_cast<size_t>(config.llm_hidden_size) *
      sizeof(float);
  if (input_elements == 0 || out_bytes != expected_out_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MiniCpmOTextStage output buffer size (", out_bytes,
        " bytes) does not match input elements (", input_elements,
        ") * llm_hidden_size (", config.llm_hidden_size,
        ") * sizeof(float) (", expected_out_bytes, " bytes)."));
  }
  return absl::WrapUnique(new MiniCpmOTextStage(
      text_queue, end_of_turn_queue, std::move(tokenizer),
      std::move(text_embedder_runner), std::move(input_buffers),
      std::move(output_buffers), config));
}

absl::StatusOr<std::vector<float>> MiniCpmOTextStage::EmbedTokenIds(
    absl::Span<const int> token_ids) {
  if (token_ids.empty()) {
    return std::vector<float>{};
  }

  LITERT_ASSIGN_OR_RETURN(size_t in_bytes, text_emb_input_buffers_[0].Size());
  const int max_batch =
      std::max(1, static_cast<int>(in_bytes / sizeof(int32_t)));
  std::vector<float> embeddings(token_ids.size() * hidden_dim_);

  int offset = 0;
  while (offset < static_cast<int>(token_ids.size())) {
    const int count =
        std::min(max_batch, static_cast<int>(token_ids.size()) - offset);
    std::vector<int32_t> batch_ids(max_batch, 0);
    for (int i = 0; i < count; ++i) {
      batch_ids[i] = static_cast<int32_t>(token_ids[offset + i]);
    }
    LITERT_RETURN_IF_ERROR(text_emb_input_buffers_[0].Write<int32_t>(
        absl::MakeConstSpan(batch_ids)));
    ABSL_RETURN_IF_ERROR(text_embedder_runner_->Run(
        "", text_emb_input_buffers_, text_emb_output_buffers_));

    std::vector<float> batch_out(max_batch * hidden_dim_);
    LITERT_RETURN_IF_ERROR(
        text_emb_output_buffers_[0].Read<float>(absl::MakeSpan(batch_out)));
    std::copy(batch_out.begin(), batch_out.begin() + count * hidden_dim_,
              embeddings.begin() + offset * hidden_dim_);
    offset += count;
  }
  return embeddings;
}

absl::StatusOr<MiniCpmOEmbeddingSegment> MiniCpmOTextStage::ProcessText(
    absl::string_view text) {
  std::vector<int> token_ids;
  if (!in_turn_) {
    std::string prefix;
    if (is_first_turn_ && !config_.system_prompt.empty()) {
      absl::StrAppend(&prefix, "<|im_start|>system\n", config_.system_prompt,
                      "<|im_end|>\n");
    }
    absl::StrAppend(&prefix, "<|im_start|>user\n");
    is_first_turn_ = false;
    in_turn_ = true;
    ABSL_ASSIGN_OR_RETURN(token_ids,
                          TokenizeWithSpecialTokens(*tokenizer_, prefix));
  }

  if (!text.empty()) {
    ABSL_ASSIGN_OR_RETURN(support::TokenIds user_tokens,
                          tokenizer_->TextToTokenIds(text));
    token_ids.insert(token_ids.end(), user_tokens.begin(), user_tokens.end());
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings,
                        EmbedTokenIds(token_ids));

  const int num_tokens = static_cast<int>(token_ids.size());
  return MiniCpmOEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
  };
}

absl::StatusOr<MiniCpmOEmbeddingSegment> MiniCpmOTextStage::ProcessEndOfTurn() {
  std::string suffix;
  if (!in_turn_) {
    if (is_first_turn_ && !config_.system_prompt.empty()) {
      absl::StrAppend(&suffix, "<|im_start|>system\n", config_.system_prompt,
                      "<|im_end|>\n");
    }
    absl::StrAppend(&suffix, "<|im_start|>user\n");
    is_first_turn_ = false;
  }
  absl::StrAppend(&suffix, "<|im_end|>\n<|im_start|>assistant\n");
  in_turn_ = false;

  ABSL_ASSIGN_OR_RETURN(std::vector<int> token_ids,
                        TokenizeWithSpecialTokens(*tokenizer_, suffix));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> embeddings,
                        EmbedTokenIds(token_ids));

  const int num_tokens = static_cast<int>(token_ids.size());
  return MiniCpmOEmbeddingSegment{
      .embeddings = std::move(embeddings),
      .token_ids = std::move(token_ids),
      .num_tokens = num_tokens,
      .is_turn_end = true,
  };
}

bool MiniCpmOTextStage::NeedScheduleInternal() const {
  return !text_queue_.Empty() || !end_of_turn_queue_.Empty();
}

absl::Status MiniCpmOTextStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  if (text_queue_.Empty() && end_of_turn_queue_.Empty()) {
    return absl::NotFoundError("Text and end-of-turn queues have no output.");
  }

  std::vector<Sequenced<TextInput>> text_items;
  while (!text_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<TextInput> sequenced_input,
                          text_queue_.PopFront());
    text_items.push_back(std::move(sequenced_input));
  }

  std::vector<Sequenced<EndOfInputTurn>> eot_items;
  while (!end_of_turn_queue_.Empty()) {
    ABSL_ASSIGN_OR_RETURN(Sequenced<EndOfInputTurn> sequenced_eot,
                          end_of_turn_queue_.PopFront());
    eot_items.push_back(std::move(sequenced_eot));
  }

  size_t text_idx = 0;
  size_t eot_idx = 0;
  while (text_idx < text_items.size() || eot_idx < eot_items.size()) {
    const bool take_text =
        text_idx < text_items.size() &&
        (eot_idx >= eot_items.size() ||
         text_items[text_idx].seqnum < eot_items[eot_idx].seqnum);
    if (take_text) {
      ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment segment,
                            ProcessText(text_items[text_idx].item.text));
      this->PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
          .seqnum = text_items[text_idx].seqnum,
          .item = std::move(segment),
      });
      ++text_idx;
    } else {
      ABSL_ASSIGN_OR_RETURN(MiniCpmOEmbeddingSegment segment,
                            ProcessEndOfTurn());
      this->PushOutput(Sequenced<MiniCpmOEmbeddingSegment>{
          .seqnum = eot_items[eot_idx].seqnum,
          .item = std::move(segment),
      });
      ++eot_idx;
    }
  }
  return absl::OkStatus();
}

void MiniCpmOTextStage::ResetInternal() {
  SingleThreadedStageWithDeque::ResetInternal();
  is_first_turn_ = true;
  in_turn_ = false;
}

}  // namespace litert::omni::any2any
